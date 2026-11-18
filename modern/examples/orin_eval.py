"""NeuOS on Jetson Orin (Nano / AGX): profile Uncertainty and run the controller.

Run as root with the board's Python, e.g. ``sudo -n /path/venv/bin/python examples/orin_eval.py``:

    orin_eval.py baseline --out base.json             # snapshot the DVFS state (print + save)
    orin_eval.py profile --workload llm --out-dir R/llm --repeats 3
    orin_eval.py run --workload llm --profile-dir R/llm --jobs 30 --mode DEFAULT \\
        --variant original --period-factor 1.3 --out R/llm/default_original.jsonl
    orin_eval.py summary R/llm/*.jsonl                # markdown table rows

Workloads: ``llm`` = two llama-server tiers (full / approx URLs) through
adapters.openai_stream; ``torch`` = torchvision resnet18 (random weights, batch 1,
224x224, CUDA) through adapters.torch_layers with SVD low-rank twins.

The DVFS state found at start (or ``--baseline FILE``) is written back in a
``finally`` block and on SIGTERM / SIGINT.
"""
import argparse
import json
import math
import os
import signal
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuos_modern import (DryRunBackend, NeuOSController, NullEnergyMeter, OrinSysfsBackend,  # noqa: E402
                          SharedCohort, TierLadder, build_config_space, load_configs,
                          load_uncertainty, save_configs, subsample)
from neuos_modern.config import format_uncertainty  # noqa: E402
from neuos_modern.profile import _Recorder, compute_uncertainty, profile, write_csv  # noqa: E402

PROMPTS = [
    "Explain in simple terms how a CPU cache works.",
    "Write a short story about a robot that learns to paint.",
    "List some practical tips for saving energy at home.",
    "Describe the water cycle to a ten-year-old.",
    "What are the main differences between TCP and UDP?",
]


def board_name():
    try:
        model = Path("/proc/device-tree/model").read_text()
    except OSError:
        return "unknown"
    return "orin-nano" if "Nano" in model else "agx-orin" if "AGX Orin" in model else "orin"


# --- workloads -----------------------------------------------------------------
class Workload:
    """``job(ctl)`` runs one job through the controller API and returns its log;
    ``n_approximable`` stages per job can be approximated."""


def llm_workload(a):
    ladder = TierLadder("llm", [{"name": a.full_name, "url": a.full_url, "model": a.full_name},
                                {"name": a.approx_name, "url": a.approx_url, "model": a.approx_name}])
    from neuos_modern.adapters.openai_stream import stream_completion
    import requests
    session = requests.Session()
    w, k = Workload(), [0]
    extra = {"temperature": 0, "ignore_eos": True, "cache_prompt": False}

    def job(ctl):
        prompt = PROMPTS[k[0] % len(PROMPTS)]
        k[0] += 1
        _, log = stream_completion(ctl, ladder, prompt, a.max_tokens, a.chunk_tokens,
                                   session=session, extra_body=extra)
        return log
    w.job, w.n_approximable, w.info = job, 1, {"ladder": ladder.tiers, "max_tokens": a.max_tokens,
                                               "chunk_tokens": a.chunk_tokens, "body": extra}
    return w


class _AlwaysApprox:
    def before_stage(self, i, approximable=False):
        return approximable

    def after_stage(self, i):
        pass


def torch_workload(a):
    import torch
    import torchvision
    from neuos_modern.adapters.torch_layers import attach, lowrank_twins
    torch.manual_seed(0)
    model = torchvision.models.resnet18(weights=None).cuda().eval()
    twins = lowrank_twins(model, a.rank) if a.rank > 0 else {}
    x = torch.randn(1, 3, 224, 224, device="cuda")
    with torch.no_grad():
        for _ in range(5):
            ref = model(x)
        neu = attach(model, _Recorder(NullEnergyMeter(), time.perf_counter), twins)
        neu.active, neu._k = _AlwaysApprox(), 0     # warm the twins up, measure their error
        for _ in range(3):
            low = model(x)
        neu.active = None
    rel_err = float((low - ref).norm() / ref.norm())
    w = Workload()

    def job(ctl):
        neu.ctl = ctl
        with torch.no_grad(), neu.job():
            model(x)
        return dict(neu.last_log or {})
    for _ in range(3):              # also sets neu.n_stages to the observed stage count
        job(_Recorder(NullEnergyMeter(), time.perf_counter))
    w.job, w.n_approximable = job, len(twins)
    w.info = {"model": "resnet18 (random weights)", "input": [1, 3, 224, 224], "rank_ratio": a.rank,
              "twins": {n: t[0].out_channels for n, t in twins.items()}, "n_stages": neu.n_stages,
              "all_twins_rel_output_error": rel_err}
    return w


WORKLOADS = {"llm": llm_workload, "torch": torch_workload}


# --- helpers -------------------------------------------------------------------
def make_backend(a):
    if a.dry_run:
        return DryRunBackend()
    base = json.loads(Path(a.baseline).read_text()) if getattr(a, "baseline", None) else None
    return OrinSysfsBackend(baseline=base)


def make_energy(a):
    if a.dry_run:
        return NullEnergyMeter()
    from neuos_modern.energy import EnergyMeter
    return EnergyMeter()


def install_signal_handlers():
    def bye(signum, frame):
        raise SystemExit(128 + signum)      # unwinds through the finally blocks
    signal.signal(signal.SIGTERM, bye)
    signal.signal(signal.SIGINT, bye)


def print_state(backend, tag):
    if isinstance(backend, OrinSysfsBackend):
        print("[%s] dvfs state: %s" % (tag, json.dumps(backend.snapshot())), flush=True)


# --- subcommands ---------------------------------------------------------------
def cmd_baseline(a):
    snap = OrinSysfsBackend().snapshot()
    print(json.dumps(snap, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(snap))


def cmd_profile(a):
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    install_signal_handlers()
    backend = make_backend(a)
    energy = make_energy(a)
    try:
        if a.dry_run:
            s = OrinSysfsBackend().enumerate_states()
            emc = [r for r in a.emc if r in s["emc_freqs_khz"]]
        else:
            s = backend.enumerate_states()
            emc = backend.probe_emc_rates([r for r in a.emc if r in s["emc_freqs_khz"]])
        n = s["n_cpus"]
        cpu = subsample(s["cpu_freqs_khz"], a.cpu_levels)
        gpu = subsample(s["gpu_freqs_hz"], a.gpu_levels)
        cores = sorted({n if c == "all" else max(1, n // 2) for c in a.cores})
        configs = build_config_space(cpu, gpu, emc, cores, n)
        save_configs(out / "configs.txt", configs)
        grid = {"board": board_name(), "workload": a.workload, "cpu_khz": cpu, "gpu_hz": gpu,
                "emc_khz": emc, "emc_requested_khz": a.emc, "cores": cores, "n_cpus": n,
                "n_configs": len(configs), "repeats": a.repeats, "states": s,
                "baseline": getattr(backend, "baseline", None),
                "rails": getattr(energy, "rails", None)}
        print("grid: %d configs  cpu=%s cores=%s gpu=%s emc=%s (requested %s)"
              % (len(configs), cpu, cores, gpu, emc, a.emc), flush=True)
        w = WORKLOADS[a.workload](a)
        grid["workload_info"] = w.info
        energy.start()
        backend.apply(None, configs[-1])                    # warm latency at the max config
        walls = []
        for _ in range(a.warm_jobs):
            t0 = time.perf_counter()
            w.job(_Recorder(energy, time.perf_counter))
            walls.append((time.perf_counter() - t0) * 1e3)
        grid["max_config_wall_ms"] = statistics.median(walls)
        grid["max_config_wall_ms_all"] = walls
        print("max-config wall latency (ms): %s" % ["%.1f" % x for x in walls], flush=True)
        t0 = time.time()
        rows = profile(w.job, configs, backend, energy, a.repeats)
        grid["profile_seconds"] = time.time() - t0
    finally:
        energy.stop()
        backend.restore()
        print_state(backend, "after restore")
    write_csv(out / "profile.csv", rows)
    u = compute_uncertainty(rows)
    (out / "Uncertainty.txt").write_text(format_uncertainty(u))
    (out / "grid.json").write_text(json.dumps(grid, indent=1))
    print("Uncertainty (%d stages): %s" % (len(u), format_uncertainty(["%.4f" % v for v in u])))


def cmd_run(a):
    pdir = Path(a.profile_dir)
    grid = json.loads((pdir / "grid.json").read_text())
    configs = load_configs(pdir / "configs.txt")
    unc = load_uncertainty(pdir / "Uncertainty.txt")
    period = a.period_ms or a.period_factor * grid["max_config_wall_ms"]
    install_signal_handlers()
    backend = make_backend(a)
    energy = make_energy(a)
    cohort = SharedCohort(a.cohort_prefix)
    try:
        w = WORKLOADS[a.workload](a)
        ctl = NeuOSController(configs, unc, period, backend, cohort, mode=a.mode,
                              variant=a.variant, seed=a.seed, energy=energy)
        energy.start()
        print("period %.1f ms (%s), %d configs, %d stages of Uncertainty"
              % (period, "fixed" if a.period_ms else "%.2f x max-config wall %.1f ms"
                 % (a.period_factor, grid["max_config_wall_ms"]), len(configs), len(unc)), flush=True)
        with open(a.out, "a") as f:
            for j in range(a.jobs):
                t0 = time.perf_counter()
                log = w.job(ctl)
                wall = (time.perf_counter() - t0) * 1e3
                st = log["stages"]
                n_apx = sum(s["approximated"] for s in st)
                rec = {"job": j, "pid": os.getpid(), "mode": a.mode, "variant": a.variant,
                       "warmup": log["warmup"], "period_ms": period, "wall_ms": wall,
                       "latency_ms": log["latency_us"] / 1e3, "sched_us": log["sched_us"],
                       "energy_mj": log["energy_mj"], "met": wall <= period,
                       "n_stages": len(st), "n_approx": n_apx,
                       "approx_rate": n_apx / w.n_approximable if w.n_approximable else 0.0,
                       "slack": log["slack"], "priority": [s["priority"] for s in st][-1],
                       "a_star": [s["a_star"] for s in st][-1], "slot": log["slot"],
                       "cohort": cohort.read(),
                       "dvfs_write_errors": getattr(backend, "write_errors", 0),
                       "configs": [s["config"] for s in st],
                       "approx": [int(s["approximated"]) for s in st],
                       "stage_lat_us": [s["latency_us"] for s in st]}
                for k in ("tier", "tokens"):
                    if k in log:
                        rec[k] = log[k]
                f.write(json.dumps(rec) + "\n")
                f.flush()
                print("job %d: wall %.1f ms (%s), %.1f mJ, approx %d/%d, cfg %d..%d, prio %.4g"
                      % (j, wall, "met" if rec["met"] else "MISS", rec["energy_mj"], n_apx,
                         w.n_approximable, min(rec["configs"]), max(rec["configs"]), rec["priority"]),
                      flush=True)
    finally:
        energy.stop()
        backend.restore()
        cohort.close()
        print_state(backend, "after restore")


def summarize(path, include_warmup=False):
    recs = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    recs = [r for r in recs if include_warmup or not r["warmup"]]
    if not recs:
        return None
    lat = sorted(r["wall_ms"] for r in recs)
    cfgs = [c for r in recs for c in r["configs"]]
    return {"file": Path(path).name, "jobs": len(recs), "period_ms": recs[0]["period_ms"],
            "mean_ms": statistics.mean(lat), "p95_ms": lat[min(len(lat) - 1, math.ceil(0.95 * len(lat)) - 1)],
            "met_pct": 100.0 * sum(r["met"] for r in recs) / len(recs),
            "energy_mj": statistics.mean(r["energy_mj"] for r in recs),
            "approx_rate": statistics.mean(r["approx_rate"] for r in recs),
            "distinct_configs": len(set(cfgs)), "cfg_min": min(cfgs), "cfg_max": max(cfgs),
            "priority_last": recs[-1]["priority"]}


def cmd_summary(a):
    print("| run | jobs | period ms | mean ms | p95 ms | met % | energy mJ/job | approx rate | distinct cfgs (min..max) | last priority |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for p in a.files:
        s = summarize(p, a.include_warmup)
        if s:
            print("| %s | %d | %.0f | %.1f | %.1f | %.0f | %.0f | %.2f | %d (%d..%d) | %.4g |" % (
                s["file"], s["jobs"], s["period_ms"], s["mean_ms"], s["p95_ms"], s["met_pct"],
                s["energy_mj"], s["approx_rate"], s["distinct_configs"], s["cfg_min"], s["cfg_max"],
                s["priority_last"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("baseline")
    b.add_argument("--out")

    def common(p):
        p.add_argument("--workload", choices=sorted(WORKLOADS), required=True)
        p.add_argument("--dry-run", action="store_true", help="do not touch DVFS / INA3221")
        p.add_argument("--full-url", default="http://127.0.0.1:18081")
        p.add_argument("--approx-url", default="http://127.0.0.1:18082")
        p.add_argument("--full-name", default="qwen2.5-1.5b-q4_k_m")
        p.add_argument("--approx-name", default="qwen2.5-0.5b-q4_k_m")
        p.add_argument("--max-tokens", type=int, default=64)
        p.add_argument("--chunk-tokens", type=int, default=8)
        p.add_argument("--rank", type=float, default=0.25,
                       help="torch: low-rank twin rank as a ratio of full rank (0 = no twins)")

    p = sub.add_parser("profile")
    common(p)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warm-jobs", type=int, default=3)
    p.add_argument("--cpu-levels", type=int, default=3)
    p.add_argument("--gpu-levels", type=int, default=4)
    p.add_argument("--cores", nargs="+", choices=("all", "half"), default=["all", "half"])
    p.add_argument("--emc", type=int, nargs="+", default=[665600, 2133000, 3199000],
                   help="EMC kHz; kept only if in possible_rates and the write reads back")

    r = sub.add_parser("run")
    common(r)
    r.add_argument("--profile-dir", required=True)
    r.add_argument("--jobs", type=int, default=30)
    r.add_argument("--mode", default="DEFAULT", choices=("DEFAULT", "MAX_ACC", "MIN_ENG"))
    r.add_argument("--variant", default="original", choices=("original", "corrected"))
    g = r.add_mutually_exclusive_group()
    g.add_argument("--period-ms", type=float)
    g.add_argument("--period-factor", type=float, default=1.3,
                   help="period = factor x median wall latency at the max config (from profile)")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--cohort-prefix", default="neuos_eval")
    r.add_argument("--baseline", help="restore this saved snapshot instead of the state at start")
    r.add_argument("--out", required=True)

    s = sub.add_parser("summary")
    s.add_argument("files", nargs="+")
    s.add_argument("--include-warmup", action="store_true")
    a = ap.parse_args(argv)
    {"baseline": cmd_baseline, "profile": cmd_profile, "run": cmd_run, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    main()
