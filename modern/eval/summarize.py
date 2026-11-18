"""Paper metrics for one or more harness run directories (stdlib only; runs on the board or locally).

    python summarize.py RUN_DIR [RUN_DIR ...]          # markdown table + summary.json per run

Definitions (measured window [t_start, t_end] from start.json):
- deadline miss ratio, per task = (late + dropped) / released. ASR and perception: periodic
  releases (1 s / 200 ms; ASR only inside Listen phases), deadline = period, a release that falls
  while the previous job still runs is dropped. Turn: released at the end of each Listen; late
  if the turn latency exceeds the turn deadline (or no audio was produced). "all" pools the jobs
  of the three tasks.
- turn latency = time from the end of the request (end of Listen) to the first audible reply
  (first sentence synthesised by MMS-TTS); p50 / p99 (nearest rank).
- LLM decoding rate = llama-server's predicted_per_second of each turn (falls back to the
  client-side token rate when the server does not report timings); mean over turns.
- TTS real-time factor = synthesis time / audio duration of every TTS call (first sentence and
  rest); p50 / p99 / max and the fraction of calls with RTF < 1.
- OOM events = OOM-killer / NvMap lines added to dmesg during the run + CUDA OOM errors logged by
  the tasks. Watchdog aborts (MemAvailable < floor) are reported separately as "abort".
- min MemAvailable = min over the 1 Hz sampler, the tasks' per-job readings and the launcher's
  watchdog (4 Hz), over the whole run including model loading.
- throttled fraction = fraction of 1 Hz samples in the window where (a) any thermal cooling
  device other than the fan (pwm-fan) has cur_state > 0 (cpufreq / devfreq thermal caps and the
  *-throttle-alert / hot-surface alerts), or (b) the BPMP actual GPU clock (max of gpcNclk) is below 95%
  of the GPU devfreq cur_freq (the frequency DVFS asked for) in two adjacent samples with the same
  devfreq cur_freq (a one-sample gap is read skew during a governor transition, not a cap).
  (a) and (b) are also reported alone.
- peak Tj = max of tj-thermal over the window.
- energy = integral of board power over the window (neuos_modern EnergyMeter at 20 Hz: VDD_IN on
  the Nano; VDD_GPU_SOC + VDD_CPU_CV + VIN_SYS_5V0 + VDDQ_VDD2_1V8AO on the AGX); also mean power.
"""
import gzip
import json
import math
import sys
from pathlib import Path

TASKS = ("asr", "perception", "turn")


def pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return xs[max(0, min(len(xs) - 1, math.ceil(p / 100 * len(xs)) - 1))]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def jl(path):
    """JSON lines from ``path`` or, when only that exists, ``path + ".gz"`` (large logs are
    stored gzipped in results/)."""
    path = Path(path)
    if path.exists():
        text = path.read_text()
    elif Path(str(path) + ".gz").exists():
        with gzip.open(str(path) + ".gz", "rt") as f:
            text = f.read()
    else:
        return []
    out = []
    for line in text.splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def summarize_run(run, write=False, show=False):
    run = Path(run)
    meta = json.loads((run / "run_meta.json").read_text()) if (run / "run_meta.json").exists() else {}
    st = json.loads((run / "start.json").read_text()) if (run / "start.json").exists() else {}
    t0, t1 = st.get("t_start"), st.get("t_end")
    res = {"run": run.name, "board": meta.get("board"), "policy": meta.get("policy"),
           "minutes": meta.get("minutes"), "abort": meta.get("abort"),
           "turn_deadline_ms": meta.get("turn_deadline_ms"), "tasks": {}}
    tot_rel = tot_miss = 0
    avail = [meta.get("min_avail_mb_launcher")]
    ooms = 0
    for t in TASKS:
        recs = jl(run / ("%s.jsonl" % t))
        jobs = [r for r in recs if r.get("ev") == "job"]
        avail += [r.get("avail_mb") for r in recs]
        ooms += sum(r.get("ev") == "oom" for r in recs)
        ex = [r for r in jobs if r["status"] != "dropped"]
        n_late = sum(r["status"] == "late" for r in jobs)
        n_drop = sum(r["status"] == "dropped" for r in jobs)
        d = {"released": len(jobs), "executed": len(ex), "met": len(ex) - n_late, "late": n_late,
             "dropped": n_drop, "miss_ratio": (n_late + n_drop) / len(jobs) if jobs else None,
             "approx_frac": mean([float(bool((r.get("neuos") or {}).get("n_approx"))) for r in ex]),
             "energy_mj_per_job": mean([(r.get("neuos") or {}).get("energy_mj") for r in ex]),
             "errors": [r.get("error") for r in recs if r.get("ev") == "error"],
             "end": [r.get("result") for r in recs if r.get("ev") == "end"]}
        loaded = [r for r in recs if r.get("ev") == "loaded"]
        if loaded:
            d["audio_source"] = loaded[0].get("audio_source")
            d["load_avail_delta_mb"] = loaded[0]["avail_before_mb"] - loaded[0]["avail_mb"]
        lat = [r["latency_ms"] for r in ex]
        d.update(latency_p50_ms=pct(lat, 50), latency_p99_ms=pct(lat, 99), latency_max_ms=pct(lat, 100))
        tiers = {}
        for r in ex:
            tiers[r.get("tier")] = tiers.get(r.get("tier"), 0) + 1
        d["tiers"] = tiers
        cfgs = [(r.get("neuos") or {}).get("cfg_last") for r in ex if r.get("neuos")]
        if cfgs:
            d["cfg_last_hist"] = {str(c): cfgs.count(c) for c in sorted(set(cfgs))}
        if t == "turn":
            tl = [r.get("turn_latency_ms") for r in ex]
            d.update(turn_latency_p50_ms=pct(tl, 50), turn_latency_p99_ms=pct(tl, 99),
                     turn_latency_max_ms=pct(tl, 100), ttft_p50_ms=pct([r["ttft_s"] * 1e3 for r in ex], 50),
                     tps_server_mean=mean([r.get("tps_server") for r in ex]),
                     tps_client_mean=mean([r.get("tps_client") for r in ex]),
                     n_tok_mean=mean([r.get("n_tok") for r in ex]))
            d["tps_mean"] = d["tps_server_mean"] if d["tps_server_mean"] is not None else d["tps_client_mean"]
            rtf = [c["synth_s"] / c["audio_s"] for r in ex for c in r.get("tts", []) if c.get("audio_s")]
            d.update(tts_rtf_p50=pct(rtf, 50), tts_rtf_p99=pct(rtf, 99), tts_rtf_max=pct(rtf, 100),
                     tts_rtf_lt1_frac=mean([float(x < 1) for x in rtf]), tts_calls=len(rtf))
        res["tasks"][t] = d
        tot_rel += len(jobs)
        tot_miss += n_late + n_drop
    res["miss_ratio_all"] = tot_miss / tot_rel if tot_rel else None
    S = jl(run / "samples.jsonl")
    avail += [s.get("avail_mb") for s in S]
    W = [s for s in S if t0 and t0 <= s["t"] <= t1] if t0 else []
    sysm = {"samples": len(W)}
    if W:
        def cdev_hot(s):
            return any((v or 0) > 0 for k, v in (s.get("cdev") or {}).items() if k and "fan" not in k)

        def gpu_low(s):
            dv, ac = s.get("gpu_devfreq_mhz") or 0, s.get("gpu_actual_mhz") or 0
            return dv > 0 and ac < 0.95 * dv
        a = [cdev_hot(s) for s in W]
        # sustained: below the request in this and an adjacent sample with the same request (a
        # single-sample gap is the skew between the two reads during a governor transition)
        lo = [gpu_low(s) for s in W]
        same = [i > 0 and W[i].get("gpu_devfreq_mhz") == W[i - 1].get("gpu_devfreq_mhz") for i in range(len(W))]
        b = [lo[i] and ((i > 0 and lo[i - 1] and same[i]) or (i + 1 < len(W) and lo[i + 1] and same[i + 1]))
             for i in range(len(W))]
        tj = [(s.get("temp") or {}).get("tj-thermal") for s in W]
        e = [s["energy_mj"] for s in W if s.get("energy_mj") is not None]
        dur = W[-1]["t"] - W[0]["t"]
        sysm.update(throttled_frac=mean([float(x or y) for x, y in zip(a, b)]), cdev_frac=mean([float(x) for x in a]),
                    gpu_clock_capped_frac=mean([float(x) for x in b]), peak_tj_c=max(x for x in tj if x is not None),
                    mean_tj_c=mean(tj), gpu_actual_mhz_mean=mean([s.get("gpu_actual_mhz") for s in W]),
                    emc_mhz_mean=mean([s.get("emc_mhz") for s in W]),
                    energy_j=(e[-1] - e[0]) / 1e3 if len(e) > 1 else None,
                    mean_power_w=(e[-1] - e[0]) / 1e3 / dur if len(e) > 1 and dur > 0 else None,
                    window_s=dur)
    oom_d = None
    if meta.get("oom_dmesg_after") is not None:
        oom_d = meta["oom_dmesg_after"] - meta["oom_dmesg_before"]
    sysm.update(oom_events=(oom_d or 0) + ooms, oom_dmesg=oom_d, oom_cuda=ooms,
                min_avail_mb=min(x for x in avail if x is not None) if any(x is not None for x in avail) else None)
    res["system"] = sysm
    if write:
        (run / "summary.json").write_text(json.dumps(res, indent=1))
    if show:
        print_table([res])
    return res


def f(x, fmt="%.1f"):
    return "-" if x is None else fmt % x


def print_table(rows):
    print("| board | policy | min | miss ASR | miss perc | miss turn | miss all | turn p50 ms | turn p99 ms "
          "| tok/s | TTS RTF p50/p99 | OOM | min avail MB | throttled % | peak Tj C | energy J | mean W | abort |")
    print("|" + "---|" * 18)
    for r in rows:
        T, S = r["tasks"], r["system"]
        print("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s/%s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["board"], r["policy"], r["minutes"],
            f(T["asr"]["miss_ratio"], "%.3f"), f(T["perception"]["miss_ratio"], "%.3f"),
            f(T["turn"]["miss_ratio"], "%.3f"), f(r["miss_ratio_all"], "%.3f"),
            f(T["turn"].get("turn_latency_p50_ms"), "%.0f"), f(T["turn"].get("turn_latency_p99_ms"), "%.0f"),
            f(T["turn"].get("tps_mean")), f(T["turn"].get("tts_rtf_p50"), "%.3f"),
            f(T["turn"].get("tts_rtf_p99"), "%.3f"), S.get("oom_events"), S.get("min_avail_mb"),
            f(100 * S["throttled_frac"] if S.get("throttled_frac") is not None else None),
            f(S.get("peak_tj_c")), f(S.get("energy_j"), "%.0f"), f(S.get("mean_power_w"), "%.2f"),
            (r.get("abort") or {}).get("reason", "-") if r.get("abort") else "-"))


def main():
    rows = [summarize_run(p, write=True) for p in sys.argv[1:]]
    print_table(rows)
    for r in rows:
        print("\n## %s / %s / %s" % (r["board"], r["policy"], r["run"]))
        for t, d in r["tasks"].items():
            print("%-10s released %d executed %d met %d late %d dropped %d miss %.3f  lat p50 %s p99 %s ms  "
                  "approx %s  tiers %s" % (t, d["released"], d["executed"], d["met"], d["late"], d["dropped"],
                                          d["miss_ratio"] or 0, f(d["latency_p50_ms"]), f(d["latency_p99_ms"]),
                                          f(d["approx_frac"], "%.2f"), d["tiers"]))
        print("system", json.dumps(r["system"]))


if __name__ == "__main__":
    main()
