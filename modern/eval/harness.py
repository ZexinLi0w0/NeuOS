"""NeuOS baseline harness on the paper's voice-assistant workload (run as root on the board).

    sudo -n PY harness.py verify  --board-config configs/orin-nano.json
    sudo -n PY harness.py profile --board-config C --profile-dir P [--tasks asr perception turn]
    sudo -n PY harness.py run     --board-config C --policy neuos|default --minutes 20 \
                                  --profile-dir P --out-dir R [--turn-deadline-ms 4000]

profile: builds the shared DVFS config space (configs.txt, grid.json, baseline.json) and the
NeuOS Uncertainty of every task, each task alone with its full tier.
run: snapshot + check the baseline, generate the seeded Listen/Respond trace, start the 1 Hz
sampler, the llama-server tiers (full; + approx under neuos), then the ASR, perception and turn
processes one by one with a memory check before / after each; NeuOS warmup jobs; then the
measured window of --minutes. A watchdog aborts the run cleanly if MemAvailable drops below the
floor (no OOM experiments). Teardown: SIGTERM to the tasks (each restores the saved DVFS
baseline), stop the servers, final restore (neuos), cohort unlink, verification, summary.
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

from common import (HERE, gguf_path, install_signal_handlers, llm_pair, load_cfg, mem_avail_mb,
                    neuos_dir, neuos_path, read_json, write_json_atomic)
from listen_trace import make_trace

TASK_ORDER = ["perception", "asr", "turn"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# --- board state / verification -------------------------------------------------
def rd(p):
    try:
        with open(p) as f:
            return f.read().strip()
    except OSError:
        return None


def board_state():
    sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()  # noqa: E731
    pols = {}
    for d in sorted(Path("/sys/devices/system/cpu/cpufreq").glob("policy*")):
        pols[d.name] = [rd(d / "scaling_governor"), int(rd(d / "scaling_min_freq")), int(rd(d / "scaling_max_freq"))]
    gdir = next(iter(sorted(Path("/sys/devices/platform").glob("*/17000000.gpu/devfreq/*"))), None)
    emc = "/sys/kernel/debug/bpmp/debug/clk/emc/"
    nv = sh("nvpmodel -q 2>/dev/null").splitlines()
    return {"nvpmodel": nv[0].split(":", 1)[-1].strip() if nv else None,
            "nvpmodel_id": nv[1].strip() if len(nv) > 1 else None,
            "nvfancontrol": sh("systemctl is-active nvfancontrol"),
            "cpus_online": rd("/sys/devices/system/cpu/online"),
            "policies": pols,
            "gpu": [int(rd(gdir / "min_freq")), int(rd(gdir / "max_freq"))] if gdir else None,
            "emc": [int(rd(emc + "rate") or -1), int(rd(emc + "mrq_rate_locked") or -1)]}


def mismatches(state, exp):
    bad = []
    if state["nvpmodel"] != exp["nvpmodel"]:
        bad.append("nvpmodel %s != %s" % (state["nvpmodel"], exp["nvpmodel"]))
    if state["nvfancontrol"] != "active":
        bad.append("nvfancontrol %s" % state["nvfancontrol"])
    if state["cpus_online"] != exp["cpus_online"]:
        bad.append("cpus online %s != %s" % (state["cpus_online"], exp["cpus_online"]))
    for k, v in exp["policies"].items():
        if state["policies"].get(k) != v:
            bad.append("%s %s != %s" % (k, state["policies"].get(k), v))
    if state["gpu"] != exp["gpu"]:
        bad.append("gpu %s != %s" % (state["gpu"], exp["gpu"]))
    if state["emc"] != exp["emc"]:
        bad.append("emc %s != %s" % (state["emc"], exp["emc"]))
    return bad


def own_processes():
    """Processes of this harness still alive (narrow patterns), excluding this process."""
    out = subprocess.run(["pgrep", "-af", r"neuos_eval/harness/(tasks|sampler)\.py|llama-server .*--port 1810[0-9]"],
                         capture_output=True, text=True).stdout.splitlines()
    return [l for l in out if not l.startswith(str(os.getpid()) + " ")]


def shm_blocks(prefix):
    return sorted(p.name for p in Path("/dev/shm").glob("*neuos*"))


def verify(cfg, out=None):
    st = board_state()
    bad = mismatches(st, cfg["expected_baseline"])
    procs = own_processes()
    shm = shm_blocks(cfg["neuos"]["cohort_prefix"])
    res = {"board": cfg["board"], "state": st, "mismatches": bad, "leftover_processes": procs,
           "neuos_shm_blocks": shm, "ok": not bad and not procs and not shm}
    print("== verify %s ==" % cfg["board"])
    print("nvpmodel: %s (id %s)  nvfancontrol: %s  cpus online: %s" % (
        st["nvpmodel"], st["nvpmodel_id"], st["nvfancontrol"], st["cpus_online"]))
    for k, v in st["policies"].items():
        print("  %s: governor %s min %d max %d" % (k, v[0], v[1], v[2]))
    print("  GPU devfreq min %s max %s" % tuple(st["gpu"] or (None, None)))
    print("  EMC rate %s mrq_rate_locked %s" % tuple(st["emc"]))
    print("mismatches vs baseline: %s" % (bad or "none"))
    print("leftover processes: %s" % (procs or "none"))
    print("/dev/shm neuos blocks: %s" % (shm or "none"))
    print("VERIFY %s" % ("OK" if res["ok"] else "FAILED"), flush=True)
    if out:
        Path(out).write_text(json.dumps(res, indent=1))
    return res


def oom_count():
    out = subprocess.run(["dmesg"], capture_output=True, text=True).stdout
    return len(re.findall(r"Out of memory|oom-kill|oom_reaper|NVMAP.*(?:failed|OOM)", out))


# --- process management -----------------------------------------------------------
class Launcher:
    def __init__(self, cfg, run_dir):
        self.cfg, self.run = cfg, Path(run_dir)
        self.procs = {}           # name -> Popen
        self.floor = cfg["mem_floor_mb"]
        self.min_avail = mem_avail_mb()
        self.events = open(self.run / "events.jsonl", "a")
        self.abort = None

    def event(self, **kw):
        kw["t"] = time.time()
        self.events.write(json.dumps(kw) + "\n")
        self.events.flush()
        log("event", json.dumps(kw))

    def env(self):
        e = dict(os.environ)
        c = self.cfg
        e.update(HF_HOME=c["hf_home"], HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1",
                 PYTHONPATH=":".join([str(HERE), neuos_dir(c)] + c.get("pythonpath_extra", [])))
        return e

    def check_mem(self, where):
        av = mem_avail_mb()
        self.min_avail = min(self.min_avail, av)
        if av < self.floor:
            self.abort = {"reason": "mem_floor", "where": where, "avail_mb": av, "floor_mb": self.floor}
            self.event(ev="abort", **self.abort)
            return False
        return True

    def preflight(self, name):
        est = self.cfg["mem_estimate_mb"].get(name, 0)
        av = mem_avail_mb()
        ok = av - est >= self.floor + self.cfg["mem_margin_mb"]
        self.event(ev="preflight", component=name, avail_mb=av, estimate_mb=est, ok=ok)
        if not ok:
            self.abort = {"reason": "preflight_mem", "component": name, "avail_mb": av, "estimate_mb": est,
                          "floor_mb": self.floor, "margin_mb": self.cfg["mem_margin_mb"]}
            self.event(ev="abort", **self.abort)
        return ok

    def wait(self, name, cond, timeout):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if not self.check_mem("loading " + name):
                return False
            if cond():
                return True
            for n, p in self.procs.items():
                if p.poll() is not None:
                    self.abort = {"reason": "process_exit", "component": n, "rc": p.returncode, "where": name}
                    self.event(ev="abort", **self.abort)
                    return False
            time.sleep(0.25)
        self.abort = {"reason": "timeout", "where": name}
        self.event(ev="abort", **self.abort)
        return False

    def start_llama(self, tier, port, key):
        import urllib.request
        c, L = self.cfg, self.cfg["llm"]
        if not self.preflight(key):
            return False
        av0 = mem_avail_mb()
        cmd = [c["llama_server"], "-m", gguf_path(c, tier), "-ngl", str(L["ngl"]), "-c", str(L["ctx"]),
               "-np", "1", "--host", "127.0.0.1", "--port", str(port)] + list(L.get("server_args", []))
        self.procs[key] = subprocess.Popen(cmd, stdout=open(self.run / ("%s.log" % key), "a"),
                                           stderr=subprocess.STDOUT, env=self.env(), start_new_session=True)

        def healthy():
            try:
                return urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=1).status == 200
            except Exception:
                return False
        ok = self.wait(key, healthy, 600)
        self.event(ev="loaded", component=key, tier=tier["name"], port=port, ok=ok,
                   avail_before_mb=av0, avail_mb=mem_avail_mb())
        return ok

    def start_task(self, name, policy, extra=()):
        if not self.preflight(name):
            return False
        av0 = mem_avail_mb()
        cmd = [self.cfg["python"], str(HERE / "tasks.py"), name, "--board-config",
               str(self.run / "board_config.json"), "--policy", policy] + list(extra)
        self.procs[name] = subprocess.Popen(cmd, stdout=open(self.run / ("%s.out" % name), "a"),
                                            stderr=subprocess.STDOUT, env=self.env(), start_new_session=True)
        ok = self.wait(name, lambda: (self.run / ("loaded_%s" % name)).exists(), 900)
        self.event(ev="loaded", component=name, ok=ok, avail_before_mb=av0, avail_mb=mem_avail_mb())
        return ok and self.check_mem("after " + name)

    def stop(self, names, timeout=30, grace=0):
        t0 = time.time()
        while grace and time.time() - t0 < grace and any(
                self.procs.get(n) and self.procs[n].poll() is None for n in names):
            time.sleep(0.1)                      # let them exit on their own first
        for n in names:
            p = self.procs.get(n)
            if p and p.poll() is None:
                p.send_signal(signal.SIGTERM)
        t0 = time.time()
        for n in names:
            p = self.procs.get(n)
            if not p:
                continue
            try:
                p.wait(max(1, timeout - (time.time() - t0)))
            except subprocess.TimeoutExpired:
                self.event(ev="sigkill", component=n)
                p.kill()
                p.wait()
            self.event(ev="exit", component=n, rc=p.returncode)


# --- grid / profile ---------------------------------------------------------------
def build_grid(cfg, pdir):
    neuos_path(cfg)
    from neuos_modern import OrinSysfsBackend, build_config_space, save_configs, subsample
    g = cfg["grid"]
    backend = OrinSysfsBackend()
    try:
        s = backend.enumerate_states()
        emc = backend.probe_emc_rates([r for r in g["emc_khz"] if r in s["emc_freqs_khz"]])
        cpu = subsample([f for f in s["cpu_freqs_khz"] if f >= g.get("cpu_min_khz", 0)], g["cpu_levels"])
        gpu = subsample(s["gpu_freqs_hz"], g["gpu_levels"])
        n = s["n_cpus"]
        cores = sorted({n if c == "all" else max(1, n // 2) for c in g["cores"]})
        configs = build_config_space(cpu, gpu, emc, cores, n)
    finally:
        backend.restore()
    save_configs(pdir / "configs.txt", configs)
    (pdir / "baseline.json").write_text(json.dumps(backend.baseline))
    grid = {"board": cfg["board"], "cpu_khz": cpu, "gpu_hz": gpu, "emc_khz": emc, "emc_requested_khz": g["emc_khz"],
            "cores": cores, "n_cpus": n, "n_configs": len(configs), "states": s, "baseline": backend.baseline}
    (pdir / "grid.json").write_text(json.dumps(grid, indent=1))
    log("grid: %d configs cpu=%s cores=%s gpu=%s emc=%s" % (len(configs), cpu, cores, gpu, emc))


def cmd_profile(a, cfg):
    pdir = Path(a.profile_dir)
    pdir.mkdir(parents=True, exist_ok=True)
    install_signal_handlers()
    bad = mismatches(board_state(), cfg["expected_baseline"])
    if bad:
        log("baseline mismatch, refusing:", bad)
        return 1
    if not (pdir / "configs.txt").exists():
        build_grid(cfg, pdir)
    (pdir / "board_config.json").write_text(json.dumps(cfg, indent=1))
    L = Launcher(cfg, pdir)
    L.run = pdir
    rc = 0
    try:
        for t in a.tasks:
            if (pdir / ("Uncertainty_%s.txt" % t)).exists():
                log("profile exists for", t)
                continue
            if t == "turn":
                full, _ = llm_pair(cfg)
                if not L.start_llama(full, cfg["llm"]["ports"][0], "llm_full"):
                    rc = 1
                    break
            if not L.preflight(t):
                rc = 1
                break
            log("profiling", t)
            cmd = [cfg["python"], str(HERE / "tasks.py"), t, "--board-config", str(pdir / "board_config.json"),
                   "--profile", "--profile-dir", str(pdir)]
            p = L.procs["prof_" + t] = subprocess.Popen(cmd, stdout=open(pdir / ("profile_%s.log" % t), "a"),
                                                        stderr=subprocess.STDOUT, env=L.env(), start_new_session=True)
            while p.poll() is None:
                if not L.check_mem("profile " + t):
                    L.stop(["prof_" + t])
                    rc = 1
                    break
                time.sleep(0.25)
            L.procs.pop("prof_" + t)
            log("profile", t, "rc", p.returncode, "min avail", L.min_avail)
            if t == "turn":
                L.stop(["llm_full"])
            if p.returncode != 0:
                rc = 1
                break
    finally:
        L.stop(list(L.procs))
        neuos_path(cfg)
        from neuos_modern import OrinSysfsBackend
        OrinSysfsBackend(baseline=read_json(pdir / "baseline.json")).restore()
        chown(pdir, cfg["workdir"])
        verify(cfg, pdir / "verify_profile.json")
    return rc


# --- run ------------------------------------------------------------------------
def chown(path, top=None):
    """Give the run directory (and the parents created under ``top``) back to the sudo user."""
    uid, gid = os.environ.get("SUDO_UID"), os.environ.get("SUDO_GID")
    if uid and gid:
        subprocess.run(["chown", "-R", "%s:%s" % (uid, gid), str(path)])
        p = Path(path).resolve().parent
        while top and str(p).startswith(str(Path(top).resolve())) and p != Path(top).resolve():
            subprocess.run(["chown", "%s:%s" % (uid, gid), str(p)])
            p = p.parent


def cmd_run(a, cfg):
    install_signal_handlers()
    run = Path(a.out_dir)
    run.mkdir(parents=True, exist_ok=True)
    if a.turn_deadline_ms:
        cfg["turn"]["deadline_ms"] = a.turn_deadline_ms
    cfg["policy"] = a.policy
    cfg["profile_dir"] = a.profile_dir
    (run / "board_config.json").write_text(json.dumps(cfg, indent=1))
    neuos_path(cfg)
    from neuos_modern import OrinSysfsBackend, SharedCohort
    st0 = board_state()
    bad = mismatches(st0, cfg["expected_baseline"])
    if bad:
        log("baseline mismatch, refusing to run:", bad)
        return 1
    if own_processes():
        log("leftover harness processes, refusing:", own_processes())
        return 1
    prefix = cfg["neuos"]["cohort_prefix"]
    SharedCohort.unlink(prefix)                          # fresh cohort (removes our own old blocks only)
    snap = OrinSysfsBackend().snapshot()
    (run / "baseline.json").write_text(json.dumps(snap))
    tr = make_trace(a.minutes, cfg["trace"]["seed"], tuple(cfg["trace"]["listen_s"]), tuple(cfg["trace"]["idle_s"]))
    (run / "trace.json").write_text(json.dumps(tr, indent=1))
    L = Launcher(cfg, run)
    meta = {"board": cfg["board"], "policy": a.policy, "minutes": a.minutes, "neuos": cfg["neuos"],
            "turn_deadline_ms": cfg["turn"]["deadline_ms"], "oom_dmesg_before": oom_count(),
            "avail_mb_start": mem_avail_mb(), "state_start": st0}
    L.event(ev="start", policy=a.policy, minutes=a.minutes)
    stopfile = run / "sampler.stop"
    L.procs["sampler"] = subprocess.Popen([cfg["python"], str(HERE / "sampler.py"), str(run / "samples.jsonl"),
                                           str(stopfile), neuos_dir(cfg)],
                                          stdout=open(run / "sampler.out", "a"), stderr=subprocess.STDOUT,
                                          env=L.env(), start_new_session=True)
    extra = ["--run-dir", str(run)] + (["--profile-dir", a.profile_dir] if a.policy == "neuos" else [])
    tasks = []
    try:
        full, approx = llm_pair(cfg)
        ok = L.start_llama(full, cfg["llm"]["ports"][0], "llm_full")
        if ok and a.policy == "neuos" and approx is not full:
            ok = L.start_llama(approx, cfg["llm"]["ports"][1], "llm_approx")
        for t in TASK_ORDER:
            if not ok:
                break
            ok = L.start_task(t, a.policy, extra)
            tasks.append(t)
        if ok:
            (run / "go_warmup").write_text("1")
            ok = L.wait("warmup", lambda: all((run / ("ready_%s" % t)).exists() for t in TASK_ORDER), 600)
        if ok:
            t_start = time.time() + 2.0
            t_end = t_start + a.minutes * 60
            write_json_atomic(run / "start.json", {"t_start": t_start, "t_end": t_end})
            meta.update(t_start=t_start, t_end=t_end)
            L.event(ev="measure_start", t_start=t_start, t_end=t_end, avail_mb=mem_avail_mb())
            while True:
                if not L.check_mem("run"):
                    break
                alive = [t for t in TASK_ORDER if L.procs[t].poll() is None]
                failed = [(t, L.procs[t].returncode) for t in TASK_ORDER
                          if L.procs[t].poll() is not None and L.procs[t].returncode != 0]
                if failed:
                    L.abort = {"reason": "task_failed", "tasks": failed}
                    L.event(ev="abort", **L.abort)
                    break
                if not alive:
                    break
                if time.time() > t_end + a.grace_s:
                    L.abort = {"reason": "grace_timeout", "alive": alive}
                    L.event(ev="abort", **L.abort)
                    break
                time.sleep(0.25)
            L.event(ev="measure_end", avail_mb=mem_avail_mb())
    finally:
        L.stop([t for t in TASK_ORDER if t in L.procs], timeout=60)   # tasks restore DVFS themselves
        L.stop([k for k in ("llm_full", "llm_approx") if k in L.procs])
        stopfile.write_text("1")
        L.stop(["sampler"], timeout=10, grace=5)
        st = board_state()
        if a.policy == "neuos" or mismatches(st, cfg["expected_baseline"]):
            OrinSysfsBackend(baseline=snap).restore()
            L.event(ev="final_restore")
        SharedCohort.unlink(prefix)
        meta.update(abort=L.abort, min_avail_mb_launcher=L.min_avail, oom_dmesg_after=oom_count(),
                    t_finish=time.time())
        (run / "run_meta.json").write_text(json.dumps(meta, indent=1))
        ver = verify(cfg, run / "verify.json")
        try:
            from summarize import summarize_run
            summarize_run(run, write=True, show=True)
        except Exception as e:                    # summary problems must not hide the run state
            log("summary failed:", repr(e))
        chown(run, cfg["workdir"])
    return 0 if (L.abort is None and ver["ok"]) else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--board-config", required=True)
    p = sub.add_parser("profile")
    p.add_argument("--board-config", required=True)
    p.add_argument("--profile-dir", required=True)
    p.add_argument("--tasks", nargs="+", default=TASK_ORDER, choices=TASK_ORDER)
    r = sub.add_parser("run")
    r.add_argument("--board-config", required=True)
    r.add_argument("--policy", choices=("neuos", "default"), required=True)
    r.add_argument("--minutes", type=float, default=20)
    r.add_argument("--profile-dir", help="required for --policy neuos")
    r.add_argument("--out-dir", required=True)
    r.add_argument("--turn-deadline-ms", type=float)
    r.add_argument("--grace-s", type=float, default=180)
    for sp in (p, r):
        sp.add_argument("--set", action="append", default=[], metavar="KEY=JSON",
                        help="override a board-config entry, e.g. --set 'grid.cores=[\"all\"]'")
    a = ap.parse_args()
    cfg = load_cfg(a.board_config)
    for kv in getattr(a, "set", []):
        key, val = kv.split("=", 1)
        d = cfg
        *path, last = key.split(".")
        for k in path:
            d = d[k]
        d[last] = json.loads(val)
    if a.cmd == "verify":
        return 0 if verify(cfg)["ok"] else 1
    if os.geteuid() != 0:
        print("run as root (sudo -n)", file=sys.stderr)
        return 1
    if a.cmd == "profile":
        return cmd_profile(a, cfg)
    if a.policy == "neuos" and not a.profile_dir:
        ap.error("--profile-dir is required for --policy neuos")
    return cmd_run(a, cfg)


if __name__ == "__main__":
    sys.exit(main())
