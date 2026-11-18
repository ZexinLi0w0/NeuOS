"""Shared helpers for the NeuOS evaluation harness (stdlib only)."""
import glob
import json
import os
import signal
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_cfg(path):
    with open(path) as f:
        return json.load(f)


def neuos_dir(cfg):
    """Directory that holds the neuos_modern package: the board config's ``neuos_modern_dir``
    when it exists, else this fork's ``modern/`` (the parent of this eval/ directory)."""
    d = cfg.get("neuos_modern_dir")
    if d and os.path.isdir(os.path.join(d, "neuos_modern")):
        return d
    return str(HERE.parent)


def neuos_path(cfg):
    """Make neuos_modern and the extra site dirs (diffusers) importable."""
    for p in [neuos_dir(cfg)] + cfg.get("pythonpath_extra", []):
        if p not in sys.path:
            sys.path.insert(0, p)


def twins_cache_path(cfg):
    pc = cfg["perception"]
    return os.path.join(cfg["workdir"], "cache", "unet_x4_twins_r%s.pt" % pc["twin_rank_ratio"])


def gguf_path(cfg, tier):
    pat = os.path.join(cfg["hf_home"], "hub", tier["gguf_glob"])
    found = sorted(glob.glob(pat))
    if not found:
        raise FileNotFoundError(pat)
    return found[0]                      # first split file; llama.cpp loads the rest


def llm_pair(cfg):
    """(full tier, approx tier) of the LLM ladder: NeuOS's full net and its twin."""
    L = cfg["llm"]
    full = L["tiers"][L["full"]]
    approx = L["tiers"][min(L["full"] + 1, len(L["tiers"]) - 1)]
    return full, approx


def asr_pair(cfg):
    A = cfg["asr"]
    return A["tiers"][A["full"]], A["tiers"][min(A["full"] + 1, len(A["tiers"]) - 1)]


def mem_avail_mb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) // 1024
    return -1


def install_signal_handlers():
    """SIGTERM / SIGINT unwind through ``finally`` (DVFS restore)."""
    def bye(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, bye)
    signal.signal(signal.SIGINT, bye)


def write_json_atomic(path, obj):
    tmp = "%s.tmp%d" % (path, os.getpid())
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


class Jsonl:
    def __init__(self, path):
        self.f = open(path, "a")

    def write(self, **rec):
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def sleep_until(t, stop=None):
    """Sleep until wall time t (time.time()); returns early if stop() becomes true."""
    while True:
        d = t - time.time()
        if d <= 0 or (stop and stop()):
            return
        time.sleep(min(d, 0.05))


def load_clips(npz):
    """LibriSpeech validation-clean utterances (npz written by fetch_clips.py, 16 kHz).
    Returns a list of float32 arrays, or None when the file is missing."""
    import numpy as np
    if not npz or not os.path.exists(npz):
        return None
    z = np.load(npz)
    offs = np.concatenate([[0], np.cumsum(z["lens"])])
    return [z["audio"][offs[i]:offs[i + 1]].astype("float32") for i in range(len(z["lens"]))]


def synthetic_clip(seconds, seed):
    import numpy as np
    sr = 16000
    t = np.arange(int(sr * seconds)) / sr
    rng = np.random.default_rng(seed)
    return (0.1 * np.sin(2 * np.pi * (180 + 40 * (seed % 5)) * t)
            + 0.02 * rng.standard_normal(t.size)).astype("float32")


# --- stage-log reduction (shared by tasks and summary) -----------------------
def stage_summary(log):
    """Compact NeuOS job log: configs, approximation flags, slack, priority, energy."""
    st = (log or {}).get("stages") or []
    if not st:
        return {}
    cfgs = [s["config"] for s in st]
    rle = []
    for c in cfgs:
        if rle and rle[-1][0] == c:
            rle[-1][1] += 1
        else:
            rle.append([c, 1])
    return {"n_stages": len(st), "warmup": log.get("warmup"), "slot": log.get("slot"),
            "stage_sum_ms": sum(s.get("latency_us", 0) for s in st) / 1e3,
            "n_cfg_changes": len(rle) - 1, "cfg_rle": rle if len(rle) <= 40 else rle[:40] + [["...", len(rle)]],
            "sched_us": log.get("sched_us"), "slack_end": log.get("slack"),
            "energy_mj": log.get("energy_mj"),
            "cfg_first": st[0]["config"], "cfg_last": st[-1]["config"],
            "cfg_min": min(s["config"] for s in st), "cfg_max": max(s["config"] for s in st),
            "n_approx": sum(bool(s["approximated"]) for s in st),
            "approx_stage0": bool(st[0]["approximated"]),
            "priority_first": st[0]["priority"], "priority_last": st[-1]["priority"],
            "a_star_last": st[-1]["a_star"],
            "slack_min": min(s.get("slack", 0.0) for s in st),
            "slack_max": max(s.get("slack", 0.0) for s in st)}
