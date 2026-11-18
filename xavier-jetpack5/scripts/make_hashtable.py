"""Xavier hash table (per-layer Uncertainty) from profile_sweep.sh logs.

Same formula as hashtables/generate_hashtable.py: for every layer l and profiled
configuration c, normalize latency and total energy by the layer's maximum over all
configurations, and take
    Uncertainty[l] = max_c (E[c,l] / max_c E[.,l]) / (L[c,l] / max_c L[.,l])
"Total" energy is, by default, the first energy column (SYS5V rail, energymon's total in
the AGX code). On the Xavier the GPU and CPU are not on SYS5V, so --total sum4 instead sums
the four measured rails (SYS5V, GPU, CPU, VDDRQ).
Differences from the original script: Python 3, one value per network layer in layer
order (the original's row/column offsets put a 0 first and a raw latency last), and
the profile comes from scripts/profile_layers.cpp instead of NeuOS' 20-repeat timing.

usage: make_hashtable.py [--total sys5v|sum4] PROFILE_DIR OUT_HASHTABLE [OUT_CSV]
"""
import csv
import sys
from pathlib import Path


TOTAL = "sys5v"


def read_log(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        f = line.split()
        energy = float(f[2]) if TOTAL == "sys5v" else sum(float(x) for x in f[2:6])
        rows.append((f[0], float(f[1]), energy))
    return rows


def main(prof_dir, out_table, out_csv=None):
    logs = sorted(Path(prof_dir).glob("[0-9]*.log"), key=lambda p: int(p.stem))
    data = {int(p.stem): read_log(p) for p in logs}
    data = {k: v for k, v in data.items() if v}
    names = [r[0] for r in next(iter(data.values()))]
    assert all([r[0] for r in v] == names for v in data.values()), "layer lists differ"
    n = len(names)
    max_l = [max(v[i][1] for v in data.values()) for i in range(n)]
    max_e = [max(v[i][2] for v in data.values()) for i in range(n)]
    unc, argmax = [], []
    for i in range(n):
        best, best_c = 0.0, None
        for c, v in data.items():
            nl = v[i][1] / max_l[i] if max_l[i] else 0.0
            ne = v[i][2] / max_e[i] if max_e[i] else 0.0
            u = ne / nl if nl else 0.0
            if u > best:
                best, best_c = u, c
        unc.append(best)
        argmax.append(best_c)
    Path(out_table).write_text("".join(f"{u}\t" for u in unc) + "\n")
    if out_csv:
        with open(out_csv, "w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(["layer", "uncertainty", "argmax_config", "max_latency_us", "max_energy_mJ",
                        "min_latency_us", "min_energy_mJ"])
            for i in range(n):
                w.writerow([names[i], f"{unc[i]:.4f}", argmax[i], f"{max_l[i]:.3f}", f"{max_e[i]:.6f}",
                            f"{min(v[i][1] for v in data.values()):.3f}",
                            f"{min(v[i][2] for v in data.values()):.6f}"])
    print(f"{len(data)} configurations, {n} layers")
    for name, u in zip(names, unc):
        print(f"{name:8s} {u:.3f}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["--total"]:
        TOTAL = args[1]
        assert TOTAL in ("sys5v", "sum4"), TOTAL
        args = args[2:]
    main(*args)
