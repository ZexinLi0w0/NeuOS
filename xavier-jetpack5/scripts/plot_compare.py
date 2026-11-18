"""Comparison figures and tables (Xavier hash table vs TX2 table vs DVFS-off baselines).

usage: python plot_compare.py COMPARE_DIR HASH_TX2 HASH_XAVIER_SYS5V HASH_XAVIER_SUM4 HASH_CSV FIG_DIR

COMPARE_DIR holds pinned_r*.log, default_r*.log, idle_{pinned,default}.tsv and, from
dvfs_run.sh, dvfs_{tx2,xsys5v,xsum4}_r*.log with clocks_*.tsv, start_*.txt, end_*.txt.
Energy is the sum of the four INA3221 rails NeuOS reads (SYS5V, GPU, CPU, VDDRQ); NeuOS'
own figure (SYS5V only) is listed too.
"""
import csv
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
D_MS = 25.0
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

METHODS = [  # key, label, log glob, idle state for per-period energy, color
    ("pinned", "Off,\npinned max", "pinned_r*.log", "pinned", BLUE),
    ("default", "Off, Linux\ndefault", "default_r*.log", "default", YELLOW),
    ("tx2", "NeuOS,\nTX2 table", "dvfs_tx2_r*.log", "default", ORANGE),
    ("xsys5v", "NeuOS, Xavier\n(SYS5V)", "dvfs_xsys5v_r*.log", "default", AQUA),
    ("xsum4", "NeuOS, Xavier\n(4 rails)", "dvfs_xsum4_r*.log", "default", MAGENTA),
]


def jobs(log):
    """(latency ms, SYS5V mJ, 4-rail mJ) per job, warmup job 0 dropped."""
    text = Path(log).read_text()
    a = np.array(re.findall(r"xup_exec_us/xup_energy_mj: (\d+) ([\d.e+-]+)", text), dtype=float)[1:]
    r = np.array(re.findall(r"rails_mj sys5v/gpu/cpu/vddrq: ([\d.e+-]+) ([\d.e+-]+) ([\d.e+-]+) ([\d.e+-]+)",
                            text), dtype=float)[1:]
    return a[:, 0] / 1000.0, a[:, 1], r.sum(axis=1)


def idle_mw(path):
    rows = [l.split() for l in Path(path).read_text().splitlines()[1:] if l.strip()]
    return float(np.mean([sum(float(x) for x in r[:4]) for r in rows]))  # 4-rail sum


def table(path):
    return [float(x) for x in Path(path).read_text().split("\t") if x.strip()]


def main(cdir, ht, hx5, hx4, hcsv, figs):
    cdir, figs = Path(cdir), Path(figs)
    figs.mkdir(parents=True, exist_ok=True)
    idle = {s: idle_mw(cdir / f"idle_{s}.tsv") for s in ("pinned", "default")}
    res = {}
    for key, label, pat, st, color in METHODS:
        runs = []
        for log in sorted(cdir.glob(pat)):
            lat, sys5v, en = jobs(log)
            # energy per 25 ms period: the job, plus idle power for the rest of the period
            per = en + idle[st] / 1000.0 * np.clip(D_MS - lat, 0, None)
            runs.append(dict(log=log.name, lat=lat, sys5v=sys5v, en=en, per=per))
        if runs:
            res[key] = (label, st, color, runs)

    # table
    print(f"Idle power (4 rails): pinned {idle['pinned']:.0f} mW, default {idle['default']:.0f} mW\n")
    print("| Method | Runs x jobs | p50 latency (ms) | p99 latency (ms) | Jobs over 25 ms | "
          "Energy per job, 4 rails (mJ) | Energy per 25 ms period, 4 rails (mJ) | NeuOS' SYS5V energy per job (mJ) |")
    print("|---|---|---|---|---|---|---|---|")
    for key, (label, st, color, runs) in res.items():
        lat = np.concatenate([r["lat"] for r in runs])
        miss = [100 * (r["lat"] > D_MS).mean() for r in runs]
        en = [r["en"].mean() for r in runs]
        per = [r["per"].mean() for r in runs]
        s5 = [r["sys5v"].mean() for r in runs]
        rng = lambda v, f: f"{np.mean(v):{f}} ({min(v):{f}}-{max(v):{f}})"
        print(f"| {label.replace(chr(10), ' ')} | {len(runs)} x {len(runs[0]['lat'])} | "
              f"{np.percentile(lat, 50):.2f} | {np.percentile(lat, 99):.2f} | {rng(miss, '.0f')}% | "
              f"{rng(en, '.1f')} | {rng(per, '.1f')} | {rng(s5, '.1f')} |")
    print("\nPer run:")
    for key, (label, st, color, runs) in res.items():
        for r in runs:
            print(f"  {r['log']}: p50 {np.median(r['lat']):.2f} ms, over 25 ms {100 * (r['lat'] > D_MS).mean():.0f}%, "
                  f"{r['en'].mean():.1f} mJ/job, {r['per'].mean():.1f} mJ/period")

    # figure: latency distribution and energy per period, one column per method
    keys = list(res)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    ax = axes[0]
    for x, k in enumerate(keys):
        label, st, color, runs = res[k]
        lat = np.concatenate([r["lat"] for r in runs])
        bp = ax.boxplot(lat, positions=[x], widths=0.5, showfliers=False, patch_artist=True)
        for part in ("boxes",):
            for b in bp[part]:
                b.set(facecolor=color, edgecolor=INK, linewidth=0.8, alpha=0.85)
        for part in ("whiskers", "caps", "medians"):
            for b in bp[part]:
                b.set(color=INK, linewidth=0.8)
    ax.axhline(D_MS, color=INK, lw=1, ls="--")
    ax.text(0.02, D_MS * 1.08, "deadline 25 ms", transform=ax.get_yaxis_transform(), fontsize=8)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylabel("latency per job (ms)")
    ax.set_xticks(range(len(keys)), [res[k][0] for k in keys], fontsize=7)
    ax.set_title("Latency (all runs pooled; box 25-75%, whiskers 1.5 IQR)", fontsize=8, loc="left")

    ax = axes[1]
    for x, k in enumerate(keys):
        label, st, color, runs = res[k]
        vals = [r["per"].mean() for r in runs]
        ax.bar(x, np.mean(vals), width=0.55, color=color, edgecolor=INK, linewidth=0.6)
        ax.scatter([x] * len(vals), vals, s=12, color=INK, zorder=3)
        ax.text(x, np.mean(vals) * 1.03, f"{np.mean(vals):.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("energy per 25 ms period (mJ, 4 rails)")
    ax.set_xticks(range(len(keys)), [res[k][0] for k in keys], fontsize=7)
    ax.set_title("Energy per period (bar: mean of runs; dots: runs)", fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(figs / "compare_latency_energy.png", dpi=200)
    plt.close(fig)

    # figure: the two tables, by layer index as NeuOS reads them
    names = [row["layer"] for row in csv.DictReader(open(hcsv))]
    ut, u5, u4 = table(ht)[: len(names)], table(hx5), table(hx4)
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    x = np.arange(len(names))
    ax.bar(x - 0.27, ut, width=0.26, color=ORANGE, label="TX2 table (shipped)")
    ax.bar(x, u5, width=0.26, color=AQUA, label="Xavier, SYS5V as total")
    ax.bar(x + 0.27, u4, width=0.26, color=MAGENTA, label="Xavier, 4 rails as total")
    ax.axhline(3, color=INK, lw=0.8, ls="--",
               label="3: schedule() lowers the configuration above, raises it below")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xticks(x, names, rotation=60, fontsize=7)
    ax.set_ylabel("Uncertainty")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.28), ncol=2)
    fig.tight_layout()
    fig.savefig(figs / "hashtable_tx2_vs_xavier.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main(*sys.argv[1:7])
