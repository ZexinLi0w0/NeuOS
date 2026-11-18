"""Figures and the summary table for xavier-jetpack5/README.md, from the raw logs in ../results.

usage: python plot_results.py RESULTS_DIR FIG_DIR
"""
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# categorical slots 1-3 of the reference palette (validated all-pairs for 3 series)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DEADLINE_MS = 25

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

RUNS = [  # (label, log, clocks tag or None, color)
    ("DVFS off (MAXN, clocks pinned)", "off_200.log", None, BLUE),
    ("DVFS on, EMC as shipped (204 MHz)", "dvfs_on.log", "on", ORANGE),
    ("DVFS on, EMC unit fix (2133 MHz)", "dvfs_emcfix.log", "emcfix", AQUA),
]


def jobs(log):
    """Per-job (latency ms, energy mJ) from NeuOS' '++++ xup_exec_us/xup_energy_mj: us mJ' lines."""
    rows = re.findall(r"xup_exec_us/xup_energy_mj: (\d+) ([\d.e+-]+)", Path(log).read_text())
    a = np.array(rows, dtype=float)
    return a[:, 0] / 1000.0, a[:, 1]


def ncores(spec):
    n = 0
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        n += int(hi or lo) - int(lo) + 1
    return n


def clocks(res, tag):
    t0 = int((res / f"start_{tag}.txt").read_text())
    t1 = int((res / f"end_{tag}.txt").read_text())
    rows = [ln.split() for ln in (res / f"clocks_{tag}.tsv").read_text().splitlines()[1:]]
    rows = [r for r in rows if len(r) >= 6 and t0 <= int(r[0]) <= t1]
    return {
        "t": np.array([(int(r[0]) - t0) / 1000.0 for r in rows]),
        "wall_s": (t1 - t0) / 1000.0,
        "cpu": np.array([int(r[1]) for r in rows]) / 1e6,
        "gpu": np.array([int(r[2]) for r in rows]) / 1e6,
        "emc": np.array([int(r[3]) for r in rows]) / 1e6,
        "cores": np.array([ncores(r[4]) for r in rows]),
        "fds": np.array([int(r[6]) for r in rows]) if len(rows[0]) > 6 else None,
    }


def main(res, figs):
    res, figs = Path(res), Path(figs)
    figs.mkdir(parents=True, exist_ok=True)
    runs = [r for r in RUNS if (res / r[1]).exists()]

    # 1. latency and energy per job: two panels sharing x (never a dual axis)
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 4.8), sharex=True)
    for label, log, _, color in runs:
        lat, en = jobs(res / log)
        x = np.arange(1, len(lat))
        axes[0].plot(x, lat[1:], color=color, lw=1.4, label=label)
        axes[1].plot(x, en[1:], color=color, lw=1.4, label=label)
    axes[0].axhline(DEADLINE_MS, color=INK, lw=1, ls="--")
    axes[0].text(0.98, DEADLINE_MS * 1.12, f"deadline {DEADLINE_MS} ms", color=INK, fontsize=8,
                 ha="right", transform=axes[0].get_yaxis_transform())
    plain = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")
    for ax, ylabel in zip(axes, ("latency per job (ms)", "energy per job (mJ)")):
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(plain)
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_ylabel(ylabel)
    axes[1].set_xlabel("job (warmup job 0 omitted)")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.32), ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "latency_energy_per_job.png", dpi=200)
    plt.close(fig)

    # 2. clocks set by NeuOS in the DVFS-on runs
    dvfs = [(label, clocks(res, tag), color) for label, _, tag, color in runs if tag]
    panels = [("CPU (GHz)", "cpu", 2.2656), ("GPU (MHz)", "gpu", 1377),
              ("EMC (MHz)", "emc", 2133), ("online CPU cores", "cores", 8)]
    fig, axes = plt.subplots(len(panels), 1, figsize=(6.4, 5.6), sharex=True)
    for ax, (ylabel, key, ceiling) in zip(axes, panels):
        for label, c, color in dvfs:
            ax.step(c["t"], c[key], where="post", color=color, lw=1.0, label=label)
        ax.axhline(ceiling, color=MUTED, lw=0.8, ls=":")
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_ylim(bottom=0, top=ceiling * 1.12)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.45), ncol=2, fontsize=8)
    axes[-1].text(0.99, 0.08, "dotted: value in the DVFS-off run", transform=axes[-1].transAxes,
                  ha="right", fontsize=7, color=MUTED)
    axes[-1].set_xlabel("time since classify.bin start (s); one sample every ~0.75 s")
    fig.tight_layout()
    fig.savefig(figs / "dvfs_on_clocks.png", dpi=200)
    plt.close(fig)

    # summary table for the README
    print("| Run | Jobs | p50 (ms) | p99 (ms) | Max (ms) | Over 25 ms | Mean energy (mJ/job) | Wall time per job (s) |")
    print("|---|---|---|---|---|---|---|---|")
    for label, log, tag, _ in runs:
        lat, en = jobs(res / log)
        lat, en = lat[1:], en[1:]
        wall = f"{clocks(res, tag)['wall_s'] / (len(lat) + 1):.1f}" if tag else "-"
        print(f"| {label} | {len(lat)} | {np.percentile(lat, 50):.2f} | {np.percentile(lat, 99):.2f} | "
              f"{lat.max():.2f} | {(lat > DEADLINE_MS).sum()} ({100 * (lat > DEADLINE_MS).mean():.0f}%) | "
              f"{en.mean():.1f} | {wall} |")
    for label, c, _ in dvfs:
        print(f"\n{label}: EMC at 204 MHz in {100 * (c['emc'] < 300).mean():.0f}% of samples; "
              f"GPU median {np.median(c['gpu']):.0f} MHz; CPU median {np.median(c['cpu']):.2f} GHz; "
              f"online cores median {np.median(c['cores']):.0f} (min {c['cores'].min()}); "
              f"max open fds {c['fds'].max() if c['fds'] is not None else 'n/a'}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
