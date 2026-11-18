"""Offline profiler: replaces hashtables/generate_hashtable.py.

For every DVFS configuration: apply it, run the job ``repeats`` times without any
scheduling or approximation, and average each stage's latency (us) and energy (mJ).
The job is any callable ``job_fn(ctl)`` written against the controller API
(``ctl.begin_job`` / ``before_stage`` / ``after_stage`` / ``end``), so every adapter
can be profiled unchanged.

CLI::

    python -m neuos_modern.profile --runner mymod:job_fn --configs all.config \\
        --every 10 --repeats 3 --csv profile.csv --uncertainty Uncertainty.txt [--dry-run]
    python -m neuos_modern.profile --from-csv profile.csv --uncertainty Uncertainty.txt
"""
import argparse
import csv
import importlib
import time
from collections import defaultdict

from .config import format_uncertainty, load_configs
from .energy import NullEnergyMeter


class _Recorder:
    """Stand-in controller: never schedules, never approximates, records stages."""

    def __init__(self, energy, clock):
        self.energy, self.clock, self.records = energy, clock, []

    def begin_job(self, n_stages):
        return self

    def before_stage(self, i, approximable=False):
        self.energy.mark()
        self._t0 = self.clock()
        return False

    def after_stage(self, i):
        self.records.append((i, (self.clock() - self._t0) * 1e6, self.energy.mark()))

    def end(self):
        return {}


def profile(job_fn, configs, backend, energy=None, repeats=1, clock=time.perf_counter):
    """Returns rows ``(config_id, stage, latency_us, energy_mj)`` (means over repeats)."""
    energy = energy or NullEnergyMeter()
    rows, prev = [], None
    for cfg in configs:
        backend.apply(prev, cfg)
        prev = cfg
        acc = defaultdict(lambda: [0.0, 0.0, 0])
        for _ in range(repeats):
            rec = _Recorder(energy, clock)
            job_fn(rec)
            for i, lat, mj in rec.records:
                a = acc[i]
                a[0] += lat
                a[1] += mj
                a[2] += 1
        rows += [(cfg.id, i, a[0] / a[2], a[1] / a[2]) for i, a in sorted(acc.items())]
    return rows


def compute_uncertainty(rows):
    """Per stage: max over configs of normEnergy / normLatency, where each value is
    normalised by its max over configs (0 when that max, or normLatency, is 0)."""
    by_stage = defaultdict(list)
    for _, stage, lat, mj in rows:
        by_stage[stage].append((lat, mj))
    out = []
    for stage in range(max(by_stage) + 1 if by_stage else 0):
        vals = by_stage.get(stage, [])
        max_lat = max([0] + [lat for lat, _ in vals])
        max_mj = max([0] + [mj for _, mj in vals])
        best = 0
        for lat, mj in vals:
            n_lat = lat / max_lat if max_lat else 0
            n_mj = mj / max_mj if max_mj else 0
            best = max(best, n_mj / n_lat if n_lat else 0)
        out.append(best)
    return out


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config_id", "stage", "latency_us", "energy_mj"])
        w.writerows(rows)


def read_csv(path):
    with open(path) as f:
        return [(int(r["config_id"]), int(r["stage"]), float(r["latency_us"]), float(r["energy_mj"]))
                for r in csv.DictReader(f)]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m neuos_modern.profile", description=__doc__.split("\n")[0])
    ap.add_argument("--runner", help="module:function, called as function(ctl) to run one job")
    ap.add_argument("--configs", help="config file (ID CPU_freq mask... GPU_freq mem_freq)")
    ap.add_argument("--every", type=int, default=1, help="profile every n-th config")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true", help="do not touch DVFS / INA3221")
    ap.add_argument("--from-csv", help="only recompute the Uncertainty line from a profile CSV")
    ap.add_argument("--csv", default="profile.csv")
    ap.add_argument("--uncertainty", default="Uncertainty.txt")
    a = ap.parse_args(argv)

    if a.from_csv:
        rows = read_csv(a.from_csv)
    else:
        if not (a.runner and a.configs):
            ap.error("--runner and --configs are required unless --from-csv is given")
        from .dvfs import DryRunBackend, OrinSysfsBackend
        from .energy import EnergyMeter
        mod, fn = a.runner.split(":")
        job_fn = getattr(importlib.import_module(mod), fn)
        configs = load_configs(a.configs)[::a.every]
        backend = DryRunBackend() if a.dry_run else OrinSysfsBackend()
        energy = NullEnergyMeter() if a.dry_run else EnergyMeter().start()
        try:
            rows = profile(job_fn, configs, backend, energy, a.repeats)
        finally:
            energy.stop()
            backend.restore()
        write_csv(a.csv, rows)
    with open(a.uncertainty, "w") as f:
        f.write(format_uncertainty(compute_uncertainty(rows)))


if __name__ == "__main__":
    main()
