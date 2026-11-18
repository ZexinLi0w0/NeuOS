"""Estimate the best *static* configuration from the profile sweep: the profiled configuration
with the lowest energy per 25 ms period among those whose whole-network latency fits the
deadline. An estimate from the per-layer profile, not a run: profile_layers.cpp synchronizes
the GPU per layer, so its whole-network numbers differ from classify.bin's. The script also
prints the profile's value for the max configuration next to the measured pinned run, as a
calibration.

usage: static_oracle.py PROFILE_DIR COMPARE_DIR
"""
import re
import sys
from pathlib import Path

import numpy as np

D_MS = 25.0


def idle_w(path):
    rows = [l.split() for l in Path(path).read_text().splitlines()[1:] if l.strip()]
    return float(np.mean([sum(float(x) for x in r[:4]) for r in rows])) / 1000.0


def main(prof, comp):
    comp = Path(comp)
    p_default, p_pinned = idle_w(comp / "idle_default.tsv"), idle_w(comp / "idle_pinned.tsv")
    rows = []
    for p in Path(prof).glob("[0-9]*.log"):
        lines = p.read_text().splitlines()
        h = lines[0].split()  # "# config ID cpu KHZ mask M gpu HZ emc MHZ"
        data = [l.split() for l in lines if l and not l.startswith("#")]
        lat = sum(float(r[1]) for r in data) / 1000.0
        en = sum(sum(float(x) for x in r[2:6]) for r in data)
        rows.append(dict(id=int(p.stem), cpu=int(h[4]), mask=h[6], gpu=int(h[8]) // 1_000_000, lat=lat, en=en))
    ok = [r for r in rows if r["lat"] <= D_MS]
    for r in ok:
        r["per"] = r["en"] + p_default * (D_MS - r["lat"])
    ok.sort(key=lambda r: r["per"])
    print(f"{len(rows)} profiled configurations, {len(ok)} fit {D_MS:g} ms (whole-network profile latency)")
    print(f"idle power, 4 rails: Linux default {p_default:.3f} W, pinned {p_pinned:.3f} W\n")
    print("| Static configuration (profile estimate) | Latency (ms) | Energy per job (mJ) | Energy per period (mJ) |")
    print("|---|---|---|---|")
    for r in ok[:5]:
        print(f"| id {r['id']}: CPU {r['cpu'] / 1e3:.0f} MHz, cores {r['mask']}, GPU {r['gpu']} MHz | "
              f"{r['lat']:.2f} | {r['en']:.1f} | {r['per']:.1f} |")
    mx = max(rows, key=lambda r: (r["cpu"], r["gpu"], r["mask"]))
    mx_per = mx["en"] + p_pinned * (D_MS - mx["lat"])
    meas = []
    for log in sorted(comp.glob("pinned_r*.log")):
        txt = log.read_text()
        lat = np.array(re.findall(r"xup_exec_us/xup_energy_mj: (\d+)", txt), float)[1:] / 1000
        rails = np.array(re.findall(r"rails_mj sys5v/gpu/cpu/vddrq: ([\d.e+-]+) ([\d.e+-]+) ([\d.e+-]+) ([\d.e+-]+)",
                                    txt), float)[1:].sum(axis=1)
        meas.append(np.mean(rails + p_pinned * np.clip(D_MS - lat, 0, None)))
    scale = np.mean(meas) / mx_per
    print(f"\ncalibration: max configuration (id {mx['id']}) profile {mx['lat']:.2f} ms, {mx_per:.1f} mJ/period; "
          f"measured pinned runs {np.mean(meas):.1f} mJ/period (x{scale:.2f})")
    print(f"best static configuration scaled by the same factor: {ok[0]['per'] * scale:.1f} mJ/period")


if __name__ == "__main__":
    main(*sys.argv[1:3])
