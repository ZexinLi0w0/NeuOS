"""1 Hz board sampler (run as root; extends the AudiOS motivation sampler) + energy integral.

Each line: wall time, every thermal zone (tj-thermal included), MemAvailable, GPU devfreq
cur_freq vs BPMP actual gpcNclk (max over GPCs), CPU cpufreq cur per policy vs BPMP nafll cluster clocks, EMC
rate, INA3221 rail power, the thermal cooling-device states, online cpus, fan PWM, and the
cumulative board energy (neuos_modern EnergyMeter: VDD_IN on the Nano, sum of rails on the AGX,
integrated at 20 Hz).

    python sampler.py OUT.jsonl STOPFILE
"""
import glob
import json
import os
import sys
import time

from common import mem_avail_mb


def rd(p, f=int):
    try:
        with open(p) as fh:
            return f(fh.read().strip())
    except Exception:
        return None


def main():
    out, stopfile = sys.argv[1], sys.argv[2]
    neuos_dir = sys.argv[3] if len(sys.argv) > 3 else None
    if neuos_dir:
        sys.path.insert(0, neuos_dir)
    from neuos_modern.energy import EnergyMeter, find_rails
    meter = EnergyMeter(period_s=0.05).start()
    TZ = {rd(z + "/type", str): z + "/temp" for z in glob.glob("/sys/class/thermal/thermal_zone*")}
    CD = {rd(c + "/type", str): c + "/cur_state" for c in glob.glob("/sys/class/thermal/cooling_device*")}
    GPU_DEV = (glob.glob("/sys/devices/platform/*/17000000.gpu/devfreq/*/cur_freq") + [None])[0]
    POL = sorted(glob.glob("/sys/devices/system/cpu/cpufreq/policy*"))
    PWM = (glob.glob("/sys/devices/platform/pwm-fan/hwmon/hwmon*/pwm1") + [None])[0]
    BPMP = "/sys/kernel/debug/bpmp/debug/clk/"
    gpcs = sorted(os.path.basename(p) for p in glob.glob(BPMP + "gpc[0-9]clk"))   # AGX: gpc0clk, gpc1clk
    clusters = sorted(os.path.basename(p) for p in glob.glob(BPMP + "nafll_cluster*_core"))
    rails = find_rails("/")
    f = open(out, "a")
    cum = 0.0
    try:
        while not os.path.exists(stopfile):
            t = time.time()
            s = {"t": round(t, 3)}
            s["temp"] = {k: (v / 1000 if v is not None else None) for k, v in ((k, rd(p)) for k, p in TZ.items())}
            s["avail_mb"] = mem_avail_mb()
            s["gpu_devfreq_mhz"] = (rd(GPU_DEV) or 0) // 1000000 if GPU_DEV else None
            s["gpc_mhz"] = {g: (rd(BPMP + g + "/rate") or 0) // 1000000 for g in gpcs}
            s["gpu_actual_mhz"] = max(s["gpc_mhz"].values()) if gpcs else None
            s["cpu_cur_mhz"] = {os.path.basename(p): (rd(p + "/scaling_cur_freq") or 0) // 1000 for p in POL}
            s["cpu_actual_mhz"] = {c: (rd(BPMP + c + "/rate") or 0) // 1000000 for c in clusters}
            s["emc_mhz"] = (rd(BPMP + "emc/rate") or 0) // 1000000
            s["power_mw"] = {k: round((rd(v) or 0) * (rd(c) or 0) / 1000) for k, (v, c) in rails.items()}
            s["cdev"] = {k: rd(v) for k, v in CD.items()}
            s["online"] = rd("/sys/devices/system/cpu/online", str)
            s["pwm"] = rd(PWM) if PWM else None
            cum += meter.mark()
            s["energy_mj"] = round(cum, 1)
            f.write(json.dumps(s) + "\n")
            f.flush()
            time.sleep(max(0.0, 1.0 - (time.time() - t)))
    finally:
        meter.stop()
        f.close()


if __name__ == "__main__":
    main()
