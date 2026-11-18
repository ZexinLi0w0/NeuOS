"""Energy measurement: replaces energymon with the Jetson INA3221 hwmon rails."""
import glob
import os
import re
import threading
import time

INA_GLOB = "sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/in*_label"


def find_rails(root="/"):
    """``{label: (voltage_path_mV, current_path_mA)}`` for every labelled INA3221 channel."""
    rails = {}
    for label_path in sorted(glob.glob(os.path.join(root, INA_GLOB))):
        d, name = os.path.split(label_path)
        idx = re.match(r"in(\d+)_label", name).group(1)
        volt, curr = os.path.join(d, "in%s_input" % idx), os.path.join(d, "curr%s_input" % idx)
        if os.path.exists(volt) and os.path.exists(curr):
            with open(label_path) as f:
                rails[f.read().strip()] = (volt, curr)
    return rails


def _read_int(path):
    with open(path) as f:
        return int(f.read().strip())


def read_power_mw(rails):
    """``{label: mW}`` with power = mV * mA / 1000."""
    return {k: _read_int(v) * _read_int(c) / 1000.0 for k, (v, c) in rails.items()}


class NullEnergyMeter:
    def start(self):
        return self

    def stop(self):
        pass

    def mark(self):
        return 0.0


SUM_LABEL = "sum of shunt voltages"


def total_power_rails(rails, rail="VDD_IN"):
    """The rails whose sum is the board's total power: ``rail`` (VDD_IN, the module
    input on Orin Nano) if present, else every rail except the INA3221 "sum of shunt
    voltages" channel (AGX Orin: VDD_GPU_SOC + VDD_CPU_CV + VIN_SYS_5V0 + VDDQ_VDD2_1V8AO)."""
    if rail in rails:
        return {rail: rails[rail]}
    return {k: v for k, v in rails.items() if k.lower() != SUM_LABEL}


class EnergyMeter:
    """Samples the total power (see total_power_rails) every ``period_s`` in a
    background thread and integrates (trapezoid) to mJ; mark() returns the energy
    since the previous mark().

    mark() does not read the rails itself (each INA3221 read is an I2C transfer,
    ~1 ms per rail on Orin, which dominated per-layer stages): it extends the
    integral from the last sample to now with the last sampled power. Energy that
    the next trapezoid adds over that extrapolation is credited to the next mark(),
    so the total is conserved."""

    def __init__(self, root="/", period_s=0.01, rail="VDD_IN", clock=time.monotonic):
        rails = total_power_rails(find_rails(root), rail)
        if not rails:
            raise RuntimeError("no INA3221 rails found under " + root)
        self._rails = rails
        self.period_s, self.clock = period_s, clock
        self._lock = threading.Lock()
        self._mj, self._last = 0.0, None   # integral up to the last sample
        self._marked = 0.0                 # integral (extrapolated) at the last mark
        self._stop = threading.Event()
        self._thread = None

    @property
    def rails(self):
        return sorted(self._rails)

    def power_mw(self):
        return sum(read_power_mw(self._rails).values())

    def _sample(self):
        p = sum(read_power_mw(self._rails).values())
        t = self.clock()
        with self._lock:
            if self._last is not None:
                t0, p0 = self._last
                self._mj += (p + p0) / 2.0 * (t - t0)  # mW * s = mJ
            self._last = (t, p)

    def _loop(self):
        while not self._stop.wait(self.period_s):
            self._sample()

    def start(self):
        self._sample()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join()

    def mark(self):
        t = self.clock()
        with self._lock:
            if self._last is None:
                return 0.0
            t0, p0 = self._last
            total = self._mj + p0 * max(0.0, t - t0)
            mj, self._marked = total - self._marked, total
        return mj
