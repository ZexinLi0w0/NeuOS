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


class EnergyMeter:
    """Samples VDD_IN (or the sum of all rails if absent) every ``period_s`` in a
    background thread and integrates (trapezoid) to mJ; mark() returns the energy
    since the previous mark()."""

    def __init__(self, root="/", period_s=0.01, rail="VDD_IN", clock=time.monotonic):
        rails = find_rails(root)
        if not rails:
            raise RuntimeError("no INA3221 rails found under " + root)
        self._rails = {rail: rails[rail]} if rail in rails else rails
        self.period_s, self.clock = period_s, clock
        self._lock = threading.Lock()
        self._mj, self._last = 0.0, None
        self._stop = threading.Event()
        self._thread = None

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
        self._sample()
        with self._lock:
            mj, self._mj = self._mj, 0.0
        return mj
