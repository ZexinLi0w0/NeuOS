"""DVFS actuation: replaces include/DVFS/setconfig.h with Jetson Orin (L4T R36) sysfs."""
import glob
import itertools
import os
import warnings

from .config import Config, sort_configs

CPU_DIR = "sys/devices/system/cpu"
GPU_DEVFREQ = "sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu"
GPU_DEVFREQ_GLOB = "sys/devices/platform/*/17000000.gpu/devfreq/*"
BPMP_DIR = "sys/kernel/debug/bpmp/debug"


class Backend:
    """apply() actuates only the knobs that differ, in the order of net.cpp
    (core masks, CPU freq, GPU freq, EMC freq). ``prev=None`` sets every knob."""

    def apply(self, prev, cfg):
        if prev is None or prev.cpu_mask != cfg.cpu_mask:
            self.set_cpu_mask(cfg.cpu_mask)
        if prev is None or prev.cpu_freq_khz != cfg.cpu_freq_khz:
            self.set_cpu_freq(cfg.cpu_freq_khz)
        if prev is None or prev.gpu_freq_hz != cfg.gpu_freq_hz:
            self.set_gpu_freq(cfg.gpu_freq_hz)
        if prev is None or prev.mem_freq != cfg.mem_freq:
            self.set_emc_freq(cfg.mem_freq)


class DryRunBackend(Backend):
    """Records knob writes as ``(knob, value)`` tuples instead of touching hardware."""

    def __init__(self, states=None):
        self.writes = []
        self.states = states or {}

    def set_cpu_mask(self, mask):
        self.writes.append(("cpu_mask", tuple(mask)))

    def set_cpu_freq(self, khz):
        self.writes.append(("cpu_freq", khz))

    def set_gpu_freq(self, hz):
        self.writes.append(("gpu_freq", hz))

    def set_emc_freq(self, khz):
        self.writes.append(("emc_freq", khz))

    def enumerate_states(self):
        return self.states

    def restore(self):
        self.writes.append(("restore", None))


def _read(path):
    with open(path) as f:
        return f.read().strip()


def _write(path, value):
    with open(path, "w") as f:
        f.write(str(value))


def _ints(text):
    return sorted({int(t) for t in text.split() if t.isdigit()})


class OrinSysfsBackend(Backend):
    """Jetson Orin sysfs/debugfs backend (root required). ``root`` prefixes every
    path so tests can point it at a fake tree."""

    def __init__(self, root="/", baseline=None):
        self.root = root
        self._cpu_khz = None
        self.write_errors = 0
        # exact state restore() writes back; pass a saved snapshot() when another
        # process may already have changed the knobs
        self.baseline = baseline if baseline is not None else self.snapshot()

    def _p(self, rel):
        return os.path.join(self.root, rel)

    # --- CPU -----------------------------------------------------------------
    def _policies(self):
        return sorted(glob.glob(self._p(CPU_DIR + "/cpufreq/policy*")))

    def _cpus(self):
        return sorted(int(os.path.basename(d)[3:]) for d in glob.glob(self._p(CPU_DIR + "/cpu[0-9]*")))

    def _online(self, cpu):
        path = self._p("%s/cpu%d/online" % (CPU_DIR, cpu))
        return not os.path.exists(path) or _read(path) == "1"

    @staticmethod
    def _pin(dir_, min_name, max_name, value):
        """Write min and max to ``value`` in the order that keeps min <= max."""
        mn, mx = os.path.join(dir_, min_name), os.path.join(dir_, max_name)
        order = (mx, mn) if value > int(_read(mx)) else (mn, mx)
        for path in order:
            _write(path, value)

    def _tolerant(self, fn, *args):
        """Per-policy / per-cpu writes can fail with EBUSY when another process is
        hot-plugging the same cpus (concurrent NeuOS clients): warn, count, go on."""
        try:
            fn(*args)
        except OSError as e:
            self.write_errors += 1
            warnings.warn("DVFS write failed (skipped): %s" % e)

    def set_cpu_freq(self, khz):
        self._cpu_khz = khz
        for pol in self._policies():
            cpus = [int(c) for c in _read(os.path.join(pol, "related_cpus")).split()]
            if any(self._online(c) for c in cpus):  # an inactive policy rejects writes
                self._tolerant(self._pin, pol, "scaling_min_freq", "scaling_max_freq", khz)

    def set_cpu_mask(self, mask):
        # online first, then offline; cpu0 is never taken offline
        for want in (1, 0):
            for cpu, m in enumerate(mask):
                if cpu > 0 and m == want:
                    self._tolerant(_write, self._p("%s/cpu%d/online" % (CPU_DIR, cpu)), m)
        if self._cpu_khz is not None:  # re-pin policies that just came back online
            self.set_cpu_freq(self._cpu_khz)

    # --- GPU -----------------------------------------------------------------
    def _gpu_dir(self):
        d = self._p(GPU_DEVFREQ)
        if os.path.isdir(d):
            return d
        found = sorted(glob.glob(self._p(GPU_DEVFREQ_GLOB)))
        if not found:
            raise FileNotFoundError("GPU devfreq directory not found under " + self.root)
        return found[0]

    def set_gpu_freq(self, hz):
        self._pin(self._gpu_dir(), "min_freq", "max_freq", hz)

    # --- EMC -----------------------------------------------------------------
    def set_emc_freq(self, khz):
        _write(self._p(BPMP_DIR + "/clk/emc/mrq_rate_locked"), 1)
        _write(self._p(BPMP_DIR + "/clk/emc/rate"), int(khz) * 1000)

    # --- discovery / restore ---------------------------------------------------
    def enumerate_states(self, capped=True):
        """Available knob values. ``capped`` keeps only CPU frequencies <= the baseline
        scaling_max_freq of every policy and GPU frequencies <= the baseline devfreq
        max_freq (the power mode's caps, e.g. 1497.6 MHz / 612 MHz in Orin Nano 15W;
        cpufreq and devfreq accept writes above them)."""
        cpu = None
        for pol in self._policies():
            f = set(_ints(_read(os.path.join(pol, "scaling_available_frequencies"))))
            cpu = f if cpu is None else cpu & f
        cpu = sorted(cpu or ())
        if capped and self.baseline.get("policies"):
            cap = min(int(p["max"]) for p in self.baseline["policies"].values())
            cpu = [f for f in cpu if f <= cap]
        gpu = _ints(_read(os.path.join(self._gpu_dir(), "available_frequencies")))
        if capped and self.baseline.get("gpu"):
            gpu = [f for f in gpu if f <= self.baseline["gpu"]["max"]]
        return {
            "cpu_freqs_khz": cpu,
            "gpu_freqs_hz": gpu,
            "emc_freqs_khz": _ints(_read(self._p(BPMP_DIR + "/emc/possible_rates"))),
            "n_cpus": len(self._cpus()),
        }

    def emc_rate_hz(self):
        return int(_read(self._p(BPMP_DIR + "/clk/emc/rate")))

    def probe_emc_rates(self, rates_khz):
        """Keep the EMC rates whose write is honoured (read back equal); some boards /
        power modes ignore EMC rate writes. Restores the baseline EMC state after."""
        ok = []
        try:
            for khz in rates_khz:
                try:
                    self.set_emc_freq(khz)
                    if self.emc_rate_hz() == int(khz) * 1000:
                        ok.append(khz)
                except OSError as e:
                    warnings.warn("EMC rate %s kHz rejected: %s" % (khz, e))
        finally:
            self._restore_emc()
        return ok

    def snapshot(self):
        """Current knob state: per policy governor / scaling_min / scaling_max, cpu
        online flags, GPU devfreq min / max, EMC rate and mrq_rate_locked. Unreadable
        parts (e.g. debugfs without root) are recorded as None and skipped by restore()."""
        snap = {"policies": {}, "online": {}, "gpu": None, "emc": None}
        for pol in self._policies():
            try:
                gov = os.path.join(pol, "scaling_governor")
                snap["policies"][os.path.basename(pol)] = {
                    "governor": _read(gov) if os.path.exists(gov) else None,
                    "min": int(_read(os.path.join(pol, "scaling_min_freq"))),
                    "max": int(_read(os.path.join(pol, "scaling_max_freq")))}
            except OSError:
                pass
        for cpu in self._cpus():
            path = self._p("%s/cpu%d/online" % (CPU_DIR, cpu))
            if os.path.exists(path):
                snap["online"][str(cpu)] = int(_read(path))
        try:
            d = self._gpu_dir()
            snap["gpu"] = {"min": int(_read(os.path.join(d, "min_freq"))),
                           "max": int(_read(os.path.join(d, "max_freq")))}
        except OSError:
            pass
        try:
            snap["emc"] = {"rate": self.emc_rate_hz(),
                           "locked": int(_read(self._p(BPMP_DIR + "/clk/emc/mrq_rate_locked")))}
        except OSError:
            pass
        return snap

    @staticmethod
    def _set_range(dir_, min_name, max_name, lo, hi):
        """Write min=lo, max=hi in an order where min <= max holds after each write."""
        mn, mx = os.path.join(dir_, min_name), os.path.join(dir_, max_name)
        if lo > int(_read(mx)):
            _write(mx, hi)
            _write(mn, lo)
        else:
            _write(mn, lo)
            _write(mx, hi)

    def _restore_emc(self):
        emc = self.baseline.get("emc")
        if emc:
            _write(self._p(BPMP_DIR + "/clk/emc/rate"), emc["rate"])
            _write(self._p(BPMP_DIR + "/clk/emc/mrq_rate_locked"), emc["locked"])

    def restore(self):
        """Write the construction-time baseline back exactly (online cpus first, then
        cpufreq governor / min / max, GPU min / max, EMC rate then lock). Best effort:
        a failing knob only warns."""
        self._cpu_khz = None
        b = self.baseline
        steps = []
        for want in (1, 0):  # online first, then offline
            for cpu, v in sorted(b["online"].items(), key=lambda kv: int(kv[0])):
                if v == want:
                    steps.append(lambda cpu=cpu, v=v: _write(self._p("%s/cpu%s/online" % (CPU_DIR, cpu)), v))
        for name, p in sorted(b["policies"].items()):
            d = self._p(CPU_DIR + "/cpufreq/" + name)

            def pol(d=d, p=p):
                if p["governor"] and _read(os.path.join(d, "scaling_governor")) != p["governor"]:
                    _write(os.path.join(d, "scaling_governor"), p["governor"])
                self._set_range(d, "scaling_min_freq", "scaling_max_freq", p["min"], p["max"])
            steps.append(pol)
        if b.get("gpu"):
            steps.append(lambda: self._set_range(self._gpu_dir(), "min_freq", "max_freq",
                                                 b["gpu"]["min"], b["gpu"]["max"]))
        steps.append(self._restore_emc)
        for step in steps:
            try:
                step()
            except OSError as e:
                warnings.warn("DVFS restore step failed: %s" % e)


def subsample(values, n):
    """``n`` evenly spaced entries of the sorted ``values``, always including the
    lowest and the highest (all of them when n >= len)."""
    v = sorted(values)
    if n >= len(v):
        return v
    if n == 1:
        return [v[-1]]
    return sorted({v[round(k * (len(v) - 1) / (n - 1))] for k in range(n)})


def build_config_space(cpu_freqs, gpu_freqs, emc_freqs, core_counts, n_cpus,
                       cpu_stride=1, gpu_stride=1, emc_stride=1):
    """Cartesian product of the knobs; masks are contiguous (first k cores online).
    ``*_stride`` keeps every n-th frequency (ascending). Returned sorted (myCompare)."""
    space = itertools.product(sorted(emc_freqs)[::emc_stride], sorted(cpu_freqs)[::cpu_stride],
                              sorted(gpu_freqs)[::gpu_stride], sorted(core_counts))
    configs = [Config(i, cpu, (1,) * k + (0,) * (n_cpus - k), gpu, emc)
               for i, (emc, cpu, gpu, k) in enumerate(space)]
    return sort_configs(configs)
