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

    def __init__(self, root="/"):
        self.root = root
        self._cpu_khz = None

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

    def set_cpu_freq(self, khz):
        self._cpu_khz = khz
        for pol in self._policies():
            cpus = [int(c) for c in _read(os.path.join(pol, "related_cpus")).split()]
            if any(self._online(c) for c in cpus):  # an inactive policy rejects writes
                self._pin(pol, "scaling_min_freq", "scaling_max_freq", khz)

    def set_cpu_mask(self, mask):
        # online first, then offline; cpu0 is never taken offline
        for want in (1, 0):
            for cpu, m in enumerate(mask):
                if cpu > 0 and m == want:
                    _write(self._p("%s/cpu%d/online" % (CPU_DIR, cpu)), m)
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
    def enumerate_states(self):
        cpu = None
        for pol in self._policies():
            f = set(_ints(_read(os.path.join(pol, "scaling_available_frequencies"))))
            cpu = f if cpu is None else cpu & f
        return {
            "cpu_freqs_khz": sorted(cpu or ()),
            "gpu_freqs_hz": _ints(_read(os.path.join(self._gpu_dir(), "available_frequencies"))),
            "emc_freqs_khz": _ints(_read(self._p(BPMP_DIR + "/emc/possible_rates"))),
            "n_cpus": len(self._cpus()),
        }

    def restore(self):
        """All cores online, CPU/GPU min/max back to the full range, EMC unlocked
        at its highest rate. Best effort: a failing knob only warns."""
        self._cpu_khz = None
        steps = [lambda: self.set_cpu_mask((1,) * len(self._cpus()))]
        for pol in self._policies():
            f = _ints(_read(os.path.join(pol, "scaling_available_frequencies")))
            steps.append(lambda pol=pol, f=f: (_write(os.path.join(pol, "scaling_max_freq"), f[-1]),
                                               _write(os.path.join(pol, "scaling_min_freq"), f[0])))

        def gpu():
            d = self._gpu_dir()
            f = _ints(_read(os.path.join(d, "available_frequencies")))
            _write(os.path.join(d, "max_freq"), f[-1])
            _write(os.path.join(d, "min_freq"), f[0])

        def emc():
            f = _ints(_read(self._p(BPMP_DIR + "/emc/possible_rates")))
            _write(self._p(BPMP_DIR + "/clk/emc/rate"), f[-1] * 1000)
            _write(self._p(BPMP_DIR + "/clk/emc/mrq_rate_locked"), 0)

        for step in steps + [gpu, emc]:
            try:
                step()
            except OSError as e:
                warnings.warn("DVFS restore step failed: %s" % e)


def build_config_space(cpu_freqs, gpu_freqs, emc_freqs, core_counts, n_cpus,
                       cpu_stride=1, gpu_stride=1, emc_stride=1):
    """Cartesian product of the knobs; masks are contiguous (first k cores online).
    ``*_stride`` keeps every n-th frequency (ascending). Returned sorted (myCompare)."""
    space = itertools.product(sorted(emc_freqs)[::emc_stride], sorted(cpu_freqs)[::cpu_stride],
                              sorted(gpu_freqs)[::gpu_stride], sorted(core_counts))
    configs = [Config(i, cpu, (1,) * k + (0,) * (n_cpus - k), gpu, emc)
               for i, (emc, cpu, gpu, k) in enumerate(space)]
    return sort_configs(configs)
