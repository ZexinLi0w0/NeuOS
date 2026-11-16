"""DVFS configurations and Uncertainty files (setup_DVFS / myCompare in net.cpp)."""
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Config:
    id: int
    cpu_freq_khz: int
    cpu_mask: Tuple[int, ...]
    gpu_freq_hz: int
    mem_freq: int  # EMC rate in kHz on Orin (MHz in the original AGX files)

    @property
    def cores(self):
        return sum(1 for m in self.cpu_mask if m == 1)


def sort_key(c):
    """AGX myCompare: mem_freq, CPU_freq, GPU_freq, number of online cores."""
    return (c.mem_freq, c.cpu_freq_khz, c.gpu_freq_hz, c.cores)


def sort_configs(configs):
    """Ascending order, so a higher index means a faster configuration.

    std::sort is not stable; ties are kept in input order here.
    """
    return sorted(configs, key=sort_key)


def parse_config_line(line):
    """``ID CPU_freq mask... GPU_freq mem_freq`` (AGX format, any mask length)."""
    t = [int(x) for x in line.split()]
    return Config(t[0], t[1], tuple(t[2:-2]), t[-2], t[-1])


def format_config_line(c):
    return " ".join(str(x) for x in (c.id, c.cpu_freq_khz, *c.cpu_mask, c.gpu_freq_hz, c.mem_freq))


def load_configs(path):
    with open(path) as f:
        return [parse_config_line(line) for line in f if line.strip()]


def save_configs(path, configs):
    with open(path, "w") as f:
        f.writelines(format_config_line(c) + "\n" for c in configs)


def load_uncertainty(path):
    """First line, tab-separated (generate_hashtable.py writes a trailing tab)."""
    with open(path) as f:
        line = f.readline()
    return [float(t) for t in line.split("\t") if t.strip()]


def format_uncertainty(values):
    """Same layout as generate_hashtable.py: ``value\\t`` per stage, no newline."""
    return "".join("%s\t" % v for v in values)
