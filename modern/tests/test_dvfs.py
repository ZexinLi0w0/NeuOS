import pytest

import neuos_modern.dvfs as dvfs
from neuos_modern import Config, OrinSysfsBackend, build_config_space
from neuos_modern.energy import EnergyMeter, find_rails, read_power_mw

CPU_F = "115200 729600 1510400"
GPU_F = "306000000 510000000 1020000000"


def put(root, rel, text):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


@pytest.fixture
def tree(tmp_path):
    cpu = "sys/devices/system/cpu"
    for n in range(6):
        put(tmp_path, f"{cpu}/cpu{n}/topology/core_id", str(n))
        if n:
            put(tmp_path, f"{cpu}/cpu{n}/online", "1")
    for pol, cpus in (("policy0", "0 1 2 3"), ("policy4", "4 5")):
        put(tmp_path, f"{cpu}/cpufreq/{pol}/related_cpus", cpus)
        put(tmp_path, f"{cpu}/cpufreq/{pol}/scaling_available_frequencies", CPU_F + " \n")
        put(tmp_path, f"{cpu}/cpufreq/{pol}/scaling_min_freq", "115200")
        put(tmp_path, f"{cpu}/cpufreq/{pol}/scaling_max_freq", "1510400")
    g = dvfs.GPU_DEVFREQ
    put(tmp_path, g + "/available_frequencies", GPU_F)
    put(tmp_path, g + "/min_freq", "306000000")
    put(tmp_path, g + "/max_freq", "1020000000")
    put(tmp_path, dvfs.BPMP_DIR + "/emc/possible_rates", "204000 665600 2133000 3199000 (kHz)\n")
    put(tmp_path, dvfs.BPMP_DIR + "/clk/emc/rate", "3199000000")
    put(tmp_path, dvfs.BPMP_DIR + "/clk/emc/mrq_rate_locked", "0")
    return tmp_path


@pytest.fixture
def writes(monkeypatch, tree):
    log = []
    real = dvfs._write

    def rec(path, value):
        log.append((str(path)[len(str(tree)) + 1:], str(value)))
        real(path, value)
    monkeypatch.setattr(dvfs, "_write", rec)
    return log


def read(root, rel):
    return (root / rel).read_text()


def test_enumerate_states(tree):
    s = OrinSysfsBackend(str(tree)).enumerate_states()
    assert s == {"cpu_freqs_khz": [115200, 729600, 1510400],
                 "gpu_freqs_hz": [306000000, 510000000, 1020000000],
                 "emc_freqs_khz": [204000, 665600, 2133000, 3199000], "n_cpus": 6}


def test_gpu_glob_fallback(tmp_path):
    put(tmp_path, "sys/devices/platform/gpu@x/17000000.gpu/devfreq/17000000.gpu/available_frequencies", GPU_F)
    assert OrinSysfsBackend(str(tmp_path))._gpu_dir().endswith("devfreq/17000000.gpu")


def test_pin_order(tree, writes):
    b = OrinSysfsBackend(str(tree))
    b.set_gpu_freq(510000000)                  # below current max: min first
    assert [w[0].rsplit("/", 1)[1] for w in writes] == ["min_freq", "max_freq"]
    writes.clear()
    b.set_gpu_freq(306000000)
    b.set_gpu_freq(1020000000)                 # above current max (306 MHz): max first
    assert [w[0].rsplit("/", 1)[1] for w in writes[2:]] == ["max_freq", "min_freq"]
    assert read(tree, dvfs.GPU_DEVFREQ + "/min_freq") == "1020000000"


def test_apply_writes_and_offline_policy(tree, writes):
    b = OrinSysfsBackend(str(tree))
    full = Config(0, 729600, (1, 1, 1, 1, 0, 0), 510000000, 2133000)
    b.apply(None, full)
    cpu = "sys/devices/system/cpu"
    assert read(tree, f"{cpu}/cpu4/online") == "0" and read(tree, f"{cpu}/cpu1/online") == "1"
    assert read(tree, f"{cpu}/cpufreq/policy0/scaling_max_freq") == "729600"
    assert read(tree, f"{cpu}/cpufreq/policy4/scaling_max_freq") == "1510400"  # inactive: untouched
    assert read(tree, dvfs.BPMP_DIR + "/clk/emc/rate") == "2133000000"
    assert read(tree, dvfs.BPMP_DIR + "/clk/emc/mrq_rate_locked") == "1"
    assert not any("cpu0/online" in w[0] for w in writes)
    writes.clear()
    b.apply(full, Config(1, 729600, (1, 1, 1, 1, 0, 0), 1020000000, 2133000))
    assert {w[0].rsplit("/", 1)[1] for w in writes} == {"min_freq", "max_freq"}  # only the GPU


def test_restore(tree):
    b = OrinSysfsBackend(str(tree))
    b.apply(None, Config(0, 115200, (1, 0, 0, 0, 0, 0), 306000000, 204000))
    b.restore()
    cpu = "sys/devices/system/cpu"
    assert all(read(tree, f"{cpu}/cpu{n}/online") == "1" for n in range(1, 6))
    assert read(tree, f"{cpu}/cpufreq/policy0/scaling_min_freq") == "115200"
    assert read(tree, f"{cpu}/cpufreq/policy0/scaling_max_freq") == "1510400"
    assert read(tree, dvfs.GPU_DEVFREQ + "/max_freq") == "1020000000"
    assert read(tree, dvfs.GPU_DEVFREQ + "/min_freq") == "306000000"
    assert read(tree, dvfs.BPMP_DIR + "/clk/emc/mrq_rate_locked") == "0"


def test_build_config_space():
    cs = build_config_space([1, 2, 3], [10, 20], [5, 6], [1, 2], 4, cpu_stride=2)
    assert len(cs) == 2 * 2 * 2 * 2
    assert cs[0].cpu_mask == (1, 0, 0, 0) and cs[1].cpu_mask == (1, 1, 0, 0)
    assert (cs[0].mem_freq, cs[-1].mem_freq, cs[-1].cpu_freq_khz) == (5, 6, 3)


def test_ina3221_rails(tmp_path):
    d = "sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon3"
    for i, (label, mv, ma) in enumerate((("VDD_IN", 5000, 1000), ("VDD_SOC", 5000, 200)), 1):
        put(tmp_path, f"{d}/in{i}_label", label)
        put(tmp_path, f"{d}/in{i}_input", str(mv))
        put(tmp_path, f"{d}/curr{i}_input", str(ma))
    rails = find_rails(str(tmp_path))
    assert read_power_mw(rails) == {"VDD_IN": 5000.0, "VDD_SOC": 1000.0}
    t = iter([0.0, 0.5, 1.0])
    m = EnergyMeter(str(tmp_path), clock=lambda: next(t))
    m._sample()
    assert m.mark() == pytest.approx(2500.0)   # 5 W for 0.5 s
    assert m.mark() == pytest.approx(2500.0)
