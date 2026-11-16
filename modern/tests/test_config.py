from pathlib import Path

from neuos_modern.config import (Config, format_uncertainty, load_configs, load_uncertainty,
                                 parse_config_line, save_configs, sort_configs)

AGX = Path(__file__).resolve().parents[2] / "examples" / "triangle" / "all-AGX.config"


def test_parse_original_agx_lines():
    with open(AGX) as f:
        lines = [next(f) for _ in range(3)]
    c = [parse_config_line(x) for x in lines]
    assert c[0] == Config(0, 115200, (1, 0, 0, 0, 0, 0, 0, 0), 114750000, 2133)
    assert c[2].id == 2 and c[2].gpu_freq_hz == 318750000 and c[2].cores == 1


def test_sort_order_mem_cpu_gpu_cores():
    cs = [Config(0, 2, (1, 1), 1, 1), Config(1, 1, (1, 1), 1, 2), Config(2, 1, (1, 1), 2, 1),
          Config(3, 1, (1, 0), 2, 1), Config(4, 1, (1, 0), 1, 1)]
    assert [c.id for c in sort_configs(cs)] == [4, 3, 2, 0, 1]


def test_roundtrip(tmp_path):
    cs = [Config(0, 115200, (1, 0, 0, 0, 0, 0), 306000000, 204000),
          Config(1, 1510400, (1, 1, 1, 1, 1, 1), 1020000000, 3199000)]
    save_configs(tmp_path / "c.config", cs)
    assert load_configs(tmp_path / "c.config") == cs


def test_uncertainty_file_trailing_tab(tmp_path):
    p = tmp_path / "U.txt"
    p.write_text(format_uncertainty([1.5, 2.0, 0]))
    assert p.read_text() == "1.5\t2.0\t0\t"
    assert load_uncertainty(p) == [1.5, 2.0, 0.0]
