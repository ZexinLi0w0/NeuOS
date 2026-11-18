import json

import pytest

from neuos_modern import Config, DryRunBackend, LocalCohort, NeuOSController, TierLadder
from neuos_modern.adapters.openai_stream import stream_completion
from neuos_modern.adapters.pipeline import Component, run_pipeline
from neuos_modern.profile import compute_uncertainty, main, profile, read_csv, write_csv

CONFIGS = [Config(i, 100, (1,), 100 * (i + 1), 100) for i in range(3)]


def test_uncertainty_hand_example():
    rows = [(0, 0, 100.0, 10.0), (1, 0, 50.0, 8.0),   # stage 0: max(1/1, 0.8/0.5) = 1.6
            (0, 1, 0.0, 5.0), (1, 1, 10.0, 0.0)]     # stage 1: zero guards -> 0
    assert compute_uncertainty(rows) == pytest.approx([1.6, 0])


def test_profile_runs_every_config_and_averages(tmp_path):
    t = [0.0]
    lat = {0: [1e-3, 3e-3], 1: [2e-3, 2e-3]}           # per stage, per repeat (s)

    def job_fn(ctl):
        j = ctl.begin_job(2)
        for i in range(2):
            assert j.before_stage(i, True) is False
            t[0] += lat[i][n[0] % 2]
            j.after_stage(i)
        n[0] += 1
        j.end()
    n = [0]
    backend = DryRunBackend()
    rows = profile(job_fn, CONFIGS, backend, repeats=2, clock=lambda: t[0])
    assert [r[:2] for r in rows] == [(c, s) for c in range(3) for s in range(2)]
    assert rows[0][2] == pytest.approx(2000.0) and rows[1][2] == pytest.approx(2000.0)
    assert sum(1 for w in backend.writes if w[0] == "gpu_freq") == 3
    write_csv(tmp_path / "p.csv", rows)
    main(["--from-csv", str(tmp_path / "p.csv"), "--uncertainty", str(tmp_path / "U.txt")])
    assert (tmp_path / "U.txt").read_text() == "0\t0\t"   # NullEnergyMeter: energy 0 -> Uncertainty 0
    assert read_csv(tmp_path / "p.csv")[0][:2] == (0, 0)


def _ctl():
    return NeuOSController(CONFIGS, [1.0], 1000, DryRunBackend(), LocalCohort(), seed=0)


def test_pipeline_selects_lower_tier_when_approximating():
    ladder = TierLadder("x", [{"name": "big"}, {"name": "small"}])
    comps = [Component(n, ladder, lambda tier, x: (x or []) + [tier["name"]]) for n in "abc"]
    out, log = run_pipeline(_ctl(), comps)
    assert out == log["tiers"] == ["small", "small", "big"]   # warmup: A* = 1 -> 2 approximations


class FakeResp:
    def __init__(self, n):
        ev = [{"choices": [{"text": "t%d " % k}]} for k in range(n)] + [{"choices": [{"text": ""}]}]
        self.lines = [b"data: " + json.dumps(e).encode() for e in ev] + [b"", b"data: [DONE]"]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def raise_for_status(self):
        pass

    def iter_lines(self):
        return iter(self.lines)


class FakeSession:
    def __init__(self, n):
        self.n, self.calls = n, []

    def post(self, url, json, stream, timeout):
        self.calls.append((url, json["model"]))
        return FakeResp(self.n)


def test_openai_stream_stages_and_tier():
    ladder = TierLadder("llm", [{"name": "7b", "url": "http://a/", "model": "m7"},
                                {"name": "3b", "url": "http://b", "model": "m3"}])
    ctl, s = _ctl(), FakeSession(20)
    text, log = stream_completion(ctl, ladder, "hi", max_tokens=20, chunk_tokens=8, session=s)
    assert s.calls == [("http://b/v1/completions", "m3")]      # warmup approximates stage 0
    assert text.split() == ["t%d" % k for k in range(20)]
    assert log["n_stages"] == 1 + 3 and log["tokens"] == 20     # 1 + 8 + 8 + 3
    assert [st["approximated"] for st in log["stages"]][1:] == [False] * 3


def test_torch_layers():
    torch = pytest.importorskip("torch")
    from neuos_modern.adapters.torch_layers import attach, ctl_job
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
    twin = torch.nn.Linear(4, 4)
    neu = attach(model, _ctl(), twins={"0": twin})
    with ctl_job(model):
        model(torch.zeros(1, 4))
    assert neu.last_log["n_stages"] == 3 and neu.last_log["stages"][0]["approximated"]
    neu.detach()
    assert "forward" not in model[0].__dict__


@pytest.mark.parametrize("stride,pad,k", [(1, 1, 3), (2, 3, 7), (2, 1, 3)])
def test_lowrank_conv_full_rank_is_exact(stride, pad, k):
    torch = pytest.importorskip("torch")
    from neuos_modern.adapters.torch_layers import lowrank_conv
    torch.manual_seed(0)
    conv = torch.nn.Conv2d(5, 7, k, stride=stride, padding=pad, bias=True).double()
    x = torch.randn(2, 5, 15, 13, dtype=torch.double)
    twin = lowrank_conv(conv, 10 ** 6)
    assert [c.kernel_size for c in twin] == [(k, 1), (1, k)]
    assert torch.allclose(twin(x), conv(x), atol=1e-9)
    low = lowrank_conv(conv, 2)
    assert low[0].out_channels == 2 and low(x).shape == conv(x).shape
    assert (low(x) - conv(x)).norm() > 1e-6


def test_lowrank_twins_selection_and_attach():
    torch = pytest.importorskip("torch")
    from neuos_modern.adapters.torch_layers import attach, ctl_job, lowrank_twins
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.ReLU(),
                                torch.nn.Conv2d(8, 8, 1), torch.nn.Conv2d(8, 4, 3, groups=2))
    twins = lowrank_twins(model, 0.5)
    assert sorted(twins) == ["0"] and twins["0"][0].out_channels == 5    # ceil(0.5 * 3*3)
    assert sorted(lowrank_twins(model, {"0": 2})) == ["0"] and lowrank_twins(model, {}) == {}
    neu = attach(model, _ctl(), twins=twins)
    with ctl_job(model):
        model(torch.zeros(1, 3, 8, 8))
    assert neu.last_log["stages"][0]["approximated"]


def test_torch_job_interrupted_mid_stage_reraises():
    torch = pytest.importorskip("torch")
    from neuos_modern.adapters.torch_layers import attach, ctl_job

    class Boom(torch.nn.Module):
        def forward(self, x):
            raise SystemExit(143)
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), Boom())
    neu = attach(model, _ctl())
    with pytest.raises(SystemExit):
        with ctl_job(model):
            model(torch.zeros(1, 4))
    assert neu.active is None
