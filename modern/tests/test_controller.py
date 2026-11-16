import random

import pytest

from neuos_modern import Config, DryRunBackend, LocalCohort, NeuOSController, Stage, run_job
from neuos_modern.controller import cohort_priority

# Sorted already; neighbours differ only in GPU frequency.
CONFIGS = [Config(i, 100, (1, 0), 100 * (i + 1), 100) for i in range(4)]
SEED0 = next(s for s in range(1000) if random.Random(s).randrange(4) == 0)  # warmup picks config 0


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def run(self, ms):
        self.t += ms / 1000.0


def make(uncertainty=(2.0, 2.0, 2.0), period_ms=100, seed=SEED0, **kw):
    clock, backend, cohort = FakeClock(), DryRunBackend(), LocalCohort()
    ctl = NeuOSController(CONFIGS, list(uncertainty), period_ms, backend, cohort,
                          seed=seed, clock=clock, **kw)
    return ctl, clock, backend, cohort


def job(ctl, clock, lat_ms, approximable=False):
    j = ctl.begin_job(len(lat_ms))
    for i, ms in enumerate(lat_ms):
        j.before_stage(i, approximable)
        clock.run(ms)
        j.after_stage(i)
    return j.end()


def test_warmup_records_deadlines_and_slack():
    ctl, clock, backend, _ = make(seed=1)
    conf = random.Random(1).randrange(4)
    log = job(ctl, clock, [10, 20, 30])
    assert backend.writes == [("cpu_mask", (1, 0)), ("cpu_freq", 100),
                              ("gpu_freq", CONFIGS[conf].gpu_freq_hz), ("emc_freq", 100)]
    assert ctl.deadlines == [10000, 20000, 30000]
    assert ctl.slack == 100 * 1000 - 60000
    assert log["warmup"] and not ctl.is_warmup
    assert all(s["config"] == conf for s in log["stages"])


def test_second_job_schedules_from_stage1_and_actuates_only_changed_knobs():
    ctl, clock, backend, _ = make()
    job(ctl, clock, [10, 20, 30])
    backend.writes.clear()
    j = ctl.begin_job(3)
    j.before_stage(0)
    assert backend.writes == [] and ctl.current == {}          # no decision at stage 0
    clock.run(10)
    j.after_stage(0)
    j.before_stage(1)
    # schedule: slack += 10000 - 10000; needed 0.333 miss -> 0 + 100/2 = 50 -> clamped to 3
    assert ctl.current[1] == 3
    assert backend.writes == [("gpu_freq", 400)]
    assert ctl.history[0][0].speedup == 1.0 and ctl.rev_history[0] == {"1.000000": 0}


def test_clamp_at_zero():
    ctl, clock, backend, _ = make(uncertainty=(4.0, 4.0, 4.0))
    job(ctl, clock, [10, 20, 30])
    j = ctl.begin_job(3)
    j.before_stage(0)
    clock.run(10)
    j.after_stage(0)
    j.before_stage(1)                    # 1*0 - 10/4 = -2.5 -> -2 -> clamped to 0
    assert ctl.current[1] == 0


def test_max_acc_never_approximates():
    ctl, clock, _, _ = make(mode="MAX_ACC")
    for _ in range(3):
        log = job(ctl, clock, [10, 20, 30], approximable=True)
        assert not any(s["approximated"] for s in log["stages"])


def test_default_approximation_follows_a_star():
    ctl, clock, _, cohort = make()
    log = job(ctl, clock, [10, 20, 30], approximable=True)
    # max 2, min 1, priority 1 -> A* = 1 each stage; max_score 2 -> 1.5 -> 1.0
    assert [s["a_star"] for s in log["stages"]] == [1, 1, 1]
    assert [s["approximated"] for s in log["stages"]] == [True, True, False]
    assert cohort.approx[log["slot"]] == 1.0


def test_cohort_priority_modes():
    data = [0, 30, 60, 0, 0, 0, 0, 0, 0, -30]
    assert cohort_priority(data, "DEFAULT", 9) == pytest.approx((2 + 3 + 0) / 3)
    assert cohort_priority(data, "MAX_ACC", 9) == pytest.approx(3.0)
    assert cohort_priority(data, "MIN_ENG", 9) == pytest.approx(0.0)
    assert cohort_priority([30] + [0] * 9, "MIN_ENG", 9) == pytest.approx(1.0)  # zeros count for min
    assert cohort_priority([0] * 10, "DEFAULT", 9) == 9                           # empty cohort guard


def test_priority_uses_cohort_after_stage():
    ctl, clock, _, cohort = make()
    job(ctl, clock, [10, 20, 30])
    cohort.data[5] = 60.0                # another job's epsilon
    log = job(ctl, clock, [10, 20, 30])
    assert log["stages"][1]["priority"] == pytest.approx(3.0)


def test_slack_persists_in_original_and_resets_in_corrected():
    for variant, expect_reset in (("original", False), ("corrected", True)):
        ctl, clock, _, _ = make(variant=variant)
        job(ctl, clock, [10, 20, 30])
        initial = ctl.slack
        job(ctl, clock, [15, 25, 35])   # overruns change the slack
        assert ctl.slack != initial
        ctl.begin_job(3)
        assert (ctl.slack == initial) == expect_reset


def test_overrun_sign_original_vs_corrected():
    for variant, sign in (("original", 1), ("corrected", -1)):
        ctl, clock, _, _ = make(variant=variant)
        job(ctl, clock, [10, 20, 30])
        initial = ctl.slack
        job(ctl, clock, [15, 20, 30])   # stage 0 overruns by 5 ms; checked at stage 1
        assert ctl.slack == initial + sign * 5000


def test_extra_stages_reuse_last_deadline():
    ctl, clock, _, _ = make()
    job(ctl, clock, [10, 20, 30])
    log = job(ctl, clock, [10, 20, 30, 30, 30])
    assert log["n_stages"] == 5 and ctl.deadline(4) == 30000 and ctl.U(4) == 2.0


def test_config_off_never_actuates():
    ctl, clock, backend, _ = make(config_on=False)
    job(ctl, clock, [10, 20, 30])
    job(ctl, clock, [10, 20, 30])
    assert backend.writes == [] and ctl.current == {}


def test_run_job_uses_approx_callable():
    ctl, clock, _, _ = make()
    stages = [Stage("a", lambda x: x + 1, lambda x: x + 100), Stage("b", lambda x: x * 2),
              Stage("c", lambda x: x - 1)]
    out, log = run_job(ctl, stages, 0)
    assert out == (0 + 100) * 2 - 1      # warmup: stage a approximated (A* = 1 < 2)
    assert [s["approximated"] for s in log["stages"]] == [True, False, False]
