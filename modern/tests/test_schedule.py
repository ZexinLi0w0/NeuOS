from neuos_modern.schedule import HistEnt, schedule, schedule_basic, to_int, to_string


def test_basic_negative_slack_adds_100_over_u():
    assert schedule_basic(1.0, -1.0, 3.0, 5) == 38          # 5 + 33.33 -> 38


def test_basic_high_uncertainty_subtracts():
    assert schedule_basic(1.0, 0.0, 4.0, 5) == 2            # 5 - 2.5 -> 2


def test_basic_truncates_toward_zero_for_negatives():
    assert schedule_basic(1.0, 0.0, 4.0, 1) == -1           # -1.5 -> -1, not -2


def test_basic_low_uncertainty_scales_by_priority():
    assert schedule_basic(0.5, 10.0, 2.0, 10) == 55         # 0.5*10 + 50
    assert schedule_basic(1.0, 0.0, 3.0, 0) == 33           # U == 3 is the "+" branch


def test_to_int_saturates_like_aarch64():
    assert to_int(float("nan")) == 0
    assert to_int(float("inf")) == 2**31 - 1
    assert to_string(2.0) == "2.000000"


def _call(slack, expected, prev_expected, measured, U, prev_cur, cur, ph, rph, h, rh, corrected=False):
    return schedule(1.0, slack, expected, prev_expected, measured, U, prev_cur, cur, ph, rph, h, rh,
                    corrected=corrected)


def test_history_miss_records_prev_and_falls_back_to_basic():
    ph, rph, h, rh = {}, {}, {}, {}
    cfg, slack = _call(100.0, 50.0, 40.0, 20.0, 4.0, 3, 7, ph, rph, h, rh)
    assert slack == 80.0                                     # 100 + (20 - 40)
    assert ph[3].speedup == 2.0 and rph == {"2.000000": 3}   # 40 / 20
    assert cfg == 4                                          # needed 0.384615 missing -> 7 - 2.5


def test_history_hit_returns_stored_config():
    rh = {"0.384615": 9}                                     # 50 / (50 + 80)
    cfg, _ = _call(100.0, 50.0, 40.0, 20.0, 4.0, 3, 7, {}, {}, {}, rh)
    assert cfg == 9


def test_needed_negative_becomes_7():
    rh = {"7.000000": 11}
    cfg, slack = _call(-200.0, 50.0, 40.0, 40.0, 4.0, 3, 7, {}, {}, {}, rh)
    assert slack == -200.0 and cfg == 11                     # 50 / -150 < 0 -> 7
    cfg, _ = _call(-200.0, 50.0, 40.0, 40.0, 4.0, 3, 7, {}, {}, {}, {})
    assert cfg == 32                                         # basic, slack < 0: 7 + 25


def test_quirk_membership_checks_cur_but_inserts_prev_cur():
    ph = {7: HistEnt(speedup=5.0)}                           # cur present -> nothing inserted
    rph = {}
    _call(100.0, 50.0, 40.0, 20.0, 4.0, 3, 7, ph, rph, {}, {})
    assert 3 not in ph and rph == {}
    ph = {3: HistEnt(speedup=5.0)}                           # prev_cur present but cur absent -> overwritten
    _call(100.0, 50.0, 40.0, 20.0, 4.0, 3, 7, ph, rph, {}, {})
    assert ph[3].speedup == 2.0 and rph == {"2.000000": 3}


def test_corrected_variant_overrun_consumes_slack():
    _, slack = _call(100.0, 50.0, 40.0, 60.0, 4.0, 3, 7, {}, {}, {}, {})
    assert slack == 120.0                                    # original: overrun adds slack
    _, slack = _call(100.0, 50.0, 40.0, 60.0, 4.0, 3, 7, {}, {}, {}, {}, corrected=True)
    assert slack == 80.0
