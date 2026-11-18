"""Port of include/triangle/schedule.h and include/DVFS/misc.h.

C++ ``float`` arithmetic is emulated with :func:`f32` on the values that become
history keys (``std::to_string``) or that persist (``slack``), and float->int
assignment truncates toward zero with aarch64 saturation (the original ran on
Jetson boards): NaN -> 0, +-inf -> INT_MAX/INT_MIN.
"""
import math
import struct
from dataclasses import dataclass

INT_MAX, INT_MIN = 2**31 - 1, -(2**31)


def f32(x):
    """Round a Python float to the nearest IEEE-754 single (C++ ``float``)."""
    try:
        return struct.unpack("f", struct.pack("f", x))[0]
    except OverflowError:
        return math.copysign(math.inf, x)


def to_int(x):
    """C++ float->int assignment on aarch64 (FCVTZS): truncate, saturate, NaN->0."""
    if math.isnan(x):
        return 0
    if math.isinf(x):
        return INT_MAX if x > 0 else INT_MIN
    return max(INT_MIN, min(INT_MAX, int(x)))


def to_string(x):
    """``std::to_string(float)`` == printf("%f")."""
    return "%f" % x


def fdiv(a, b):
    """IEEE division (C++ float semantics) instead of ZeroDivisionError."""
    if b == 0:
        if a == 0 or math.isnan(a):
            return math.nan
        return math.copysign(math.inf, a) * math.copysign(1.0, b)
    return a / b


@dataclass
class HistEnt:
    """struct hist_ent (powerup is never written by the original code)."""
    powerup: float = 0.0
    speedup: float = 0.0


def schedule_basic(priority, slack, uncertainty, current):
    """schedule(priority, slack, Uncertainty, current_configuration)."""
    if slack < 0:
        return to_int(current + fdiv(100.0, uncertainty))
    if uncertainty > 3:
        return to_int(priority * current - fdiv(10.0, uncertainty))
    return to_int(priority * current + fdiv(100.0, uncertainty))


def schedule(priority, slack, expected_deadline, prev_expected_deadline, measured_exec,
             uncertainty, prev_current, current, prev_history, rev_prev_history,
             history, rev_history, corrected=False):
    """History-based schedule() overload. Returns ``(configuration, new_slack)``.

    The C++ version takes ``slack`` by reference; here the updated value is returned.
    ``corrected=False`` keeps the original sign (``slack += measured - expected``);
    ``corrected=True`` lets an overrun consume slack (``slack -= measured - expected``).
    """
    delta = f32(measured_exec - prev_expected_deadline)
    slack = f32(slack - delta if corrected else slack + delta)

    # Quirk kept: membership is tested with `current` but the entry is stored at `prev_current`.
    if current not in prev_history:
        ent = HistEnt(speedup=f32(fdiv(prev_expected_deadline, measured_exec)))
        prev_history[prev_current] = ent
        rev_prev_history[to_string(ent.speedup)] = prev_current

    needed = f32(fdiv(expected_deadline, f32(expected_deadline + slack)))
    if needed < 0:
        needed = 7.0
    key = to_string(needed)
    if key in rev_history:
        return rev_history[key], slack
    return schedule_basic(priority, slack, uncertainty, current), slack
