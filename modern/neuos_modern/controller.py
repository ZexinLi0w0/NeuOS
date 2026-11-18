"""NeuOS runtime policy: port of Net::setup_DVFS / warmup / ForwardFromTo (net.cpp).

One *job* is one inference (one ForwardFromTo call); one *stage* is one Caffe layer
in the original, any unit of work here.
"""
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Optional

from .config import sort_configs
from .energy import NullEnergyMeter
from .schedule import f32, schedule

DEADLINE = 30  # #define DEADLINE 30 in net.cpp (divisor of the cohort priority)
MODES = ("DEFAULT", "MAX_ACC", "MIN_ENG")
VARIANTS = ("original", "corrected")


def _us(seconds):
    return int(round(seconds * 1e6))  # integer microseconds, like energymon_gettime_us


def cohort_priority(data, mode, previous):
    """Mean of (1 + x/DEADLINE) over the slots with data != 0, where x is the slot's
    own epsilon (DEFAULT), or the max / min over *all* slots, zeros included
    (MAX_ACC / MIN_ENG), as in net.cpp. With no non-zero slot the original divides
    0/0 (NaN); here ``previous`` is kept instead."""
    if mode == "MAX_ACC":
        xs = [max(data)] * sum(1 for d in data if d != 0)
    elif mode == "MIN_ENG":
        xs = [min(data)] * sum(1 for d in data if d != 0)
    else:
        xs = [d for d in data if d != 0]
    if not xs:
        return previous
    total = 0.0
    for x in xs:
        total = f32(total + f32(1 + x / DEADLINE))
    return f32(total / len(xs))


class NeuOSController:
    def __init__(self, configs, uncertainty, period_ms, backend, cohort, mode="DEFAULT",
                 variant="original", config_on=True, seed=None, energy=None,
                 clock=time.perf_counter):
        if mode not in MODES or variant not in VARIANTS:
            raise ValueError("mode must be in %s and variant in %s" % (MODES, VARIANTS))
        self.configs = sort_configs(configs)
        self.uncertainty = [f32(float(u)) for u in uncertainty]
        self.period_ms = period_ms
        self.backend, self.cohort = backend, cohort
        self.mode, self.variant, self.config_on = mode, variant, config_on
        self.rng = random.Random(seed)
        self.energy = energy or NullEnergyMeter()
        self.clock = clock
        self.is_warmup = True
        self.deadlines = []
        self.slack = 0.0
        self.current = {}       # current_configuration[i]; unset entries -> _init_cfg
        self._init_cfg = 0
        self.history = defaultdict(dict)      # history[i]: {config: HistEnt}
        self.rev_history = defaultdict(dict)  # rev_history[i]: {to_string(speedup): config}

    # Stages beyond the warmup job's length reuse the last deadline / uncertainty.
    def deadline(self, i):
        return self.deadlines[min(i, len(self.deadlines) - 1)]

    def U(self, i):
        return self.uncertainty[min(i, len(self.uncertainty) - 1)]

    def cfg(self, i):
        return self.current.get(i, self._init_cfg)

    def initial_slack(self):
        return f32(self.period_ms * 1000 - sum(self.deadlines))

    def _warmup(self):
        if not self.config_on:
            return
        conf = self.rng.randrange(len(self.configs))
        self.backend.apply(None, self.configs[conf])
        self.current.clear()
        self._init_cfg = conf

    def begin_job(self, n_stages):
        if self.is_warmup:
            self._warmup()
        elif self.variant == "corrected":
            self.slack = self.initial_slack()
        t0 = self.clock()
        slot = self.cohort.acquire_slot()
        return Job(self, n_stages, slot, _us(self.clock() - t0))


class Job:
    def __init__(self, ctl, n_stages, slot, sched_us):
        self.ctl, self.n_stages, self.slot = ctl, n_stages, slot
        self.sched_us = sched_us
        self.warmup = ctl.is_warmup
        self.epsilon, self.priority = 0.0, 1.0
        self.max_score = float(n_stages - 1)               # end - start
        self.min_score = self.max_score - self.max_score * 0.5
        self.last_exec_us = 0
        self.stages = []
        self._t0 = None

    def before_stage(self, i, approximable=False):
        """Decision point before stage i. Returns True if the approximate variant
        (next lower tier) must be run."""
        c = self.ctl
        if c.mode == "MAX_ACC":
            a_star = self.max_score
        else:
            a_star = math.ceil(self.priority * self.min_score + (1 - self.priority) * self.max_score)
        c.cohort.publish(self.slot, self.epsilon, self.max_score)

        if not self.warmup and i > 0:
            t = c.clock()
            if c.config_on:
                new, c.slack = schedule(
                    self.priority, c.slack, c.deadline(i), c.deadline(i - 1),
                    f32(float(self.last_exec_us) + self.sched_us), c.U(i), c.cfg(i - 1), c.cfg(i),
                    c.history[i - 1], c.rev_history[i - 1], c.history[i], c.rev_history[i],
                    corrected=(c.variant == "corrected"))
            self.sched_us += _us(c.clock() - t)
            if c.config_on:
                # original clamps only the top; the bottom clamp avoids Python's negative indexing
                c.current[i] = max(0, min(new, len(c.configs) - 1))
                c.backend.apply(c.configs[c.cfg(i - 1)], c.configs[c.cfg(i)])

        approx = a_star < self.max_score and approximable
        self.stages.append({"stage": i, "config": c.cfg(i), "approximated": approx,
                            "a_star": a_star, "priority": self.priority})
        if approx:
            self.max_score -= 0.5
        c.energy.mark()
        self._t0 = c.clock()
        return approx

    def after_stage(self, i):
        c = self.ctl
        exec_us = _us(c.clock() - self._t0)
        energy_mj = c.energy.mark()
        self.last_exec_us = exec_us
        if self.warmup:
            c.deadlines.append(exec_us)
        else:
            self.epsilon = c.slack
            t = c.clock()
            self.priority = cohort_priority(c.cohort.read(), c.mode, self.priority)
            self.sched_us += _us(c.clock() - t)
        self.stages[-1].update(latency_us=exec_us, energy_mj=energy_mj, slack=c.slack)

    def end(self):
        c = self.ctl
        if self.warmup:
            c.slack = c.initial_slack()
        c.is_warmup = False
        return {"warmup": self.warmup, "slot": self.slot, "n_stages": len(self.stages),
                "sched_us": self.sched_us, "slack": c.slack,
                "latency_us": sum(s["latency_us"] for s in self.stages),
                "energy_mj": sum(s["energy_mj"] for s in self.stages),
                "stages": self.stages}


@dataclass
class Stage:
    name: str
    full: Callable
    approx: Optional[Callable] = None


def run_job(ctl, stages, x=None):
    """Run one job; each stage callable maps the previous stage's output to its own.
    Returns ``(output, log)``."""
    job = ctl.begin_job(len(stages))
    for i, st in enumerate(stages):
        use_approx = job.before_stage(i, st.approx is not None)
        x = (st.approx if use_approx else st.full)(x)
        job.after_stage(i)
    return x, job.end()
