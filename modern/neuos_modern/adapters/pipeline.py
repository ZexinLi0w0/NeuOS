"""Cascade of components (e.g. ASR -> LLM -> TTS): each component is one stage with
its own tier ladder; approximation runs it at the next lower tier."""
from dataclasses import dataclass
from typing import Any, Callable

from ..tiers import TierLadder


@dataclass
class Component:
    name: str
    ladder: TierLadder
    run: Callable[[dict, Any], Any]  # run(tier, x) -> y


def run_pipeline(ctl, components, x=None):
    """Run one job; returns ``(output, log)``."""
    job = ctl.begin_job(len(components))
    tiers = []
    for i, comp in enumerate(components):
        tier = comp.ladder.select(job.before_stage(i, comp.ladder.approximable))
        x = comp.run(tier, x)
        job.after_stage(i)
        tiers.append(tier.get("name"))
    log = job.end()
    log["tiers"] = tiers
    return x, log
