"""Model tier ladders: the port's generalisation of the original low-rank twin.

A ladder lists tiers best-first. ``approximate=True`` selects one tier below the
default (one step, like full conv -> low-rank conv in the original).
"""
import json
from dataclasses import dataclass
from typing import List


@dataclass
class TierLadder:
    name: str
    tiers: List[dict]
    default: int = 0

    def select(self, approximate):
        return self.tiers[min(self.default + 1, len(self.tiers) - 1) if approximate else self.default]

    @property
    def approximable(self):
        return self.default + 1 < len(self.tiers)


def load_ladders(path):
    """JSON ``{name: {"tiers": [...], "default": k}}`` -> ``{name: TierLadder}``."""
    with open(path) as f:
        raw = json.load(f)
    return {k: TierLadder(k, v["tiers"], v.get("default", 0)) for k, v in raw.items()}
