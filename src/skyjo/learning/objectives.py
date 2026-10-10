"""The supported auxiliary objectives and their immutable training weights."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping

OBJECTIVE_NAMES = ("round_score", "round_raw_score", "round_doubled")


@dataclasses.dataclass(frozen=True, slots=True)
class ResolvedObjectives:
    entries: tuple[tuple[str, float], ...] = ()

    @property
    def weights(self) -> dict[str, float]:
        return dict(self.entries)

    def shapes(self, players: int) -> dict[str, tuple[int, ...]]:
        return {name: (players,) for name, _ in self.entries}


ObjectiveConfig = Mapping[str, float] | ResolvedObjectives | None


def resolve(config: ObjectiveConfig = None) -> ResolvedObjectives:
    if isinstance(config, ResolvedObjectives):
        return config
    entries = []
    for name, weight in sorted((config or {}).items()):
        if name not in OBJECTIVE_NAMES:
            raise ValueError(f"Unknown auxiliary objective: {name}")
        if type(weight) not in (float, int) or not math.isfinite(weight) or weight < 0:
            raise ValueError(f"Auxiliary weight must be finite and nonnegative: {name}")
        if weight > 0:
            entries.append((name, float(weight)))
    return ResolvedObjectives(tuple(entries))
