"""Exact board symmetries and the legal action partition used by search."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

import numpy as np

from . import game as sj


def _slot_signatures(state: sj.Skyjo) -> tuple[tuple[int, tuple[int, ...]], ...]:
    fingers = np.argmax(sj.get_table(state)[0], axis=-1)
    columns = [
        tuple(sorted(int(card) for card in fingers[:, column]))
        for column in range(sj.COLUMN_COUNT)
    ]
    return tuple(
        (int(fingers[row, column]), columns[column])
        for row in range(sj.ROW_COUNT)
        for column in range(sj.COLUMN_COUNT)
    )


def slot_orbits(state: sj.Skyjo) -> tuple[tuple[int, ...], ...]:
    """Partition all board slots, including slots with no legal action."""
    groups: dict[tuple[int, tuple[int, ...]], list[int]] = {}
    for slot, signature in enumerate(_slot_signatures(state)):
        groups.setdefault(signature, []).append(slot)
    return tuple(tuple(group) for group in groups.values())


@dataclasses.dataclass(frozen=True, slots=True)
class ActionGroups:
    """Legal actions grouped by exact symmetry, ordered by representative.

    A singleton partition follows the same path as pooled search. Tree nodes,
    visits, priors, and random noise remain owned by MCTS.
    """

    members: tuple[tuple[int, ...], ...]

    @property
    def representatives(self) -> tuple[int, ...]:
        return tuple(group[0] for group in self.members)

    @classmethod
    def from_state(cls, state: sj.Skyjo, *, merge: bool = True) -> ActionGroups:
        actions = tuple(int(action) for action in np.flatnonzero(sj.actions(state)))
        if not merge or all(action < sj.MASK_FLIP for action in actions):
            return cls(tuple((action,) for action in actions))

        signatures = _slot_signatures(state)
        groups: dict[tuple[int, int, tuple[int, ...]], list[int]] = {}
        for action in actions:
            family = sj.MASK_FLIP if action < sj.MASK_REPLACE else sj.MASK_REPLACE
            finger, column = signatures[action - family]
            key = (family, finger, column)
            groups.setdefault(key, []).append(action)
        return cls(tuple(tuple(group) for group in groups.values()))

    def aggregate(self, action_values: np.ndarray) -> np.ndarray:
        """Sum action mass into a MASK_SIZE vector indexed by representatives."""
        result = np.zeros(sj.MASK_SIZE, dtype=np.float64)
        for group in self.members:
            result[group[0]] = sum(float(action_values[action]) for action in group)
        return result

    def policy_from_visits(
        self, visits: Mapping[int, int], temperature: float = 1.0
    ) -> np.ndarray:
        """Apply temperature to group visits, then distribute mass to members."""
        if not np.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        counts = np.asarray(
            [visits.get(group[0], 0) for group in self.members], dtype=np.float64
        )
        visited = counts > 0
        if not visited.any():
            raise ValueError("policy requires at least one visited legal action")
        probabilities = np.zeros(sj.MASK_SIZE, dtype=np.float32)
        if temperature == 0:
            probabilities[self.members[int(counts.argmax())][0]] = 1
            return probabilities

        logs = np.log(counts[visited])
        with np.errstate(over="ignore"):
            weights = np.exp((logs - logs.max()) / temperature)
        for index, probability in zip(
            np.flatnonzero(visited), weights / weights.sum(), strict=True
        ):
            group = self.members[index]
            probabilities[list(group)] = probability / len(group)
        return probabilities


def safe_to_merge_actions(state: sj.Skyjo, budget: int) -> bool:
    """Conservatively exclude recycling anywhere below a search root.

    ``budget`` includes existing root visits and the requested new iterations.
    Unseen deck counts include hidden cards on every player's board. Revealing
    a hidden card reduces both counts; only an ordinary draw consumes slack.
    After the first draw, every additional draw requires a card-slot expansion
    and a draw expansion, so a path has at most ceil(budget / 2) draws. The extra
    card leaves enough unseen cards for all final reveals without recycling.

    This relies on the current MCTS expansion schedule: enumerated outcomes add
    breadth, and boundary samples restart from the same immutable state.
    """
    hidden = int(state.table[: state.players, :, :, sj.FINGER_HIDDEN].sum())
    slack = int(state.deck.sum()) - hidden
    return slack >= (budget + 1) // 2 + 1
