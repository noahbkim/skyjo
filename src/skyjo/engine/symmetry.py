"""Exact board-slot symmetries, independent of model and search behavior."""

from __future__ import annotations

import numpy as np

from skyjo.engine import game as sj


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


def symmetrize_policy_target(
    state: sj.Skyjo,
    policy_target: np.ndarray[tuple[int], np.float32],
) -> np.ndarray[tuple[int], np.float32]:
    """Average positional policy mass over board-symmetry action orbits.

    Rows may be permuted independently within a column, and columns may be
    permuted as units. Two active-player slots therefore share an orbit when
    their finger states match and their columns contain the same multiset of
    finger states. Flip and replace actions are averaged independently.
    """
    if policy_target.shape != (sj.MASK_SIZE,):
        raise ValueError(
            f"policy_target must have shape {(sj.MASK_SIZE,)}, "
            f"got {policy_target.shape}"
        )

    symmetrized = np.array(policy_target, dtype=np.float32, copy=True)
    orbits = slot_orbits(state)
    for action_offset in (sj.MASK_FLIP, sj.MASK_REPLACE):
        for slots in orbits:
            action_indices = np.asarray(slots, dtype=np.intp) + action_offset
            symmetrized[action_indices] = symmetrized[action_indices].mean()
    return symmetrized
