from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from helpers import NaiveQuickFinishPlayer

from skyjo.engine import game as sj
from skyjo.engine import symmetry
from skyjo.learning import targets
from skyjo.simulation import play


def state_with_symmetric_active_board() -> sj.Skyjo:
    state = sj.new(players=2, top=sj.CARD_0)
    table = state.table.copy()
    table[0] = 0
    finger_states = np.array(
        [
            [sj.FINGER_HIDDEN, sj.CARD_P1, sj.FINGER_HIDDEN, sj.FINGER_HIDDEN],
            [sj.FINGER_HIDDEN, sj.FINGER_HIDDEN, sj.FINGER_HIDDEN, sj.CARD_P3],
            [sj.CARD_P1, sj.FINGER_HIDDEN, sj.CARD_P2, sj.CARD_P4],
        ]
    )
    for row in range(sj.ROW_COUNT):
        for column in range(sj.COLUMN_COUNT):
            table[0, row, column, finger_states[row, column]] = 1
    return dataclasses.replace(state, table=table)


def test_symmetrize_policy_target_averages_exact_action_orbits() -> None:
    state = state_with_symmetric_active_board()
    original = np.arange(1, sj.MASK_SIZE + 1, dtype=np.float32)
    original /= original.sum()
    unchanged = original.copy()
    expected = original.copy()
    slot_orbits = ([0, 4, 5, 9], [1, 8], [2, 6])
    for action_offset in (sj.MASK_FLIP, sj.MASK_REPLACE):
        for slots in slot_orbits:
            actions = np.asarray(slots) + action_offset
            expected[actions] = expected[actions].mean()

    actual = symmetry.symmetrize_policy_target(state, original)

    assert np.array_equal(original, unchanged)
    assert np.allclose(actual, expected)
    assert np.array_equal(actual[: sj.MASK_FLIP], original[: sj.MASK_FLIP])
    assert actual.sum() == pytest.approx(original.sum())
    assert np.array_equal(
        symmetry.symmetrize_policy_target(state, actual),
        actual,
    )


def test_symmetrize_policy_target_rejects_wrong_shape() -> None:
    state = state_with_symmetric_active_board()

    with pytest.raises(ValueError, match="policy_target must have shape"):
        symmetry.symmetrize_policy_target(
            state,
            np.zeros(sj.MASK_SIZE - 1, dtype=np.float32),
        )


def test_full_game_conversion_symmetrizes_only_the_stored_target() -> None:
    state = state_with_symmetric_active_board()
    posterior = np.arange(1, sj.MASK_SIZE + 1, dtype=np.float32)
    posterior /= posterior.sum()
    unchanged = posterior.copy()
    completed = play.play_game([NaiveQuickFinishPlayer() for _ in range(2)])
    last_round = completed.rounds[-1]
    # Isolate one recorded decision and a genuine completed-game snapshot.
    history = [
        play.RoundHistoryEntry(state, sj.MASK_FLIP_SECOND_BELOW, posterior),
        last_round.history[-1],
    ]
    result = play.GameResult(
        (
            play.RoundResult(
                history,
                last_round.round_scores,
                last_round.cumulative_scores,
                last_round.ending_player,
            ),
        )
    )
    game_data = targets.build_training_batch(result)

    assert np.array_equal(posterior, unchanged)
    assert np.array_equal(
        game_data.targets["policy"][0],
        symmetry.symmetrize_policy_target(state, posterior),
    )
    assert not np.array_equal(game_data.targets["policy"][0], posterior)
