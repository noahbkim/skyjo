from __future__ import annotations

import numpy as np
import pytest

from skyjo import game as sj
from skyjo import play, skynet


def state_with_symmetric_active_board() -> sj.Skyjo:
    state = sj.new(players=2, top=sj.CARD_0)
    table = state[1].copy()
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
    return (*state[:1], table, *state[2:])


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

    actual = skynet.symmetrize_policy_target(state, original)

    assert np.array_equal(original, unchanged)
    assert np.allclose(actual, expected)
    assert np.array_equal(actual[: sj.MASK_FLIP], original[: sj.MASK_FLIP])
    assert actual.sum() == pytest.approx(original.sum())
    assert np.array_equal(
        skynet.symmetrize_policy_target(state, actual),
        actual,
    )


def test_symmetrize_policy_target_rejects_wrong_shape() -> None:
    state = state_with_symmetric_active_board()

    with pytest.raises(ValueError, match="policy_target must have shape"):
        skynet.symmetrize_policy_target(
            state,
            np.zeros(sj.MASK_SIZE - 1, dtype=np.float32),
        )


def test_game_history_conversion_symmetrizes_only_the_stored_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = state_with_symmetric_active_board()
    posterior = np.arange(1, sj.MASK_SIZE + 1, dtype=np.float32)
    posterior /= posterior.sum()
    unchanged = posterior.copy()
    history = [
        play.GameHistoryEntry(state, sj.MASK_FLIP_SECOND_BELOW, posterior)
        for _ in range(19)
    ]
    history.append(play.GameHistoryEntry(state, None, None))
    monkeypatch.setattr(
        play,
        "simulate_game_end",
        lambda *args, **kwargs: (
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.25, 0.75], dtype=np.float32),
            np.array([10.0, 20.0], dtype=np.float32),
            np.zeros(2 * sj.COLUMN_COUNT, dtype=np.float32),
        ),
    )

    game_data, _ = play.game_history_to_game_data(history)

    assert game_data[0].action == sj.MASK_FLIP_SECOND_BELOW
    assert np.array_equal(posterior, unchanged)
    assert np.array_equal(
        game_data[0].targets["policy"],
        skynet.symmetrize_policy_target(state, posterior),
    )
    assert not np.array_equal(game_data[0].targets["policy"], posterior)
