import dataclasses

import numpy as np
import pytest

from skyjo import game, observations
from skyjo.terminal_benchmark import (
    enumerate_one_hidden,
    guaranteed_terminal,
    random_stream,
    reconstruct,
    sample_outcomes,
    summarize_outcomes,
    terminal_credit,
)


def final_action_state(*, turn=3, scores=(80, 94), hidden=1):
    state = game.new(players=2, top=game.CARD_P12)
    state.game[game.GAME_ACTION : game.GAME_ACTION + game.ACTION_SIZE] = 0
    state.game[game.GAME_ACTION + game.ACTION_REPLACE] = 1
    state.game[game.GAME_SCORES : game.GAME_SCORES + 2] = scores
    state.game[game.GAME_LAST_REVEALED_TURNS : game.GAME_LAST_REVEALED_TURNS + 2] = turn
    state.table[:2] = 0
    state.table[:2, ..., game.FINGER_CLEARED] = 1
    hands = ((None, 0, 1), (1, 2, 3))
    for player, hand in enumerate(hands):
        for row, value in enumerate(hand):
            if player == 0 and row < hidden:
                value = None
            card = game.FINGER_HIDDEN if value is None else value + 2
            state.table[player, row, 0] = 0
            state.table[player, row, 0, card] = 1
            if card < game.CARD_SIZE:
                state.deck[card] -= 1
    state = dataclasses.replace(state, turn=turn, countdown=1)
    game.validate(state)
    return state


def test_reconstruction_retains_turn_rotation_and_progress_features():
    state = final_action_state(turn=7)
    restored = reconstruct(
        observations.get_spatial_state_numpy(state),
        observations.get_non_spatial_state_numpy(state),
    )
    assert game.get_player(restored) == 1
    assert game.hash_skyjo(restored) == game.hash_skyjo(state)
    assert not np.shares_memory(restored.table, state.table)


def test_exact_hidden_card_distribution_preserves_actor_and_shared_ties():
    state = final_action_state()
    # Replacing the actor's visible zero by12 leaves raw13+hidden. The opponent
    # finishes at100; actor wins below hidden7, ties at7, loses above7.
    values, terminal, weights = enumerate_one_hidden(state, game.MASK_REPLACE + 4)
    cards = np.flatnonzero(state.deck)
    expected_actor = np.where(cards - 2 < 7, 1, np.where(cards - 2 == 7, 0.5, 0))
    np.testing.assert_array_equal(values[:, 1], expected_actor)
    np.testing.assert_array_equal(values[:, 0], 1 - expected_actor)
    assert terminal.all()
    assert guaranteed_terminal(state)
    assert weights.sum() == pytest.approx(1)
    assert np.dot(weights, values[:, 1]) == pytest.approx(
        np.dot(state.deck[cards] / state.deck.sum(), expected_actor)
    )


def test_continuing_outcomes_keep_their_full_mass_in_win_bounds():
    # One player wins half the samples; the other half continue, with no winner
    # label assigned. Conditioning on termination would incorrectly report1.0.
    values = np.array([[1, 0], [0, 0]], dtype=float)
    terminal = np.array([True, False])
    result = summarize_outcomes(values, terminal, weights=np.array([0.5, 0.5]))
    assert result["terminal_probability"] == 0.5
    np.testing.assert_array_equal(result["eventual_win_bounds"], [[0.5, 1], [0, 0.5]])
    state = final_action_state(scores=(0, 0))
    outcome = game.apply_action(state, game.MASK_REPLACE, rng=random_stream(1, 2, 0, 0))
    credit, ended = terminal_credit(outcome)
    assert not ended
    np.testing.assert_array_equal(credit, [0, 0])


def test_exact_enumeration_rejects_action_dependent_recycling_support():
    state = final_action_state()
    state.game[game.GAME_DISCARDS : game.GAME_DISCARDS + game.CARD_SIZE] += state.deck
    state.deck[:] = 0
    game.validate(state)
    with pytest.raises(ValueError, match="nonempty initial deck"):
        enumerate_one_hidden(state, game.MASK_REPLACE + 4)


def test_estimates_are_reproducible_and_independent_of_reference_budget():
    def estimates(reference_budget):
        reference = random_stream(123, 2, 10, 20)
        for _ in range(reference_budget):
            reference.random()
        estimate = random_stream(123, 3, 10, 20)
        return [estimate.random() for _ in range(10)]

    assert estimates(1) == estimates(8192)
    assert estimates(1)[0] != random_stream(123, 2, 10, 20).random()
    assert estimates(1)[0] != random_stream(123, 1, 10, 20).random()


def test_multiple_reveals_use_engine_scoring_and_unconditioned_denominator():
    state = final_action_state(scores=(80, 80), hidden=2)
    values, terminal = sample_outcomes(
        state, game.MASK_REPLACE + 8, 128, random_stream(19, 2, 0, 24)
    )
    assert terminal.any() and not terminal.all()
    np.testing.assert_array_equal(values.sum(1), terminal.astype(float))
    result = summarize_outcomes(values, terminal)
    assert sum(result["terminal_win_contribution"]) == pytest.approx(terminal.mean())
    assert result["hoeffding_radius"] > 0
    with pytest.raises(ValueError, match="exactly one hidden"):
        enumerate_one_hidden(state, game.MASK_REPLACE + 8)
