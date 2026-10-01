import random
from itertools import cycle

import numpy as np
import pytest

import skyjo as sj
from skyjo import play, player


@pytest.fixture
def equal_hands(monkeypatch):
    """Deal identical 21-point hands without matching columns each round."""

    def draws():
        while True:
            # Discard, first cards, second cards, then ten draw/replace turns.
            cards = [12, 0, 0, 0, 1, 1, 1]
            hidden_cards = cycle(range(5, 13))
            for value in (0, 1, 2, 3, 4) * 2:
                for _ in range(3):
                    cards.extend((value, next(hidden_cards)))
            remaining = list(sj.CARD_COUNTS)
            for value in cards:
                card = value + 2
                assert remaining[card] > 0
                yield (sum(remaining[:card]) + 0.5) / sum(remaining)
                remaining[card] -= 1

    stream = draws()
    monkeypatch.setattr(random, "random", lambda: next(stream))


def completed_round(hands, previous_scores, ending_player=0):
    """Build a scored round with one remaining column per player.

    Hands and prior totals are supplied in fixed player order.
    """
    players = len(hands)
    game, table, deck, _, _, _, _ = sj.new(players=players, top=sj.CARD_P12)
    turn = players * 4 + ending_player
    game[sj.GAME_ACTION : sj.GAME_ACTION + sj.ACTION_SIZE] = 0
    game[sj.GAME_ACTION + sj.ACTION_DRAW_OR_TAKE] = 1
    game[sj.GAME_SCORES : sj.GAME_SCORES + players] = np.roll(
        previous_scores, -ending_player
    )
    game[sj.GAME_LAST_REVEALED_TURNS : sj.GAME_LAST_REVEALED_TURNS + players] = turn
    table[:players] = 0
    table[:players, :, :, sj.FINGER_CLEARED] = 1
    for relative_player in range(players):
        fixed_player = (relative_player + ending_player) % players
        for row, value in enumerate(hands[fixed_player]):
            card = value + 2
            table[relative_player, row, 0] = 0
            table[relative_player, row, 0, card] = 1
            deck[card] -= 1
    state = game, table, deck, players, turn, None, 0
    assert sj.validate(state)
    return state


@pytest.mark.parametrize("total,game_over", [(99, False), (100, True), (101, True)])
def test_game_threshold_is_checked_only_after_round_scoring(total, game_over):
    state = completed_round(((0, 1, 2), (1, 2, 3)), (10, total - 6))
    in_progress = (*state[:6], None)

    assert not sj.get_game_over(in_progress)
    np.testing.assert_array_equal(sj.get_game_scores(in_progress), (10, total - 6))
    assert sj.get_round_over(state)
    np.testing.assert_array_equal(sj.get_game_scores(state), (13, total))
    assert sj.get_game_over(state) is game_over


def test_game_scores_include_penalties_and_negative_points_without_mutation():
    state = completed_round(
        ((-2, 0, 1), (0, 1, 2), (1, 2, 3)), (10, 20, 30), ending_player=2
    )
    original_hash = sj.hash_skyjo(state)

    # Player 2 ends with six points but loses the round, so receives twelve.
    np.testing.assert_array_equal(
        sj.get_fixed_perspective_round_scores(state), (-1, 3, 12)
    )
    np.testing.assert_array_equal(sj.get_game_scores(state), (42, 9, 23))
    for _ in range(2):
        np.testing.assert_array_equal(
            sj.get_fixed_perspective_game_scores(state), (9, 23, 42)
        )
    scores = sj.get_game_scores(state)
    scores[:] = 999
    assert sj.hash_skyjo(state) == original_hash


def test_next_round_preserves_identity_and_starts_with_the_ending_player():
    previous = completed_round(
        ((-2, 0, 1), (0, 1, 2), (1, 2, 3)), (10, 20, 30), ending_player=2
    )
    original_hash = sj.hash_skyjo(previous)
    rng = random.Random(0)

    state = sj.start_next_round(previous, rng=rng)

    assert sj.validate(state)
    assert sj.get_player(state) == 2
    assert sj.get_countdown(state) is None
    assert sj.get_action(state) == sj.ACTION_FLIP_SECOND
    assert sj.get_discard_count(state) == 0
    assert [sj.get_facedown_count(state, i) for i in range(3)] == [11, 11, 11]
    np.testing.assert_array_equal(sj.get_last_revealed_turns(state), (2, 2, 2))
    for old, new in zip(previous[:3], state[:3]):
        assert not np.shares_memory(old, new)

    for expected_player in (2, 0, 1):
        assert sj.get_player(state) == expected_player
        np.testing.assert_array_equal(
            sj.get_fixed_perspective_game_scores(state), (9, 23, 42)
        )
        state = sj.apply_action(state, sj.MASK_FLIP_SECOND_BELOW, rng=rng)

    assert sj.get_player(state) == 2
    assert sj.get_action(state) == sj.ACTION_DRAW_OR_TAKE
    assert sj.validate(state)
    np.testing.assert_array_equal(
        sj.get_fixed_perspective_game_scores(state), (9, 23, 42)
    )
    assert sj.hash_skyjo(previous) == original_hash


def test_round_boundary_rejects_actions_and_invalid_transitions():
    state = completed_round(((0, 1, 2), (1, 2, 3)), (0, 0))
    assert not sj.actions(state).any()
    assert list(sj.get_actions(state)) == []
    with pytest.raises(ValueError):
        sj.apply_action(state, sj.MASK_DRAW)
    with pytest.raises(ValueError):
        sj.start_next_round(sj.new(players=2))
    finished_game = completed_round(((0, 1, 2), (1, 2, 3)), (0, 94))
    with pytest.raises(ValueError):
        sj.start_next_round(finished_game)


def test_full_game_returns_round_histories_and_shared_winners(equal_hands):
    result = play.play_game([player.NaiveQuickFinishPlayer() for _ in range(3)])

    assert len(result.rounds) == 3
    assert result.final_scores == (126, 63, 63)
    assert result.winners == (1, 2)
    for number, round_result in enumerate(result.rounds, start=1):
        assert round_result.round_scores == (42, 21, 21)
        assert round_result.cumulative_scores == (42 * number, 21 * number, 21 * number)
        assert round_result.ending_player == 0
        assert round_result.history[-1].action is None
        assert all(entry.action is not None for entry in round_result.history[:-1])
        assert sj.get_round_over(round_result.history[-1].state)
        assert sj.get_game_over(round_result.history[-1].state) is (number == 3)
        np.testing.assert_array_equal(
            sj.get_fixed_perspective_game_scores(round_result.history[0].state),
            (42 * (number - 1), 21 * (number - 1), 21 * (number - 1)),
        )

    # A completed round remains usable by the existing training/statistics path.
    data, stats = play.game_history_to_game_data(result.rounds[0].history)
    assert len(data) == len(result.rounds[0].history) - 1
    np.testing.assert_array_equal(stats.scores_state_value, (42, 21, 21))


@pytest.mark.parametrize("runner", [play.play_round, play.play])
def test_round_runners_stop_before_the_game_ends(runner, equal_hands):
    history = runner([player.NaiveQuickFinishPlayer() for _ in range(3)])

    assert sj.get_round_over(history[-1].state)
    assert not sj.get_game_over(history[-1].state)
    assert history[-1].action is None
    assert history[-1].action_probabilities is None
