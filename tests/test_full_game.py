import dataclasses
import random
from itertools import cycle

import numpy as np
import pytest
from helpers import NaiveQuickFinishPlayer

from skyjo import game as sj
from skyjo import observations, play, player


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
    initial = sj.new(players=players, top=sj.CARD_P12)
    game = initial.game
    table = initial.table
    deck = initial.deck
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
    state = sj.Skyjo(
        game=game, table=table, deck=deck, players=players, turn=turn, countdown=0
    )
    assert sj.validate(state)
    return state


@pytest.mark.parametrize("total,game_over", [(99, False), (100, True), (101, True)])
def test_game_threshold_is_checked_only_after_round_scoring(total, game_over):
    state = completed_round(((0, 1, 2), (1, 2, 3)), (10, total - 6))
    in_progress = dataclasses.replace(state, countdown=None)

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
    for old, new in zip(
        (previous.game, previous.table, previous.deck),
        (state.game, state.table, state.deck),
        strict=True,
    ):
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
    result = play.play_game([NaiveQuickFinishPlayer() for _ in range(3)])

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

    data, stats = play.game_result_to_game_data(result)
    assert len(data) == sum(len(r.history) - 1 for r in result.rounds)
    np.testing.assert_array_equal(stats.scores_state_value, (126, 63, 63))


def test_play_round_stops_before_the_game_ends(equal_hands):
    history = play.play_round([NaiveQuickFinishPlayer() for _ in range(3)])

    assert sj.get_round_over(history[-1].state)
    assert not sj.get_game_over(history[-1].state)
    assert history[-1].action is None
    assert history[-1].action_probabilities is None


def test_full_game_targets_use_final_shared_winners_without_resampling(monkeypatch):
    from skyjo import skynet

    first = completed_round(((0, 1, 2), (1, 2, 3), (2, 3, 4)), (30, 0, 0))
    final = completed_round(((11, 12, 12), (-1, 1, 3), (-1, 0, 1)), (33, 6, 9))
    # Player zero wins the first round but loses the game; players 1 and 2 tie.
    assert sj.get_fixed_perspective_winner(first) == 0
    initial = sj.start_round(sj.new(players=3), rng=random.Random(3))
    second_seat = sj.apply_action(
        initial, sj.MASK_FLIP_SECOND_BELOW, rng=random.Random(4)
    )
    later = sj.start_next_round(first, rng=random.Random(5))

    def history(states, terminal):
        entries = []
        for state in states:
            mask = sj.actions(state).astype(np.float32)
            entries.append(
                play.RoundHistoryEntry(
                    state, int(np.flatnonzero(mask)[0]), mask / mask.sum()
                )
            )
        return entries + [play.RoundHistoryEntry(terminal, None, None)]

    result = play.GameResult(
        (
            play.RoundResult(
                history([initial, second_seat], first), (3, 6, 9), (33, 6, 9), 0
            ),
            play.RoundResult(history([later], final), (70, 3, 0), (103, 9, 9), 0),
        )
    )

    def no_simulation(*args, **kwargs):
        raise AssertionError("Target conversion must use observed results")

    monkeypatch.setattr(sj, "apply_action", no_simulation)
    monkeypatch.setattr(sj, "start_next_round", no_simulation)
    monkeypatch.setattr(random, "random", no_simulation)
    data, stats = play.game_result_to_game_data(result)

    assert len(data) == stats.game_length == 3
    np.testing.assert_array_equal(data[0].targets["value"], [0, 0.5, 0.5])
    np.testing.assert_array_equal(data[1].targets["value"], [0.5, 0.5, 0])
    np.testing.assert_array_equal(data[2].targets["value"], [0, 0.5, 0.5])
    for row in data:
        assert set(row.targets) == {"value", "policy"}
        assert sj.actions(row.state)[row.action]
        assert row.targets["policy"].sum() == pytest.approx(1)
    np.testing.assert_array_equal(stats.scores_state_value, [103, 9, 9])
    np.testing.assert_array_equal(
        skynet.skyjo_to_game_state_value(final), [0, 0.5, 0.5]
    )
    assert stats.action_counts.sum() == 3


@pytest.mark.parametrize("finishes_game", [False, True])
def test_search_caches_one_boundary_sample_and_bootstraps_in_fixed_order(
    monkeypatch, finishes_game
):
    from skyjo import mcts, skynet

    # All replacement choices are deterministic. Ending the turn moves to seat 1.
    completed = completed_round(
        ((0, 1, 2), (3, 4, 5), (6, 7, 8)),
        (200, 0, 0) if finishes_game else (10, 20, 30),
    )
    state = sj.apply_action(dataclasses.replace(completed, countdown=2), sj.MASK_TAKE)
    assert sj.get_round_about_to_end(state)
    assert all(
        not sj.is_action_random(int(action), state) for action in sj.get_actions(state)
    )

    class FixedPredictor:
        def __init__(self):
            self.states = []

        def predict(self, state):
            self.states.append(state)
            mask = sj.actions(state).astype(np.float32)
            assert mask.any(), "Never send a completed round to the model"
            return skynet.SkyNetPrediction(
                value_output=np.array([0.1, 0.2, 0.7], dtype=np.float32),
                policy_output=mask / mask.sum(),
            )

    client = FixedPredictor()
    applications = []
    apply = sj.apply_action

    def record_apply(state, action, **kwargs):
        applications.append(action)
        return apply(state, action, **kwargs)

    monkeypatch.setattr(sj, "apply_action", record_apply)
    root = mcts.run_mcts(state, client, iterations=100)
    assert len(applications) == 3  # Exactly one per legal boundary action.
    assert (
        root.visit_count
        == sum(child.visit_count for child in root.children.values())
        == 100
    )
    expected = [0, 1, 0] if finishes_game else [0.7, 0.1, 0.2]
    np.testing.assert_allclose(root.state_value, expected, atol=1e-6)
    for child in root.children.values():
        np.testing.assert_allclose(child.state_value, expected, atol=1e-6)
        if not finishes_game:
            assert sj.get_player(child.next_round_state) == 1
            assert sj.get_game_scores(child.next_round_state).any()
    prediction_count = len(client.states)
    assert prediction_count == (1 if finishes_game else 4)
    mcts.run_mcts(state, client, iterations=10, root_node=root)
    assert len(applications) == 3
    assert len(client.states) == prediction_count
    assert root.visit_count == 110


def test_full_game_replay_keeps_rounds_together(tmp_path, equal_hands):
    from skyjo import buffer

    result = play.play_game([NaiveQuickFinishPlayer() for _ in range(3)])
    assert len(result.rounds) == 3
    data, _ = play.game_result_to_game_data(result)
    game_length = sum(len(round_result.history) - 1 for round_result in result.rounds)
    assert len(data) == game_length
    replay = buffer.ReplayBuffer(
        max_size=game_length * 2,
        spatial_input_shape=(3, sj.ROW_COUNT, sj.COLUMN_COUNT, sj.FINGER_SIZE),
        non_spatial_input_shape=observations.get_non_spatial_input_shape(3),
        action_mask_shape=(sj.MASK_SIZE,),
    )
    for game_index in (4, 5, 6):
        replay.add_game_data(data, game_index=game_index)
    assert replay.game_indices == (5, 6)
    assert len(replay) == game_length * 2
    loaded = buffer.ReplayBuffer.load(replay.save(tmp_path / "all-games"))
    training, validation = loaded.split_by_game(0.5, seed=0)
    assert set(training.game_indices).isdisjoint(validation.game_indices)
    assert set(training.game_indices + validation.game_indices) == {5, 6}
    for number, selected in enumerate((training, validation)):
        assert len(selected) == game_length
        batch = selected.ordered_batch()
        assert batch.non_spatial_inputs[:, sj.GAME_SCORES : sj.GAME_SCORES + 3].any()
        resaved = buffer.ReplayBuffer.load(selected.save(tmp_path / f"split-{number}"))
        np.testing.assert_array_equal(
            resaved.ordered_batch().targets["value"], batch.targets["value"]
        )


def test_round_statistics_count_turns_and_player_rounds(equal_hands):
    from skyjo import game_stats

    result = play.play_game([NaiveQuickFinishPlayer() for _ in range(3)])
    stats = game_stats.analyze_game(result)
    summary = game_stats.summarize_games([stats])
    assert len(stats.rounds) == 3
    for observed in stats.rounds:
        assert observed.turns == 30
        assert observed.decisions == 63  # Three setup reveals, two decisions/turn.
        assert observed.raw_scores == (21, 21, 21)
        assert observed.scores == (42, 21, 21)
        assert observed.ending_reason == "natural"
        assert not observed.partial_start
    assert summary["round/score/count"] == 9
    assert summary["round/score/mean"] == 28
    assert summary["round/score_adjustment_rate"] == pytest.approx(1 / 3)
    assert summary["round/turns/p90"] == 30
    assert summary["round/no_progress_rate"] == 0
    assert summary["game/rounds/mean"] == 3


def test_round_statistics_preserve_seats_clears_and_partial_history():
    from skyjo import game_stats

    final = completed_round(
        ((-2, 0, 1), (0, 1, 2), (1, 2, 3)), (10, 20, 90), ending_player=2
    )
    # A partial round already in its last turn. Its end trigger is not observed.
    before = dataclasses.replace(final, countdown=1)
    history = [
        play.RoundHistoryEntry(before, sj.MASK_REPLACE, None),
        play.RoundHistoryEntry(final, None, None),
    ]
    result = play.GameResult((play.RoundResult(history, (-1, 3, 12), (9, 23, 102), 2),))
    stats = game_stats.analyze_game(result)
    observed = stats.rounds[0]
    assert observed.raw_scores == (-1, 3, 6)
    assert observed.cumulative_scores == (9, 23, 102)
    assert observed.cleared_columns == (3, 3, 3)
    assert observed.partial_start and observed.ending_reason == "unknown"
    summary = game_stats.summarize_games([stats])
    assert summary["round/turns/count"] == 0
    assert "round/turns/mean" not in summary
    assert "round/no_progress_rate" not in summary
    assert summary["round/score/count"] == 3


def test_no_progress_end_is_identified_before_automatic_reveals():
    from skyjo import game_stats

    class StallPlayer(player.AbstractPlayer):
        def get_action_probabilities(self, state):
            phase = sj.get_action(state)
            if phase == sj.ACTION_FLIP_SECOND:
                action = sj.MASK_FLIP_SECOND_RIGHT
            elif phase == sj.ACTION_DRAW_OR_TAKE:
                action = sj.MASK_TAKE
            else:
                action = sj.MASK_REPLACE  # Repeatedly replace a known card.
            return self._action_to_action_probabilities(action, state)

    history = play.play_round([StallPlayer(), StallPlayer()])
    final = history[-1].state
    result = play.RoundResult(
        history,
        tuple(sj.get_fixed_perspective_round_scores(final)),
        tuple(sj.get_fixed_perspective_game_scores(final)),
        sj.get_player(final),
    )
    stats = game_stats.analyze_round(result)
    assert stats.ending_reason == "no_progress"
    assert not stats.partial_start
    assert all(sj.get_is_visible(final, i) for i in range(2))
