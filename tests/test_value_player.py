"""One-action expectations, fixed-seat values, and reproducible chance sampling."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from test_full_game import completed_round
from test_mcts_boundary_sampling import hidden_boundary_state

from skyjo.engine import game as sj
from skyjo.search.batching import evaluate_concurrently
from skyjo.search.evaluator import Prediction
from skyjo.search.value_player import ValueConfig, ValuePlayer
from skyjo.simulation.decisions import validate_decision


class CheckedEvaluator:
    def __init__(self, value):
        self.value = value
        self.batches = []

    def evaluate(self, states):
        self.batches.append(list(states))
        # Poison the unused policy to catch accidental dependence on it.
        return [
            Prediction(np.asarray(self.value(state)), np.full(sj.MASK_SIZE, np.nan))
            for state in states
        ]


def action_values(decision):
    section = decision.diagnostics[0]
    return dict(zip(section.row_actions, (row[1] for row in section.rows), strict=True))


@pytest.mark.parametrize("chance", ["enumerated", "sampled"])
def test_chance_weights_and_sample_averages_batch_only_immediate_successors(chance):
    state = replace(completed_round(((1, 2, 3), (4, 5, 6)), (0, 0)), countdown=None)
    deck = np.zeros_like(state.deck)
    deck[sj.CARD_0], deck[sj.CARD_P12] = 1, 3
    state = replace(state, deck=deck)
    original = sj.hash_skyjo(state)

    def value(child):
        phase = sj.get_action(child)
        assert phase in (sj.ACTION_FLIP_OR_REPLACE, sj.ACTION_REPLACE)
        assert sj.get_player(child) == sj.get_player(state)
        p = (
            0.6
            if phase == sj.ACTION_REPLACE
            else (0.1 if sj.get_top(child) == sj.CARD_0 else 0.9)
        )
        return [p, 1 - p]

    evaluator = CheckedEvaluator(value)
    draws = iter([0.1, 0.3, 0.5, 0.8])  # One low card and three high cards.
    rng = SimpleNamespace(random=lambda: next(draws))
    decision = ValuePlayer(
        evaluator, config=ValueConfig(chance=chance, samples=4), rng=rng
    ).get_decision(state)
    validate_decision(state, decision)
    assert action_values(decision) == pytest.approx(
        {sj.MASK_DRAW: 0.7, sj.MASK_TAKE: 0.6}
    )
    assert (
        decision.probabilities.argmax() == sj.MASK_DRAW
    )  # Unweighted averaging would take.
    assert [len(batch) for batch in evaluator.batches] == [
        3 if chance == "enumerated" else 5
    ]
    assert sj.hash_skyjo(state) == original
    assert list(draws) == ([0.1, 0.3, 0.5, 0.8] if chance == "enumerated" else [])


def test_value_selection_keeps_original_seat_after_turn_changes():
    state = replace(
        completed_round(((1, 2, 3), (4, 5, 6)), (0, 0), ending_player=1), countdown=None
    )
    state = sj.apply_action(state, sj.MASK_TAKE)
    actions = tuple(sj.get_actions(state))
    expected = {}
    for i, action in enumerate(actions):
        child = sj.apply_action(state, action)
        assert sj.get_player(child) == 0
        expected[sj.hash_skyjo(child)] = [0.9 - i * 0.3, 0.1 + i * 0.3]
    evaluator = CheckedEvaluator(lambda child: expected[sj.hash_skyjo(child)])
    decision = ValuePlayer(evaluator).get_decision(state)
    assert decision.probabilities.argmax() == actions[-1]
    np.testing.assert_allclose(list(action_values(decision).values()), [0.1, 0.4, 0.7])
    assert [len(batch) for batch in evaluator.batches] == [len(actions)]


def test_terminal_values_share_ties_without_calling_any_evaluator():
    completed = completed_round(((10, 11, 12), (10, 11, 12)), (100, 101))
    state = sj.apply_action(replace(completed, countdown=2), sj.MASK_TAKE)
    evaluator = CheckedEvaluator(lambda _: pytest.fail("Terminal values must be exact"))

    class Boundary:
        def evaluate(self, states, rng):
            pytest.fail("Terminal games must bypass the boundary model")

    decision = ValuePlayer(
        evaluator, boundary_evaluator=Boundary(), config=ValueConfig(boundary_samples=3)
    ).get_decision(state)
    actions = tuple(sj.get_actions(state))
    np.testing.assert_allclose(list(action_values(decision).values()), [0, 0.5, 1])
    assert decision.probabilities.argmax() == actions[-1]
    assert not evaluator.batches


@pytest.mark.parametrize("custom_boundary", [False, True])
def test_continuing_boundaries_use_next_deals_or_the_supplied_evaluator(
    custom_boundary,
):
    completed = completed_round(((10, 11, 12), (10, 11, 12)), (0, 0))
    state = sj.apply_action(replace(completed, countdown=2), sj.MASK_TAKE)
    original = sj.hash_skyjo(state)

    def next_deal_value(child):
        assert not custom_boundary
        assert sj.get_action(child) == sj.ACTION_FLIP_SECOND
        assert not sj.get_round_over(child)
        assert sj.get_game_scores(child).sum() > 0
        return [0.2, 0.8]

    evaluator = CheckedEvaluator(next_deal_value)

    class Boundary:
        def evaluate(self, states, rng):
            assert len(states) == 9  # Three actions, three round realizations each.
            assert all(
                sj.get_round_over(child) and not sj.get_game_over(child)
                for child in states
            )
            return np.tile([0.7, 0.3], (len(states), 1))

    decision = ValuePlayer(
        evaluator,
        boundary_evaluator=Boundary() if custom_boundary else None,
        config=ValueConfig(boundary_samples=3),
        rng=np.random.default_rng(5),
    ).get_decision(state)
    np.testing.assert_allclose(
        list(action_values(decision).values()), 0.7 if custom_boundary else 0.2
    )
    assert decision.probabilities.argmax() == min(sj.get_actions(state))
    assert [len(batch) for batch in evaluator.batches] == (
        [] if custom_boundary else [9]
    )
    assert sj.hash_skyjo(state) == original


@pytest.mark.parametrize("chance", ["enumerated", "sampled"])
def test_cooperative_values_and_rng_match_synchronous_play(chance):
    completed = completed_round(((1, 2, 3), (4, 5, 6)), (0, 0))
    states = [replace(completed, countdown=None), hidden_boundary_state(completed)]
    evaluator = CheckedEvaluator(
        lambda child: [sj.get_top(child) / 20, 1 - sj.get_top(child) / 20]
    )
    config = ValueConfig(chance=chance, samples=4, boundary_samples=3)
    expected = []
    for seed, state in enumerate(states):
        rng = np.random.default_rng(seed)
        decision = ValuePlayer(evaluator, config=config, rng=rng).get_decision(state)
        expected.append((decision, rng.bit_generator.state))
    streams = [np.random.default_rng(seed) for seed in range(len(states))]
    actual = evaluate_concurrently(
        (
            ValuePlayer(evaluator, config=config, rng=rng).decision_steps(
                state, diagnostics=True
            )
            for state, rng in zip(states, streams, strict=True)
        ),
        concurrency=2,
    )
    for (reference, rng_state), decision, rng in zip(
        expected, actual, streams, strict=True
    ):
        np.testing.assert_array_equal(reference.probabilities, decision.probabilities)
        assert reference.diagnostics == decision.diagnostics
        assert rng_state == rng.bit_generator.state


@pytest.mark.parametrize(
    "options", [{"chance": "other"}, {"samples": 0}, {"boundary_samples": 0}]
)
def test_value_config_rejects_unusable_sampling_settings(options):
    with pytest.raises(ValueError):
        ValueConfig(**options)
