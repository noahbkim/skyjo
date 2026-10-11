"""Boundary outcome estimates and per-traversal returns have distinct weights."""

import dataclasses

import numpy as np
import pytest
from test_full_game import completed_round

from skyjo.engine import game as sj
from skyjo.learning.boundary_inference import ScoreBoundaryEvaluator
from skyjo.search import mcts
from skyjo.search.evaluator import NextDealEvaluator, Prediction


class RootOnlyEvaluator:
    def __init__(self, action=None):
        self.states = []
        self.action = action

    def evaluate(self, states):
        predictions = []
        for state in states:
            self.states.append(state)
            policy = sj.actions(state).astype(np.float32)
            if self.action is not None and len(self.states) == 1:
                assert policy[self.action]
                policy.fill(0)
                policy[self.action] = 1
            predictions.append(
                Prediction(
                    np.array([0.1, 0.2, 0.7], dtype=np.float32),
                    policy / policy.sum(),
                )
            )
        return predictions


def hidden_boundary_state(completed):
    state = sj.apply_action(dataclasses.replace(completed, countdown=2), sj.MASK_TAKE)
    state = dataclasses.replace(state, table=state.table.copy(), deck=state.deck.copy())
    card = int(np.argmax(state.table[1, 0, 0]))
    state.table[1, 0, 0] = 0
    state.table[1, 0, 0, sj.FINGER_HIDDEN] = 1
    state.deck[card] += 1
    return state


@pytest.mark.parametrize("iterations", [1, 2, 4])
def test_boundary_pools_outcomes_but_backs_up_fresh_returns(monkeypatch, iterations):
    first = completed_round(
        ((-2, 0, 1), (0, 1, 2), (1, 2, 3)), (10, 20, 30), ending_player=2
    )
    second = completed_round(
        ((-2, 0, 1), (0, 1, 2), (1, 2, 3)), (20, 30, 40), ending_player=2
    )
    state = hidden_boundary_state(first)
    applications, evaluated = [], []

    def sample(unchanged, action, *, rng):
        assert unchanged is state
        applications.append(action)
        return first if len(applications) <= 2 else second

    class Boundary:
        def evaluate(self, states, rng):
            evaluated.append(states)
            return np.array(
                [[0, 1, 0] if item is first else [1, 0, 0] for item in states],
                dtype=np.float32,
            )

    monkeypatch.setattr(sj, "apply_action", sample)
    evaluator, boundary = RootOnlyEvaluator(sj.MASK_REPLACE), Boundary()
    config = mcts.SearchConfig(boundary_samples=2, fpu_reduction=1)
    root = mcts.run_mcts(
        state, evaluator, iterations, config=config, boundary_evaluator=boundary
    )
    child = root.children[sj.MASK_REPLACE]
    assert [len(batch) for batch in evaluated] == [2] + [1] * (iterations - 1)
    assert applications == [sj.MASK_REPLACE] * (iterations + 1)
    assert len(evaluator.states) == 1
    boundary_value, ancestor_value = {1: (0, 0), 2: (1 / 3, 0.5), 4: (0.6, 0.75)}[
        iterations
    ]
    np.testing.assert_allclose(
        child.state_value, [boundary_value, 1 - boundary_value, 0]
    )
    np.testing.assert_allclose(
        root.state_value, [ancestor_value, 1 - ancestor_value, 0]
    )
    assert root.visit_count == child.visit_count == iterations
    assert child.sample_count == iterations + 1


def test_terminal_boundaries_use_exact_ties_without_a_model():
    completed = completed_round(((0, 1, 2), (1, 2, 3), (1, 2, 3)), (200, 0, 6))
    state = sj.apply_action(dataclasses.replace(completed, countdown=2), sj.MASK_TAKE)
    evaluator = RootOnlyEvaluator()
    root = mcts.run_mcts(
        state,
        evaluator,
        100,
        config=mcts.SearchConfig(boundary_samples=10),
        boundary_evaluator=ScoreBoundaryEvaluator(None),
    )
    assert len(evaluator.states) == 1
    np.testing.assert_allclose(root.state_value, [0, 0.5, 0.5], atol=1e-6)
    assert all(child.sample_count == 1 for child in root.children.values())


@pytest.mark.parametrize("samples", [0, 1.5, True])
def test_invalid_sample_budget_fails_before_search(samples):
    with pytest.raises(ValueError, match="positive integer"):
        mcts.SearchConfig(boundary_samples=samples)


def test_deterministic_round_completion_preserves_next_deal_sample_budget():
    completed = completed_round(((0, 1, 2), (3, 4, 5), (6, 7, 8)), (10, 20, 30))
    state = sj.apply_action(dataclasses.replace(completed, countdown=2), sj.MASK_TAKE)
    assert not state.table[:, :, :, sj.FINGER_HIDDEN].any()
    evaluator = RootOnlyEvaluator(sj.MASK_REPLACE)
    config = mcts.SearchConfig(boundary_samples=10, fpu_reduction=1)
    root = mcts.run_mcts(
        state,
        evaluator,
        3,
        config=config,
        rng=np.random.default_rng(11),
    )
    # One root prediction, ten initial deals, and two fresh revisit deals.
    assert len(evaluator.states) == 13
    deals = evaluator.states[1:]
    assert all(not sj.get_round_over(deal) for deal in deals)
    assert len({sj.hash_skyjo(deal) for deal in deals}) > 1
    np.testing.assert_allclose(root.state_value, [0.1, 0.2, 0.7], atol=1e-6)
    assert root.children[sj.MASK_REPLACE].sample_count == 12
    assert all(
        sj.hash_skyjo(deal) not in {sj.hash_skyjo(old) for old in deals[:10]}
        for deal in evaluator.states[11:]
    )


def test_stochastic_terminal_batch_does_not_cache_future_continuing_outcomes(
    monkeypatch,
):
    hands = ((0, 1, 2), (1, 2, 3), (1, 2, 3))
    terminal = completed_round(hands, (200, 0, 0))
    continuing = completed_round(hands, (10, 20, 30))
    state = hidden_boundary_state(terminal)
    outcomes = iter([terminal, terminal, continuing, terminal])
    applications = []

    def sample(unchanged, action, *, rng):
        assert unchanged is state
        applications.append(action)
        return next(outcomes)

    monkeypatch.setattr(sj, "apply_action", sample)
    root_evaluator = RootOnlyEvaluator(sj.MASK_REPLACE)
    deal_evaluator = RootOnlyEvaluator()
    config = mcts.SearchConfig(boundary_samples=2, fpu_reduction=1)
    root = mcts.run_mcts(
        state,
        root_evaluator,
        3,
        config=config,
        boundary_evaluator=NextDealEvaluator(deal_evaluator),
        rng=np.random.default_rng(3),
    )
    assert len(applications) == 4
    assert len(deal_evaluator.states) == 1
    assert not sj.get_round_over(deal_evaluator.states[0])
    child = root.children[sj.MASK_REPLACE]
    np.testing.assert_allclose(child.state_value, [0.025, 0.425, 0.55])
    np.testing.assert_allclose(root.state_value, [0.1 / 3, 1.2 / 3, 1.7 / 3])
