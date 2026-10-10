"""Completed rounds are sampled once per action and evaluated before averaging."""

import dataclasses

import numpy as np
import pytest
from test_full_game import completed_round

from skyjo.engine import game as sj
from skyjo.learning.boundary_inference import ScoreBoundaryEvaluator
from skyjo.search import mcts
from skyjo.search.evaluator import Prediction


class RootOnlyEvaluator:
    def __init__(self):
        self.states = []

    def evaluate(self, states):
        self.states.extend(states)
        return [
            Prediction(
                np.array([0.1, 0.2, 0.7], dtype=np.float32),
                sj.actions(state) / sj.actions(state).sum(),
            )
            for state in states
        ]


def test_search_caches_sampled_values_before_averaging_without_dealing(monkeypatch):
    first = completed_round(
        ((-2, 0, 1), (0, 1, 2), (1, 2, 3)), (10, 20, 30), ending_player=2
    )
    second = completed_round(
        ((-2, 0, 1), (0, 1, 2), (1, 2, 3)), (20, 30, 40), ending_player=2
    )
    state = sj.apply_action(dataclasses.replace(first, countdown=2), sj.MASK_TAKE)
    card = int(np.argmax(state.table[1, 0, 0]))
    state.table[1, 0, 0] = 0
    state.table[1, 0, 0, sj.FINGER_HIDDEN] = 1
    state.deck[card] += 1
    applications, evaluated = [], []

    def sample(unchanged, action, *, rng):
        assert unchanged is state
        applications.append(action)
        return first if len(applications) % 2 else second

    class Boundary:
        def evaluate(self, states, rng):
            assert len(states) == 10
            evaluated.append(states)
            return np.array(
                [
                    [0.1, 0.2, 0.7] if item is first else [0.5, 0.4, 0.1]
                    for item in states
                ],
                dtype=np.float32,
            )

    monkeypatch.setattr(sj, "apply_action", sample)
    evaluator, boundary = RootOnlyEvaluator(), Boundary()
    config = mcts.SearchConfig(boundary_samples=10)
    root = mcts.run_mcts(
        state, evaluator, 100, config=config, boundary_evaluator=boundary
    )
    assert len(evaluated) == len(root.children) == 3
    assert len(applications) == 30
    assert len(evaluator.states) == 1
    np.testing.assert_allclose(root.state_value, [0.3, 0.3, 0.4], atol=1e-6)
    assert sum(child.visit_count for child in root.children.values()) == 100
    mcts.run_mcts(state, evaluator, 10, config=config, root_node=root)
    assert len(applications) == 30


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


@pytest.mark.parametrize("samples", [0, 1.5, True])
def test_invalid_sample_budget_fails_before_search(samples):
    with pytest.raises(ValueError, match="positive integer"):
        mcts.SearchConfig(boundary_samples=samples)


def test_deterministic_round_completion_preserves_next_deal_sample_budget():
    completed = completed_round(((0, 1, 2), (3, 4, 5), (6, 7, 8)), (10, 20, 30))
    state = sj.apply_action(dataclasses.replace(completed, countdown=2), sj.MASK_TAKE)
    assert not state.table[:, :, :, sj.FINGER_HIDDEN].any()
    evaluator = RootOnlyEvaluator()
    root = mcts.run_mcts(
        state,
        evaluator,
        1,
        config=mcts.SearchConfig(boundary_samples=10),
        rng=np.random.default_rng(11),
    )
    # One root prediction and all ten independently dealt continuing states.
    assert len(evaluator.states) == 11
    deals = evaluator.states[1:]
    assert all(not sj.get_round_over(deal) for deal in deals)
    assert len({sj.hash_skyjo(deal) for deal in deals}) > 1
    np.testing.assert_allclose(root.state_value, [0.1, 0.2, 0.7], atol=1e-6)
