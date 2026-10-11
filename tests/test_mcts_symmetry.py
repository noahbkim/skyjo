"""Grouped action behavior, inherited settings, and search-budget safety."""

import numpy as np
import pytest
from test_action_symmetry import RECYCLING_BOARD, board_state, with_slack

from skyjo.engine import game as sj
from skyjo.search import mcts
from skyjo.search.evaluator import Prediction


class FixedEvaluator:
    def __init__(self, first_action=None, force_draw_path=False):
        self.states = []
        self.first_action = first_action
        self.force_draw_path = force_draw_path

    def evaluate(self, states):
        predictions = []
        for state in states:
            self.states.append(state)
            policy = sj.actions(state).astype(np.float32)
            if self.first_action is not None and len(self.states) == 1:
                policy.fill(0)
                policy[self.first_action] = 1
            elif self.force_draw_path:
                action = (
                    sj.MASK_DRAW
                    if sj.get_action(state) == sj.ACTION_DRAW_OR_TAKE
                    else sj.MASK_REPLACE + 1
                )
                assert policy[action]
                policy.fill(0)
                policy[action] = 1
            else:
                policy *= np.arange(1, sj.MASK_SIZE + 1, dtype=np.float32)
            predictions.append(
                Prediction(
                    np.full(state.players, 1 / state.players, dtype=np.float32),
                    policy / policy.sum(),
                )
            )
        return predictions


def decision_nodes(root):
    pending = [root]
    while pending:
        node = pending.pop()
        if isinstance(node, mcts.DecisionStateNode):
            yield node
        if not isinstance(node, mcts.RoundBoundaryNode):
            pending.extend(node.children.values())


def test_expansion_aggregates_unequal_priors_and_root_noise():
    state, evaluator = board_state(), FixedEvaluator()
    config = mcts.SearchConfig(dirichlet_epsilon=0.25)
    root = mcts.run_mcts(
        state, evaluator, 0, config=config, rng=np.random.default_rng(4)
    )
    assert len(root.children) < sj.actions(state).sum()
    assert root.dirichlet_noise.sum() == pytest.approx(1)
    for group in root.action_groups.members:
        expected = sum(
            0.75 * root.model_prediction.policy[action]
            + 0.25 * root.dirichlet_noise[action]
            for action in group
        )
        assert root.action_probability(group[0]) == pytest.approx(expected)
    assert len(evaluator.states) == 1


@pytest.mark.parametrize("exact_chance", [False, True])
@pytest.mark.parametrize("merge", [False, True])
def test_search_inherits_group_mode_and_conserves_visits(exact_chance, merge):
    state = board_state(phase=sj.ACTION_DRAW_OR_TAKE)
    config = mcts.SearchConfig(
        merge_symmetric_actions=merge,
        after_state_evaluate_all_children=exact_chance,
        fpu_reduction=1,
    )
    root = mcts.run_mcts(
        state,
        FixedEvaluator(sj.MASK_DRAW),
        12,
        config=config,
        rng=np.random.default_rng(41),
    )
    assert (
        root.visit_count
        == sum(child.visit_count for child in root.children.values())
        == 12
    )
    decisions = list(decision_nodes(root))
    assert len(decisions) > 2
    for node in decisions:
        assert node.context is root.context
        assert node.effective_merge_symmetric_actions is merge
        child_visits = sum(child.visit_count for child in node.children.values())
        if node.is_expanded:
            assert 0 <= node.visit_count - child_visits <= 1
            assert tuple(node.children) == node.action_groups.representatives
            if not merge:
                assert len(node.children) == sj.actions(node.state).sum()
        if child_visits:
            policy = node.policy_targets()
            assert policy.sum() == pytest.approx(1)
            assert not policy[~sj.actions(node.state).astype(bool)].any()
            for group in node.action_groups.members:
                np.testing.assert_array_equal(
                    policy[list(group)], np.full(len(group), policy[group[0]])
                )
    chance = root.children[sj.MASK_DRAW]
    if exact_chance:
        assert sum(chance.child_weights.values()) == pytest.approx(1)
        assert len(chance.children) == np.count_nonzero(state.deck)


def test_recycling_falls_back_to_ordinary_search():
    state = board_state(
        RECYCLING_BOARD,
        phase=sj.ACTION_REPLACE,
        countdown=1,
        opponents_cleared=True,
        deck_counts={1: 2},
    )
    evaluator = FixedEvaluator()
    root = mcts.run_mcts(state, evaluator, 2)
    assert not root.effective_merge_symmetric_actions
    assert tuple(root.children) == tuple(sj.get_actions(state))
    assert root.children[16] is not root.children[21]


@pytest.mark.parametrize("iterations, merged", [(1, True), (3, False)])
def test_full_search_budget_controls_inherited_symmetry_mode(iterations, merged):
    state = with_slack(board_state(phase=sj.ACTION_REPLACE), 2)
    root = mcts.run_mcts(
        state, FixedEvaluator(), iterations, rng=np.random.default_rng(4)
    )
    assert all(
        node.effective_merge_symmetric_actions is merged
        for node in decision_nodes(root)
    )


@pytest.mark.parametrize("exact_chance", [False, True])
def test_deepest_draw_path_respects_slack_bound(exact_chance, monkeypatch):
    board = (("H", 0, "X", "X"), (2, 2, "X", "X"), (3, 3, "X", "X"))
    state = board_state(board, phase=sj.ACTION_DRAW_OR_TAKE, deck_counts={1: 7})
    evaluator = FixedEvaluator(force_draw_path=True)
    prepare = sj.prepare_draw_pile

    def no_recycling(observed):
        assert observed.deck.sum() > 0
        return prepare(observed)

    monkeypatch.setattr(sj, "prepare_draw_pile", no_recycling)
    config = mcts.SearchConfig(
        fpu_reduction=1, after_state_evaluate_all_children=exact_chance
    )
    root = mcts.run_mcts(state, evaluator, 8, config=config)
    assert root.effective_merge_symmetric_actions
    assert min(int(observed.deck.sum()) for observed in evaluator.states) == 3
    assert root.visit_count == 8


@pytest.mark.parametrize("first_action", [sj.MASK_TAKE, sj.MASK_DRAW])
@pytest.mark.parametrize("score_boundary", [False, True])
def test_grouped_descendants_reach_supplied_boundary_evaluators(
    first_action, score_boundary
):
    state = board_state(
        RECYCLING_BOARD,
        phase=sj.ACTION_DRAW_OR_TAKE,
        countdown=2,
        opponents_cleared=True,
    )
    evaluator = FixedEvaluator(first_action)
    completed = []

    class Scores:
        def evaluate(self, states, rng):
            assert all(sj.get_round_over(state) for state in states)
            completed.extend(states)
            return np.tile([0.4, 0.6], (len(states), 1)).astype(np.float32)

    config = mcts.SearchConfig(fpu_reduction=1, boundary_samples=3)
    root = mcts.run_mcts(
        state,
        evaluator,
        12,
        config=config,
        boundary_evaluator=Scores() if score_boundary else None,
        rng=np.random.default_rng(33),
    )
    descendants = [
        node for node in decision_nodes(root) if node is not root and node.is_expanded
    ]
    assert descendants
    assert all(node.effective_merge_symmetric_actions for node in descendants)
    assert any(
        len(node.children) < sj.actions(node.state).sum() for node in descendants
    )
    if score_boundary:
        assert completed
    assert (
        root.visit_count
        == sum(child.visit_count for child in root.children.values())
        == 12
    )


@pytest.mark.parametrize("iterations", [-1, 1.5, True])
def test_invalid_iteration_budget_fails_before_inference(iterations):
    with pytest.raises(ValueError, match="nonnegative integer"):
        mcts.run_mcts(None, None, iterations)
