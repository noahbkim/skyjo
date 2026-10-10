"""Search integration: shared priors, visits, inherited modes and safe reuse."""

from __future__ import annotations

import dataclasses
import random

import numpy as np
import pytest

from skyjo import boundary_inference
from skyjo import game as sj
from skyjo import mcts, skynet, symmetry
from test_action_symmetry import RECYCLING_BOARD, board_state, with_slack


class FixedPredictor:
    def __init__(self, *, first_action=None, force_draw_path=False):
        self.states = []
        self.first_action = first_action
        self.force_draw_path = force_draw_path

    def predict(self, state):
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
        return skynet.SkyNetPrediction(
            value_output=np.full(state.players, 1 / state.players, dtype=np.float32),
            policy_output=policy / policy.sum(),
        )

    def predict_many(self, states):
        return [self.predict(state) for state in states]


def decision_nodes(root):
    pending = [root]
    while pending:
        node = pending.pop()
        if isinstance(node, mcts.DecisionStateNode):
            yield node
        if not isinstance(node, mcts.RoundBoundaryNode):
            pending.extend(node.children.values())


def test_expansion_aggregates_unequal_priors_and_each_root_noise_refresh(monkeypatch):
    state = board_state()
    client = FixedPredictor()
    draws = []

    def draw_noise(alpha):
        draws.append(alpha.copy())
        weights = np.arange(1, len(alpha) + 1, dtype=np.float64)
        return weights / weights.sum()

    monkeypatch.setattr(np.random, "dirichlet", draw_noise)
    root = mcts.run_mcts(state, client, 0, dirichlet_epsilon=0.25)
    assert len(draws) == 1
    assert len(draws[0]) == sj.actions(state).sum()
    assert root.action_groups is not None
    assert tuple(root.children) == root.action_groups.representatives
    assert len(root.children) < sj.actions(state).sum()
    for group in root.action_groups.members:
        expected = sum(
            0.75 * root.model_prediction.policy_output[action]
            + 0.25 * root.dirichlet_noise[action]
            for action in group
        )
        assert root.action_probability(group[0]) == pytest.approx(expected)

    mcts.run_mcts(state, client, 0, root_node=root)
    for group in root.action_groups.members:
        assert root.action_probability(group[0]) == pytest.approx(
            root.model_prediction.policy_output[list(group)].sum()
        )
    root.select_child()
    assert not root.dirichlet_noise.any()
    assert len(client.states) == 1


@pytest.mark.parametrize("exact_chance", [False, True])
@pytest.mark.parametrize("merge", [False, True])
def test_search_inherits_group_mode_and_conserves_visits_below_root(exact_chance, merge):
    random.seed(41)
    np.random.seed(41)
    state = board_state(phase=sj.ACTION_DRAW_OR_TAKE)
    root = mcts.run_mcts(
        state, FixedPredictor(first_action=sj.MASK_DRAW), 12,
        merge_symmetric_actions=merge,
        after_state_evaluate_all_children=exact_chance,
        fpu_reduction=1,
    )
    assert root.visit_count == sum(child.visit_count for child in root.children.values()) == 12
    decisions = list(decision_nodes(root))
    assert len(decisions) > 2
    expanded = [node for node in decisions if node.is_expanded]
    assert any(sj.get_action(node.state) == sj.ACTION_FLIP_OR_REPLACE for node in expanded)
    for node in decisions:
        assert node.merge_symmetric_actions is merge
        assert node.effective_merge_symmetric_actions is merge
        child_visits = sum(child.visit_count for child in node.children.values())
        if node.is_expanded:
            # Directly expanded leaves back up their initial evaluation once;
            # chance outcomes are evaluated before receiving their first visit.
            assert 0 <= node.visit_count - child_visits <= 1
            assert tuple(node.children) == node.action_groups.representatives
            if not merge:
                assert len(node.children) == sj.actions(node.state).sum()
        if child_visits:
            policy = node.policy_targets()
            assert np.isfinite(policy).all()
            assert policy.sum() == pytest.approx(1)
            assert not policy[~sj.actions(node.state).astype(bool)].any()
            for group in node.action_groups.members:
                np.testing.assert_array_equal(
                    policy[list(group)], np.full(len(group), policy[group[0]])
                )
    chance = root.children[sj.MASK_DRAW]
    assert isinstance(chance, mcts.AfterStateNode)
    if exact_chance:
        assert chance.child_weight_total == pytest.approx(1)
        assert sum(chance.child_weights.values()) == pytest.approx(1)
        assert len(chance.children) == np.count_nonzero(state.deck)


def test_recycling_falls_back_to_ordinary_search_and_never_upgrades():
    state = board_state(
        RECYCLING_BOARD, phase=sj.ACTION_REPLACE, countdown=1,
        opponents_cleared=True, deck_counts={1: 2},
    )
    root = mcts.run_mcts(state, FixedPredictor(), 2)
    assert root.merge_symmetric_actions
    assert not root.effective_merge_symmetric_actions
    assert tuple(root.children) == tuple(sj.get_actions(state))
    assert 16 in root.children and 21 in root.children
    assert root.children[16] is not root.children[21]
    mcts.run_mcts(state, FixedPredictor(), 0, root_node=root)
    assert not root.effective_merge_symmetric_actions


def test_reused_root_rejects_unsafe_budget_state_or_mode_before_mutation():
    state = with_slack(board_state(phase=sj.ACTION_REPLACE), 2)
    client = FixedPredictor()
    root = mcts.run_mcts(state, client, 1, dirichlet_epsilon=0.3)
    assert root.effective_merge_symmetric_actions
    assert mcts.run_mcts(state, client, 1, root_node=root, dirichlet_epsilon=0.3) is root
    assert root.visit_count == 2
    noise, value = root.dirichlet_noise.copy(), root.state_value_total.copy()
    children = tuple(
        (action, id(child), child.visit_count) for action, child in root.children.items()
    )
    for changed_state, iterations, settings in (
        (state, 1, {}),
        (dataclasses.replace(state, turn=state.turn + 1), 0, {}),
        (state, 0, {"merge_symmetric_actions": False}),
    ):
        with pytest.raises(ValueError):
            mcts.run_mcts(changed_state, client, iterations, root_node=root, **settings)
        np.testing.assert_array_equal(root.dirichlet_noise, noise)
        np.testing.assert_array_equal(root.state_value_total, value)
        assert root.dirichlet_epsilon == 0.3
        assert root.visit_count == 2
        assert tuple(
            (action, id(child), child.visit_count)
            for action, child in root.children.items()
        ) == children


def test_preexpanded_unvisited_chance_child_can_be_promoted_and_ordinary_child_stays_ordinary():
    state = board_state(phase=sj.ACTION_DRAW_OR_TAKE)
    for merge in (False, True):
        client = FixedPredictor(first_action=sj.MASK_DRAW)
        root = mcts.run_mcts(
            state, client, 1,
            after_state_evaluate_all_children=True, merge_symmetric_actions=merge,
        )
        chance = root.children[sj.MASK_DRAW]
        promoted = next(iter(chance.children.values()))
        assert promoted.is_expanded and promoted.visit_count == 0
        reused = mcts.run_mcts(
            promoted.state, client, 1,
            root_node=promoted, merge_symmetric_actions=merge,
        )
        assert reused is promoted
        assert promoted.effective_merge_symmetric_actions is merge
        assert promoted.visit_count == 1
        assert promoted.policy_targets().sum() == pytest.approx(1)

    # A fallback tree can become safe after advancing, but its existing groups
    # and cached priors remain ordinary when that descendant becomes the root.
    fallback_state = with_slack(state, 1)
    client = FixedPredictor(first_action=sj.MASK_TAKE)
    root = mcts.run_mcts(fallback_state, client, 1)
    ordinary_child = root.children[sj.MASK_TAKE]
    assert not ordinary_child.effective_merge_symmetric_actions
    assert symmetry.safe_to_merge_actions(ordinary_child.state, 0)
    mcts.run_mcts(ordinary_child.state, client, 0, root_node=ordinary_child)
    assert not ordinary_child.effective_merge_symmetric_actions
    assert len(ordinary_child.children) == sj.actions(ordinary_child.state).sum()


@pytest.mark.parametrize("exact_chance", [False, True])
def test_deepest_draw_path_respects_slack_bound(exact_chance, monkeypatch):
    board = (("H", 0, "X", "X"), (2, 2, "X", "X"), (3, 3, "X", "X"))
    state = board_state(board, phase=sj.ACTION_DRAW_OR_TAKE, deck_counts={1: 7})
    client = FixedPredictor(force_draw_path=True)
    prepare = sj.prepare_draw_pile

    def no_recycling(observed):
        assert observed.deck.sum() > 0
        return prepare(observed)

    monkeypatch.setattr(sj, "prepare_draw_pile", no_recycling)
    root = mcts.run_mcts(
        state, client, 8,
        fpu_reduction=1, after_state_evaluate_all_children=exact_chance,
    )
    assert root.effective_merge_symmetric_actions
    # Each two expansions admit one further ordinary draw; hidden cards remain.
    assert min(int(observed.deck.sum()) for observed in client.states) == 3
    assert all(
        observed.table[:, :, :, sj.FINGER_HIDDEN].sum() == 2
        for observed in client.states
    )
    assert root.visit_count == 8


@pytest.mark.parametrize("score_boundary", [False, True])
@pytest.mark.parametrize("first_action", [sj.MASK_TAKE, sj.MASK_DRAW])
def test_grouped_descendants_reach_both_boundary_evaluators(
    monkeypatch, score_boundary, first_action
):
    state = board_state(
        RECYCLING_BOARD, phase=sj.ACTION_DRAW_OR_TAKE,
        countdown=2, opponents_cleared=True,
    )
    client = FixedPredictor(first_action=first_action)
    completed_batches = []
    settings = {}
    if score_boundary:
        monkeypatch.setattr(boundary_inference, "load_boundary_model", lambda *args: None)

        def evaluate(model, states):
            completed_batches.append(states)
            assert len(states) == 3
            assert all(sj.get_round_over(observed) for observed in states)
            return np.tile([0.4, 0.6], (len(states), 1)).astype(np.float32)

        monkeypatch.setattr(boundary_inference, "predict_completed_rounds", evaluate)
        settings = {"boundary_samples": 3, "boundary_value_checkpoint": "frozen.pth"}
    random.seed(33)
    np.random.seed(33)
    root = mcts.run_mcts(state, client, 12, fpu_reduction=1, **settings)
    descendants = [
        node for node in decision_nodes(root) if node is not root and node.is_expanded
    ]
    assert descendants
    boundaries = [
        child
        for node in descendants
        for child in node.children.values()
        if isinstance(child, mcts.RoundBoundaryNode)
    ]
    assert boundaries
    assert all(node.effective_merge_symmetric_actions for node in descendants)
    assert any(len(node.children) < sj.actions(node.state).sum() for node in descendants)
    assert root.visit_count == sum(child.visit_count for child in root.children.values()) == 12
    if score_boundary:
        assert completed_batches
        assert all(child.next_round_state is None for child in boundaries)
    else:
        assert any(child.next_round_state is not None for child in boundaries)


@pytest.mark.parametrize("iterations", [-1, 1.5, True])
def test_invalid_iteration_budget_fails_before_inference(iterations):
    with pytest.raises(ValueError, match="nonnegative integer"):
        mcts.run_mcts(None, None, iterations)
