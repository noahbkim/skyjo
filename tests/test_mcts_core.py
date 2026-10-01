from __future__ import annotations

import random
import types

import numpy as np
import pytest
import torch

from skyjo import game as sj
from skyjo import mcts, parallel_mcts, predictor, skynet


def make_model() -> skynet.SimpleSkyNet:
    torch.manual_seed(0)
    return skynet.SimpleSkyNet(
        spatial_input_shape=(2, sj.ROW_COUNT, sj.COLUMN_COUNT, sj.FINGER_SIZE),
        non_spatial_input_shape=skynet.get_non_spatial_input_shape(2),
        value_output_shape=(2,),
        policy_output_shape=(sj.MASK_SIZE,),
        hidden_layers=[8],
        device=torch.device("cpu"),
    )


def run(search, *, batched_leaf_count: int | None = None):
    random.seed(10)
    np.random.seed(10)
    state = sj.start_round(sj.new(players=2, top=sj.CARD_0), rng=random)
    client = predictor.LocalPredictorClient(make_model(), max_batch_size=64)
    kwargs = {}
    if batched_leaf_count is not None:
        kwargs.update(batched_leaf_count=batched_leaf_count, virtual_loss=0.5)
    return search(
        state,
        client,
        iterations=8,
        **kwargs,
    )


def test_batch_size_one_matches_sequential_search_exactly() -> None:
    sequential = run(mcts.run_mcts)
    batched = run(parallel_mcts.run_mcts, batched_leaf_count=1)
    assert sequential.visit_count == batched.visit_count
    assert np.allclose(sequential.state_value, batched.state_value)
    assert np.array_equal(
        sequential.policy_targets(),
        batched.policy_targets(),
    )
    assert {
        action: child.visit_count for action, child in sequential.children.items()
    } == {action: child.visit_count for action, child in batched.children.items()}


def test_larger_batch_hint_preserves_search_accounting() -> None:
    root = run(parallel_mcts.run_mcts, batched_leaf_count=4)
    policy = root.policy_targets()
    assert root.visit_count == 8
    assert np.isclose(policy.sum(), 1.0)
    assert np.all(policy[sj.actions(root.state) == 0] == 0)
    for child in root.children.values():
        if hasattr(child, "virtual_loss_total"):
            assert child.virtual_loss_total == 0
        else:
            assert child.virtual_loss == 0


def test_larger_batch_changes_evaluation_scheduling() -> None:
    class RecordingClient(predictor.LocalPredictorClient):
        def __init__(self, model):
            super().__init__(model, max_batch_size=64)
            self.sent_batch_sizes = []

        def send(self) -> None:
            self.sent_batch_sizes.append(len(self.current_inputs))
            super().send()

    random.seed(10)
    np.random.seed(10)
    state = sj.start_round(sj.new(players=2, top=sj.CARD_0), rng=random)
    client = RecordingClient(make_model())
    parallel_mcts.run_mcts(
        state,
        client,
        iterations=8,
        batched_leaf_count=4,
        virtual_loss=0.5,
    )
    assert max(client.sent_batch_sizes) > 1


def make_prediction(
    value: tuple[float, float], policy: dict[int, float]
) -> skynet.SkyNetPrediction:
    probabilities = np.zeros(sj.MASK_SIZE, dtype=np.float32)
    for action, probability in policy.items():
        probabilities[action] = probability
    return skynet.SkyNetPrediction(
        value_output=np.asarray(value, dtype=np.float32),
        policy_output=probabilities,
    )


def test_ucb_uses_parent_value_fpu_and_prior_on_first_selection() -> None:
    state = sj.start_round(sj.new(players=2, top=sj.CARD_0), rng=random)
    parent = mcts.DecisionStateNode(
        state=state,
        parent=None,
        action=None,
        c_puct=1.5,
        fpu_reduction=0.1,
    )
    parent.model_prediction = make_prediction(
        (0.7, 0.3), {sj.MASK_DRAW: 0.8, sj.MASK_TAKE: 0.2}
    )
    parent.is_expanded = True
    draw = types.SimpleNamespace(
        action=sj.MASK_DRAW,
        has_value_estimate=False,
        visit_count=0,
        virtual_loss=0.0,
    )
    take = types.SimpleNamespace(
        action=sj.MASK_TAKE,
        has_value_estimate=False,
        visit_count=0,
        virtual_loss=0.0,
    )
    parent.children = {sj.MASK_DRAW: draw, sj.MASK_TAKE: take}

    assert mcts.ucb_score(draw, parent) == pytest.approx(1.8)
    assert mcts.ucb_score(take, parent) == pytest.approx(0.9)
    assert parent.select_child() is draw


def test_exact_chance_update_replaces_propagated_return() -> None:
    state = sj.start_round(sj.new(players=2, top=sj.CARD_0), rng=random)
    root = mcts.DecisionStateNode(state=state, parent=None, action=None)
    root.model_prediction = make_prediction((0.5, 0.5), {sj.MASK_DRAW: 1.0})
    chance = mcts.AfterStateNode(state=state, action=sj.MASK_DRAW, parent=root)
    child_state = sj.draw(sj.preordain(state, sj.CARD_P5))
    child = mcts.DecisionStateNode(
        state=child_state,
        parent=chance,
        action=sj.MASK_DRAW,
    )
    child.model_prediction = make_prediction((0.6, 0.4), {})
    child.is_expanded = True
    child_hash = sj.hash_skyjo(child_state)
    chance.children = {child_hash: child}
    chance.child_weights = {child_hash: 0.25}
    chance.child_weight_total = 1.0
    chance.state_value_total = np.array([0.45, 0.55], dtype=np.float32)
    chance.all_children_discovered = True
    chance.is_expanded = True

    mcts.backpropagate(
        [root, chance, child], np.array([0.8, 0.2], dtype=np.float32)
    )

    assert np.allclose(child.state_value, [0.8, 0.2])
    assert np.allclose(chance.state_value, [0.5, 0.5])
    assert np.allclose(root.state_value, [0.5, 0.5])


def test_reused_root_requires_matching_scoring_and_resets_noise() -> None:
    state = sj.start_round(sj.new(players=2, top=sj.CARD_0), rng=random)
    client = predictor.LocalPredictorClient(make_model(), max_batch_size=64)
    root = mcts.run_mcts(
        state,
        client,
        iterations=0,
        dirichlet_epsilon=0.25,
        c_puct=1.5,
        fpu_reduction=0.0,
    )
    assert root.dirichlet_noise.sum() == pytest.approx(1.0)

    reused = mcts.run_mcts(
        state,
        client,
        iterations=0,
        dirichlet_epsilon=0.0,
        c_puct=1.5,
        fpu_reduction=0.0,
        root_node=root,
    )
    assert reused is root
    assert root.dirichlet_epsilon == 0.0
    assert not root.dirichlet_noise.any()

    with pytest.raises(ValueError, match="scoring configuration"):
        mcts.run_mcts(
            state,
            client,
            iterations=0,
            c_puct=1.0,
            root_node=root,
        )
