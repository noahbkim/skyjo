from __future__ import annotations

import random

import numpy as np
import torch

from skyjo import game as sj
from skyjo import mcts, parallel_mcts, predictor, skynet


def make_model() -> skynet.SimpleSkyNet:
    torch.manual_seed(0)
    return skynet.SimpleSkyNet(
        spatial_input_shape=(2, sj.ROW_COUNT, sj.COLUMN_COUNT, sj.FINGER_SIZE),
        non_spatial_input_shape=(sj.GAME_SIZE,),
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
        terminal_state_initial_rollouts=2,
        **kwargs,
    )


def test_batched_adapter_uses_shared_node_core() -> None:
    assert parallel_mcts.DecisionStateNode is mcts.DecisionStateNode
    assert parallel_mcts.AfterStateNode is mcts.AfterStateNode


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
