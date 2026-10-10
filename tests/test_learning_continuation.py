"""Continuation retains learning state while allowing explicit optimizer changes."""

import copy

import numpy as np
import torch

from skyjo.engine import game
from skyjo.learning import batches, buffer, checkpoint, continuation, learner, models

SETTINGS = {
    "embedding_dimensions": 4,
    "global_state_embedding_dimensions": 8,
    "num_heads": 1,
}


def make_learner(seed=4, rate=0.001):
    return learner.Learner.create(
        SETTINGS, players=2, device="cpu", seed=seed, learn_rate=rate
    )


def replay_data():
    states = [game.new(players=2, top=i) for i in range(4)]
    inputs = batches.states_to_batch(states)
    data = batches.TrainingBatch(
        inputs.spatial_inputs,
        inputs.non_spatial_inputs,
        inputs.action_masks,
        {
            "value": np.array([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=np.float32),
            "policy": inputs.action_masks / inputs.action_masks.sum(-1, keepdims=True),
        },
    )
    replay = buffer.ReplayBuffer(
        4,
        inputs.spatial_inputs.shape[1:],
        inputs.non_spatial_inputs.shape[1:],
        inputs.action_masks.shape[1:],
    )
    replay.append(data, buffer.GameProvenance(0))
    return replay


def test_continuation_preserves_moments_sampler_and_overrides_only_settings(tmp_path):
    replay = replay_data()
    parent = make_learner()
    parent.fit(replay, steps=2, batch_size=2)
    configuration = {
        "seed": 4,
        "players": 2,
        "model": models.resolve(SETTINGS),
        "auxiliary_objectives": {},
        "optimizer": {"type": "adam"},
        "run_id": "parent",
    }
    path = checkpoint.save_checkpoint(
        tmp_path / "parent.pth",
        model=parent.model,
        optimizer=parent.optimizer,
        configuration=configuration,
        progress=checkpoint.TrainingProgress(optimizer_steps=2),
        sampling_rng=parent.sampling_rng,
        continuation_state={"generated_positions": 4, "next_game_index": 1},
    )
    source = continuation.load(path, configuration)
    assert source.provenance["run_id"] == "parent"
    assert source.next_game(replay) == 1
    assert source.generated_positions == 4
    inherited_optimizer = copy.deepcopy(parent.optimizer.state_dict())

    child = make_learner(seed=99, rate=0.01)
    source.restore(child.model, child.optimizer, sampling_rng=child.sampling_rng)
    assert child.optimizer.param_groups[0]["lr"] == 0.01
    for index, state in child.optimizer.state_dict()["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(
                value, inherited_optimizer["state"][index][name], rtol=0, atol=0
            )
    # With identical next-run settings, inherited state yields identical updates.
    parent.optimizer.param_groups[0]["lr"] = 0.01
    parent.fit(replay, steps=2, batch_size=2)
    child.fit(replay, steps=2, batch_size=2)
    for expected, actual in zip(
        parent.model.parameters(), child.model.parameters(), strict=True
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    np.testing.assert_array_equal(
        parent.sampling_rng.integers(1000, size=20),
        child.sampling_rng.integers(1000, size=20),
    )
