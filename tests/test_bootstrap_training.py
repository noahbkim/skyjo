"""Weights-only initialization must not inherit stale training state or targets."""

import dataclasses
import hashlib

import pytest
import torch

from skyjo.experiments import experiment_config, training_setup
from skyjo.experiments.settings import SelfPlayRunConfig
from skyjo.learning import checkpoint
from skyjo.learning.learner import Learner


def test_bootstrap_copies_legacy_weights_and_starts_fresh(tmp_path):
    model = {
        "embedding_dimensions": 4,
        "global_state_embedding_dimensions": 8,
        "num_heads": 1,
        "policy_normalization": "actions",
    }
    source = Learner.create(model, players=2, device="cpu", learn_rate=0.001, seed=0)
    for parameter in source.model.parameters():
        parameter.grad = torch.ones_like(parameter)
    source.optimizer.step()
    path = tmp_path / "early.pth"
    checkpoint.save_checkpoint(
        path,
        model=source.model,
        optimizer=source.optimizer,
        configuration={"model": model, "players": 2},
        progress=checkpoint.TrainingProgress(iteration=19, optimizer_steps=100),
        sampling_rng=source.sampling_rng,
    )
    payload = checkpoint.decode_checkpoint(path)
    payload["configuration"]["model"].pop("policy_normalization")
    payload["configuration"]["model"].update(
        non_spatial_input_shape=[69], auxiliary_objectives={}
    )
    torch.save(payload, path)
    config = experiment_config.resolve_configuration(
        {
            "seed": 7,
            "bootstrap_checkpoint": str(path),
            "model": model | {"policy_normalization": "groups"},
            "replay": {"capacity": 16},
        },
        base_directory=tmp_path,
    )
    settings = SelfPlayRunConfig.from_resolved(config)
    session = training_setup.prepare_training(
        settings, tmp_path / "checkpoints", parent=None, run_id=None
    )
    for name, expected in source.model.state_dict().items():
        torch.testing.assert_close(
            session.learner.model.state_dict()[name], expected, rtol=0, atol=0
        )
    assert session.learner.model.policy_normalization == "groups"
    assert not session.learner.optimizer.state
    assert len(session.replay) == 0
    assert (
        session.state.progress.iteration == session.state.progress.optimizer_steps == 0
    )
    assert session.state.next_game_index == 0
    provenance = session.checkpoint_configuration["bootstrap"]
    assert provenance["source_progress"]["iteration"] == 19
    assert provenance["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    fresh = Learner.create(model, players=2, device="cpu", learn_rate=0.001, seed=7)
    assert (
        session.learner.sampling_rng.bit_generator.state
        == fresh.sampling_rng.bit_generator.state
    )

    incompatible = dataclasses.replace(
        settings, model=dataclasses.replace(settings.model, embedding_dimensions=8)
    )
    with pytest.raises(ValueError, match="architecture"):
        training_setup.prepare_training(
            incompatible, tmp_path / "bad", parent=None, run_id=None
        )


@pytest.mark.parametrize(
    "other",
    [{"initial_checkpoint": "old.pth"}, {"replay": {"initial_dataset": "old-replay"}}],
)
def test_bootstrap_rejects_old_training_state_before_launch(tmp_path, other):
    with pytest.raises(ValueError, match="fresh replay"):
        experiment_config.resolve_configuration(
            {"bootstrap_checkpoint": "early.pth", **other}, base_directory=tmp_path
        )
