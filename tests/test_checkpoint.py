from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from skyjo.learning import checkpoint


def make_training_objects():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    loss = model(torch.ones(1, 2)).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    return model, optimizer, scheduler


def test_checkpoint_round_trip_restores_training_and_rng_state(tmp_path) -> None:
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    model, optimizer, scheduler = make_training_objects()
    expected_parameters = [
        parameter.detach().clone() for parameter in model.parameters()
    ]
    configuration = {"model": {"width": 2}, "training": {"learn_rate": 0.01}}
    progress = checkpoint.TrainingProgress(3, 2, 50, 800)
    path = tmp_path / "checkpoint_test.pth"
    checkpoint.save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        configuration=configuration,
        progress=progress,
    )
    expected_random = (random.random(), np.random.random(), torch.rand(1))

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)

    restored = checkpoint.load_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        expected_configuration=configuration,
    )

    assert restored == progress
    for actual, expected in zip(model.parameters(), expected_parameters, strict=True):
        assert torch.equal(actual, expected)
    assert scheduler.last_epoch == 1
    assert random.random() == expected_random[0]
    assert np.random.random() == expected_random[1]
    assert torch.equal(torch.rand(1), expected_random[2])


def test_checkpoint_rejects_raw_state_dict(tmp_path) -> None:
    model = torch.nn.Linear(2, 1)
    path = tmp_path / "legacy.pth"
    torch.save(model.state_dict(), path)
    with pytest.raises(checkpoint.CheckpointFormatError, match="raw state_dict"):
        checkpoint.load_checkpoint(path, model=model)


def test_checkpoint_validates_configuration(tmp_path) -> None:
    model, optimizer, _ = make_training_objects()
    path = tmp_path / "checkpoint_test.pth"
    checkpoint.save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        configuration={"players": 2},
    )
    with pytest.raises(ValueError, match="configuration"):
        checkpoint.load_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            expected_configuration={"players": 3},
        )


def test_resumed_training_matches_uninterrupted_training(tmp_path) -> None:
    torch.manual_seed(12)
    continuous = torch.nn.Linear(2, 1)
    interrupted = torch.nn.Linear(2, 1)
    interrupted.load_state_dict(continuous.state_dict())
    continuous_optimizer = torch.optim.Adam(continuous.parameters(), lr=0.01)
    interrupted_optimizer = torch.optim.Adam(interrupted.parameters(), lr=0.01)

    def step(model, optimizer) -> None:
        optimizer.zero_grad()
        model(torch.tensor([[1.0, -1.0]])).square().sum().backward()
        optimizer.step()

    step(continuous, continuous_optimizer)
    step(continuous, continuous_optimizer)
    step(interrupted, interrupted_optimizer)
    path = tmp_path / "resume.pth"
    checkpoint.save_checkpoint(
        path,
        model=interrupted,
        optimizer=interrupted_optimizer,
        configuration={"run": "test"},
        progress=checkpoint.TrainingProgress(iteration=1),
    )

    resumed = torch.nn.Linear(2, 1)
    resumed_optimizer = torch.optim.Adam(resumed.parameters(), lr=0.01)
    checkpoint.load_checkpoint(
        path,
        model=resumed,
        optimizer=resumed_optimizer,
        expected_configuration={"run": "test"},
    )
    step(resumed, resumed_optimizer)

    for expected, actual in zip(
        continuous.parameters(), resumed.parameters(), strict=True
    ):
        assert torch.equal(expected, actual)


@pytest.mark.parametrize(
    "missing",
    [
        "configuration",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "rng_state",
        "sampling_rng_state",
        "progress",
    ],
)
def test_incomplete_resume_does_not_mutate_runtime(tmp_path, missing):
    model, optimizer, scheduler = make_training_objects()
    sampling_rng = np.random.default_rng(10)
    path = tmp_path / "incomplete.pth"
    checkpoint.save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        configuration={"players": 2},
        sampling_rng=sampling_rng,
    )
    payload = torch.load(path, weights_only=False)
    payload.pop(missing)
    torch.save(payload, path)
    with torch.no_grad():
        model.weight.add_(10)
    before = model.weight.detach().clone()
    rng_before = torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        checkpoint.load_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_configuration={"players": 2},
            sampling_rng=sampling_rng,
        )
    assert torch.equal(model.weight, before)
    assert torch.equal(torch.get_rng_state(), rng_before)


def test_weights_only_checkpoint_loading(tmp_path):
    model = torch.nn.Linear(2, 1)
    path = tmp_path / "weights.pth"
    checkpoint.save_checkpoint(path, model=model, optimizer=None)
    restored = torch.nn.Linear(2, 1)
    checkpoint.load_checkpoint(path, model=restored, restore_rng=False)
    assert torch.equal(restored.weight, model.weight)


@pytest.mark.parametrize(
    "field,invalid",
    [
        ("progress", {"optimizer_steps": -1}),
        ("rng_state", {}),
        ("optimizer_state_dict", {}),
        ("scheduler_state_dict", {}),
    ],
)
def test_malformed_resume_metadata_is_rejected_before_loading_weights(
    tmp_path, field, invalid
):
    model, optimizer, scheduler = make_training_objects()
    path = tmp_path / "malformed.pth"
    checkpoint.save_checkpoint(
        path, model=model, optimizer=optimizer, scheduler=scheduler
    )
    payload = torch.load(path, weights_only=False)
    payload[field] = invalid
    torch.save(payload, path)
    with torch.no_grad():
        model.weight.add_(10)
    before = model.weight.detach().clone()
    with pytest.raises(checkpoint.CheckpointFormatError):
        checkpoint.load_checkpoint(
            path, model=model, optimizer=optimizer, scheduler=scheduler
        )
    assert torch.equal(model.weight, before)


def test_mismatched_model_shape_is_rejected_before_any_parameter_changes(tmp_path):
    model = torch.nn.Linear(2, 2)
    path = checkpoint.save_checkpoint(
        tmp_path / "model.pth", model=model, optimizer=None
    )
    payload = checkpoint.decode_checkpoint(path)
    payload["model_state_dict"]["weight"] = torch.ones_like(model.weight) * 99
    payload["model_state_dict"]["bias"] = torch.zeros(3)
    before = model.weight.detach().clone()
    with pytest.raises(checkpoint.CheckpointFormatError, match="model parameters"):
        checkpoint.restore_checkpoint(payload, model=model, restore_rng=False)
    torch.testing.assert_close(model.weight, before, rtol=0, atol=0)
