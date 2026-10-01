from __future__ import annotations

import functools
import random

import numpy as np
import pytest
import torch

from skyjo import checkpoint


def sample_loss(value, *, scale):
    return value * scale


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
    expected_parameters = [parameter.detach().clone() for parameter in model.parameters()]
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


def test_checkpoint_normalizes_partial_callable_settings():
    normalized = checkpoint.normalize_configuration(
        functools.partial(sample_loss, scale=0.25)
    )

    assert normalized == {
        "callable": f"{__name__}.sample_loss",
        "args": [],
        "keywords": {"scale": 0.25},
    }


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


@pytest.mark.parametrize("configuration", [None, {"training_semantics": "round_win_v1"}])
def test_full_game_resume_rejects_unmarked_or_round_checkpoints(tmp_path, configuration):
    from skyjo import buffer

    source = torch.nn.Linear(2, 1)
    path = checkpoint.save_checkpoint(
        tmp_path / "old.pth", model=source, optimizer=None, configuration=configuration,
    )
    destination = torch.nn.Linear(2, 1)
    before = {key: value.clone() for key, value in destination.state_dict().items()}
    with pytest.raises(checkpoint.CheckpointFormatError, match="semantics"):
        checkpoint.load_checkpoint(
            path, model=destination,
            required_training_semantics=buffer.FULL_GAME_TRAINING_SEMANTICS,
        )
    torch.testing.assert_close(destination.state_dict(), before)
