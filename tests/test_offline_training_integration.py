from __future__ import annotations

import numpy as np
import pytest
import torch
import typer
from typer.testing import CliRunner

import run_train_epoch
from skyjo.engine import game as sj
from skyjo.learning import (
    batches,
    buffer,
    checkpoint,
    learner,
    observations,
    replay_io,
    skynet,
)


def make_dataset(path, extra_targets=None):
    extra_targets = extra_targets or {}
    replay_buffer = buffer.ReplayBuffer(
        max_size=32,
        spatial_input_shape=(2, sj.ROW_COUNT, sj.COLUMN_COUNT, sj.FINGER_SIZE),
        non_spatial_input_shape=observations.get_non_spatial_input_shape(2),
        action_mask_shape=(sj.MASK_SIZE,),
        target_specs=(
            *buffer.core_target_specs(2, (sj.MASK_SIZE,)),
            *(
                buffer.TargetShapeSpec(name, value.shape)
                for name, value in extra_targets.items()
            ),
        ),
    )
    for game_index in range(4):
        state = sj.new(players=2, top=game_index)
        action_mask = sj.actions(state).astype(np.float32)
        policy = action_mask / action_mask.sum()
        inputs = batches.states_to_batch([state, state])
        data = batches.TrainingBatch(
            inputs.spatial_inputs,
            inputs.non_spatial_inputs,
            inputs.action_masks,
            {
                "value": np.tile([game_index % 2, (game_index + 1) % 2], (2, 1)),
                "policy": np.stack([policy, policy]),
                **{
                    name: np.stack([value, value])
                    for name, value in extra_targets.items()
                },
            },
        )
        replay_buffer.append(
            data, buffer.GameProvenance(game_index, game_index * 2, game_index * 2 + 1)
        )
    replay_io.save(replay_buffer, path, generation_metadata={"test": True})
    return replay_io.load(path)


def assert_optimizer_states_equal(left, right) -> None:
    left_state = left.state_dict()
    right_state = right.state_dict()
    assert left_state["param_groups"] == right_state["param_groups"]
    assert left_state["state"].keys() == right_state["state"].keys()
    for parameter_id in left_state["state"]:
        for name, expected in left_state["state"][parameter_id].items():
            actual = right_state["state"][parameter_id][name]
            if isinstance(expected, torch.Tensor):
                assert torch.equal(expected, actual)
            else:
                assert expected == actual


def test_loaded_dataset_training_resume_matches_uninterrupted_and_evaluates(tmp_path):
    complete = make_dataset(tmp_path / "dataset")
    training, validation = complete.split_by_game(0.5, seed=3)
    settings = {
        "embedding_dimensions": 4,
        "global_state_embedding_dimensions": 8,
        "num_heads": 1,
    }

    def create(seed):
        return learner.Learner.create(
            settings, players=2, device="cpu", learn_rate=1e-3, seed=seed
        )

    continuous = create(4)
    continuous.fit(training, steps=4, batch_size=2)
    interrupted = create(4)
    interrupted.fit(training, steps=2, batch_size=2)
    progress = checkpoint.TrainingProgress(optimizer_steps=2, sampled_positions=4)
    path = checkpoint.save_checkpoint(
        tmp_path / "resume.pth",
        model=interrupted.model,
        optimizer=interrupted.optimizer,
        configuration=settings,
        progress=progress,
        sampling_rng=interrupted.sampling_rng,
    )
    resumed = create(999)
    restored = checkpoint.load_checkpoint(
        path,
        model=resumed.model,
        optimizer=resumed.optimizer,
        expected_configuration=settings,
        sampling_rng=resumed.sampling_rng,
    )
    assert restored == progress
    resumed.fit(training, steps=2, batch_size=2)
    for expected, actual in zip(
        continuous.model.parameters(), resumed.model.parameters(), strict=True
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert_optimizer_states_equal(continuous.optimizer, resumed.optimizer)
    assert continuous.evaluate(validation, batch_size=2) == resumed.evaluate(
        validation, batch_size=2
    )
    np.testing.assert_array_equal(
        continuous.sampling_rng.integers(100, size=20),
        resumed.sampling_rng.integers(100, size=20),
    )


def test_offline_training_entrypoint_reports_losses_and_resumes(tmp_path):
    make_dataset(tmp_path / "dataset")
    first_checkpoint = tmp_path / "step_1.pth"
    resumed_checkpoint = tmp_path / "step_2.pth"
    app = typer.Typer()
    app.command()(run_train_epoch.main)
    common_arguments = [
        str(tmp_path / "dataset"),
        "--batch-size",
        "2",
        "--validation-fraction",
        "0.5",
        "--embedding-dimensions",
        "4",
        "--global-state-embedding-dimensions",
        "8",
        "--num-heads",
        "1",
    ]

    first = CliRunner().invoke(
        app,
        [
            *common_arguments,
            "--steps",
            "1",
            "--output-checkpoint",
            str(first_checkpoint),
        ],
    )
    assert first.exit_code == 0, first.output
    assert "optimizer_steps: 1" in first.output
    assert "train_total_loss:" in first.output
    assert "validation_total_loss:" in first.output

    resumed = CliRunner().invoke(
        app,
        [
            *common_arguments,
            "--steps",
            "2",
            "--checkpoint",
            str(first_checkpoint),
            "--output-checkpoint",
            str(resumed_checkpoint),
        ],
    )
    assert resumed.exit_code == 0, resumed.output
    assert "optimizer_steps: 2" in resumed.output
    payload = torch.load(resumed_checkpoint, weights_only=False)
    assert (
        payload["configuration"]["model"]["name"]
        == skynet.EQUIVARIANT_ARCHITECTURE_NAME
    )
    assert payload["progress"]["optimizer_steps"] == 2
    assert payload["progress"]["sampled_positions"] == 4


def test_offline_training_selects_games_and_runs_without_validation(tmp_path):
    make_dataset(tmp_path / "dataset")
    step_zero_checkpoint = tmp_path / "step_0.pth"
    resumed_checkpoint = tmp_path / "step_2.pth"
    app = typer.Typer()
    app.command()(run_train_epoch.main)
    common_arguments = [
        str(tmp_path / "dataset"),
        "--game-index",
        "0",
        "--game-index",
        "1",
        "--validation-fraction",
        "0",
        "--batch-size",
        "2",
        "--embedding-dimensions",
        "4",
        "--global-state-embedding-dimensions",
        "8",
        "--num-heads",
        "1",
    ]

    step_zero = CliRunner().invoke(
        app,
        [
            *common_arguments,
            "--steps",
            "0",
            "--output-checkpoint",
            str(step_zero_checkpoint),
        ],
    )
    assert step_zero.exit_code == 0, step_zero.output
    assert "dataset_games: 2" in step_zero.output
    assert "dataset_positions: 4" in step_zero.output
    assert "selected_game_indices: 0,1" in step_zero.output
    assert "training_games: 2" in step_zero.output
    assert "train_total_loss:" in step_zero.output
    assert "validation_total_loss:" not in step_zero.output

    resumed = CliRunner().invoke(
        app,
        [
            *common_arguments,
            "--steps",
            "2",
            "--checkpoint",
            str(step_zero_checkpoint),
            "--output-checkpoint",
            str(resumed_checkpoint),
        ],
    )
    assert resumed.exit_code == 0, resumed.output
    payload = torch.load(resumed_checkpoint, weights_only=False)
    assert payload["configuration"]["dataset"]["game_indices"] == [0, 1]
    assert payload["configuration"]["dataset"]["validation_fraction"] == 0
    assert payload["progress"]["optimizer_steps"] == 2
    assert payload["progress"]["sampled_positions"] == 4

    mismatched_selection = CliRunner().invoke(
        app,
        [
            str(tmp_path / "dataset"),
            "--game-index",
            "0",
            "--game-index",
            "2",
            "--validation-fraction",
            "0",
            "--batch-size",
            "2",
            "--embedding-dimensions",
            "4",
            "--global-state-embedding-dimensions",
            "8",
            "--num-heads",
            "1",
            "--steps",
            "2",
            "--checkpoint",
            str(step_zero_checkpoint),
        ],
    )
    assert mismatched_selection.exit_code != 0
    assert "checkpoint configuration does not match" in mismatched_selection.output


def test_offline_training_rejects_unknown_game_index(tmp_path):
    make_dataset(tmp_path / "dataset")
    app = typer.Typer()
    app.command()(run_train_epoch.main)

    result = CliRunner().invoke(
        app,
        [str(tmp_path / "dataset"), "--game-index", "99", "--validation-fraction", "0"],
    )

    assert result.exit_code != 0
    assert "unknown --game-index value(s): 99" in result.output


@pytest.mark.parametrize("auxiliary", [None, '{"round_score": 0.1}'])
def test_offline_objectives_are_explicit_for_available_labels(tmp_path, auxiliary):
    make_dataset(
        tmp_path / "dataset",
        extra_targets={
            "round_score": np.array([0.2, 0.4], dtype=np.float32),
            "round_raw_score": np.zeros((2,), dtype=np.float32),
        },
    )
    saved = tmp_path / "trained.pth"
    app = typer.Typer()
    app.command()(run_train_epoch.main)
    arguments = [
        str(tmp_path / "dataset"),
        "--steps",
        "1",
        "--batch-size",
        "2",
        "--validation-fraction",
        "0",
        "--embedding-dimensions",
        "4",
        "--global-state-embedding-dimensions",
        "8",
        "--num-heads",
        "1",
        "--output-checkpoint",
        str(saved),
    ]
    if auxiliary is not None:
        arguments.extend(["--auxiliary-objectives", auxiliary])
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 0, result.output
    assert "train_total_loss:" in result.output
    assert ("train_round_score_loss:" in result.output) == (auxiliary is not None)
    payload = torch.load(saved, weights_only=False)
    configuration = payload["configuration"]["model"]
    assert configuration["name"] == skynet.EQUIVARIANT_ARCHITECTURE_NAME
    assert payload["configuration"]["auxiliary_objectives"] == (
        {} if auxiliary is None else {"round_score": 0.1}
    )


def test_offline_enabled_objective_requires_matching_labels(tmp_path):
    make_dataset(tmp_path / "dataset")
    app = typer.Typer()
    app.command()(run_train_epoch.main)
    result = CliRunner().invoke(
        app,
        [
            str(tmp_path / "dataset"),
            "--auxiliary-objectives",
            '{"round_score": 0.1}',
        ],
    )
    assert result.exit_code != 0
    assert "Missing required auxiliary" in result.output
    assert "round_score" in result.output
