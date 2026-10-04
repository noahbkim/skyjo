from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch
import typer
from typer.testing import CliRunner

import run_train_epoch  # noqa: E402
from skyjo import batches, buffer, checkpoint, losses, observations, play, skynet, train
from skyjo import game as sj


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
        game_data = [
            play.GameDataPoint(
                state,
                None,
                {
                    batches.VALUE_TARGET_NAME: np.array(
                        [game_index % 2, (game_index + 1) % 2],
                        dtype=np.float32,
                    ),
                    batches.POLICY_TARGET_NAME: policy,
                    **extra_targets,
                },
            )
            for _ in range(2)
        ]
        replay_buffer.add_game_data(
            game_data,
            game_index=game_index,
            play_seed=game_index * 2,
            target_seed=game_index * 2 + 1,
        )
    replay_buffer.save(path, generation_metadata={"test": True})
    return buffer.ReplayBuffer.load(path)


def make_model() -> skynet.EquivariantSkyNet:
    return skynet.EquivariantSkyNet(
        spatial_input_shape=(
            2,
            sj.ROW_COUNT,
            sj.COLUMN_COUNT,
            sj.FINGER_SIZE,
        ),
        non_spatial_input_shape=observations.get_non_spatial_input_shape(2),
        value_output_shape=(2,),
        policy_output_shape=(sj.MASK_SIZE,),
        device=torch.device("cpu"),
        embedding_dimensions=4,
        global_state_embedding_dimensions=8,
        num_heads=1,
    )


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


def test_loaded_dataset_training_resume_matches_uninterrupted_and_evaluates(
    tmp_path,
):
    complete = make_dataset(tmp_path / "dataset")
    training_buffer, validation_buffer = complete.split_by_game(0.5, seed=3)
    configuration = {
        "model": {"name": "equivariant", "width": 4},
        "training": {"batch_size": 2, "loss": "base"},
        "dataset": {"dataset_id": complete.dataset_id, "split_seed": 3},
    }
    loss_function = losses.base_loss

    torch.manual_seed(4)
    initial_model = make_model()
    initial_state = {
        name: value.detach().clone()
        for name, value in initial_model.state_dict().items()
    }

    continuous = make_model()
    continuous.load_state_dict(initial_state)
    continuous_optimizer = train.make_optimizer(continuous, 1e-3)
    np.random.seed(19)
    train.train_steps(
        continuous,
        training_buffer,
        training_batch_size=2,
        optimizer_steps=4,
        optimizer=continuous_optimizer,
        loss_function=loss_function,
    )

    interrupted = make_model()
    interrupted.load_state_dict(initial_state)
    interrupted_optimizer = train.make_optimizer(interrupted, 1e-3)
    np.random.seed(19)
    train.train_steps(
        interrupted,
        training_buffer,
        training_batch_size=2,
        optimizer_steps=2,
        optimizer=interrupted_optimizer,
        loss_function=loss_function,
    )
    saved_progress = checkpoint.TrainingProgress(
        optimizer_steps=2,
        sampled_positions=4,
        trained_positions=4,
    )
    checkpoint_path = tmp_path / "resume.pth"
    checkpoint.save_checkpoint(
        checkpoint_path,
        model=interrupted,
        optimizer=interrupted_optimizer,
        configuration=configuration,
        progress=saved_progress,
    )

    resumed = make_model()
    resumed_optimizer = train.make_optimizer(resumed, 1e-3)
    restored_progress = checkpoint.load_checkpoint(
        checkpoint_path,
        model=resumed,
        optimizer=resumed_optimizer,
        expected_configuration=configuration,
    )
    assert restored_progress == saved_progress
    train.train_steps(
        resumed,
        training_buffer,
        training_batch_size=2,
        optimizer_steps=2,
        optimizer=resumed_optimizer,
        loss_function=loss_function,
    )
    final_progress = dataclasses.replace(
        restored_progress,
        optimizer_steps=4,
        sampled_positions=8,
        trained_positions=8,
    )

    for expected, actual in zip(
        continuous.parameters(), resumed.parameters(), strict=True
    ):
        assert torch.equal(expected, actual)
    assert_optimizer_states_equal(continuous_optimizer, resumed_optimizer)
    assert final_progress.optimizer_steps == 4
    assert final_progress.sampled_positions == 8

    continuous_loss = train.evaluate_loss(
        continuous,
        validation_buffer,
        evaluation_batch_size=2,
        loss_function=loss_function,
    )
    resumed_loss = train.evaluate_loss(
        resumed,
        validation_buffer,
        evaluation_batch_size=2,
        loss_function=loss_function,
    )
    assert continuous_loss == resumed_loss
    assert set(resumed_loss) == {
        "total_loss",
        "outcome_value_loss",
        "policy_loss",
    }


def test_offline_training_entrypoint_reports_losses_and_resumes(tmp_path):
    dataset = make_dataset(tmp_path / "dataset")
    first_checkpoint = tmp_path / "step_1.pth"
    resumed_checkpoint = tmp_path / "step_2.pth"
    app = typer.Typer()
    app.command()(run_train_epoch.main)
    common_arguments = [
        str(dataset.path),
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
    dataset = make_dataset(tmp_path / "dataset")
    step_zero_checkpoint = tmp_path / "step_0.pth"
    resumed_checkpoint = tmp_path / "step_2.pth"
    app = typer.Typer()
    app.command()(run_train_epoch.main)
    common_arguments = [
        str(dataset.path),
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
            str(dataset.path),
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
    assert isinstance(mismatched_selection.exception, ValueError)
    assert "checkpoint configuration does not match" in str(
        mismatched_selection.exception
    )


def test_offline_training_rejects_unknown_game_index(tmp_path):
    dataset = make_dataset(tmp_path / "dataset")
    app = typer.Typer()
    app.command()(run_train_epoch.main)

    result = CliRunner().invoke(
        app,
        [str(dataset.path), "--game-index", "99", "--validation-fraction", "0"],
    )

    assert result.exit_code != 0
    assert "unknown --game-index value(s): 99" in result.output


@pytest.mark.parametrize("auxiliary", [None, '{"round_score": 0.1}'])
def test_offline_objectives_are_explicit_even_with_legacy_extra_labels(
    tmp_path, auxiliary
):
    dataset = make_dataset(
        tmp_path / "dataset",
        extra_targets={
            "round_score": np.array([0.2, 0.4], dtype=np.float32),
            # Archived datasets may contain labels from retired objectives.
            "future_clear": np.zeros((2, sj.COLUMN_COUNT), dtype=np.float32),
        },
    )
    saved = tmp_path / "trained.pth"
    app = typer.Typer()
    app.command()(run_train_epoch.main)
    arguments = [
        str(dataset.path),
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
    assert configuration["auxiliary_objectives"] == (
        {} if auxiliary is None else {"round_score": 0.1}
    )


def test_offline_enabled_objective_requires_matching_labels(tmp_path):
    dataset = make_dataset(tmp_path / "dataset")
    app = typer.Typer()
    app.command()(run_train_epoch.main)
    result = CliRunner().invoke(
        app,
        [
            str(dataset.path),
            "--auxiliary-objectives",
            '{"round_score": 0.1}',
        ],
    )
    assert result.exit_code != 0
    assert "Missing required auxiliary" in result.output
    assert "round_score" in result.output
