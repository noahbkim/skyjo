"""Train and validate a Skyjo model from a saved replay dataset.

This remains a narrow offline entrypoint: it loads one dataset, performs an
exact cumulative optimizer-step budget, optionally saves a resumable
checkpoint, and reports essential loss and timing diagnostics.
"""

from __future__ import annotations

import dataclasses
import functools
import pathlib
import random
import time

import numpy as np
import torch
import typer

from skyjo import buffer, checkpoint, skynet, train, train_utils

DEFAULT_SEED = 0
DEFAULT_BATCH_SIZE = 256
DEFAULT_OPTIMIZER_STEPS = 1
DEFAULT_LEARN_RATE = 1e-3
DEFAULT_EMBEDDING_DIMENSIONS = 32
DEFAULT_GLOBAL_STATE_EMBEDDING_DIMENSIONS = 64
DEFAULT_NUM_HEADS = 2
DEFAULT_VALIDATION_FRACTION = 0.1


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(
    training_data_buffer: buffer.ReplayBuffer,
    device: torch.device,
    embedding_dimensions: int,
    global_state_embedding_dimensions: int,
    num_heads: int,
) -> skynet.SkyNet:
    value_targets = training_data_buffer.target_buffers[
        train_utils.VALUE_TARGET_NAME
    ]
    policy_targets = training_data_buffer.target_buffers[
        train_utils.POLICY_TARGET_NAME
    ]
    model_class = (
        skynet.EquivariantSkyNetWithAuxiliaryHeads
        if train_utils.FUTURE_CLEAR_TARGET_NAME
        in training_data_buffer.target_buffers
        else skynet.EquivariantSkyNet
    )
    return model_class(
        spatial_input_shape=training_data_buffer.spatial_input_buffer.shape[1:],
        non_spatial_input_shape=training_data_buffer.non_spatial_input_buffer.shape[
            1:
        ],
        value_output_shape=value_targets.shape[1:],
        policy_output_shape=policy_targets.shape[1:],
        device=device,
        embedding_dimensions=embedding_dimensions,
        global_state_embedding_dimensions=global_state_embedding_dimensions,
        num_heads=num_heads,
    )


def _print_loss(prefix: str, loss_details: train_utils.LossDetails) -> None:
    for name, value in sorted(loss_details.items()):
        typer.echo(f"{prefix}_{name}: {value:.8f}")


def _select_game_indices(
    replay_buffer: buffer.ReplayBuffer,
    game_indices: list[int] | None,
) -> buffer.ReplayBuffer:
    """Select games by their persisted IDs, preserving dataset order."""
    if not game_indices:
        return replay_buffer
    if len(set(game_indices)) != len(game_indices):
        raise typer.BadParameter("--game-index values must be unique")

    requested = set(game_indices)
    missing = requested.difference(replay_buffer.game_indices)
    if missing:
        missing_text = ", ".join(str(game_index) for game_index in sorted(missing))
        raise typer.BadParameter(f"unknown --game-index value(s): {missing_text}")
    selected_positions = [
        position
        for position, game_index in enumerate(replay_buffer.game_indices)
        if game_index in requested
    ]
    return replay_buffer.select_games(selected_positions)


def main(
    dataset_path: pathlib.Path = typer.Argument(
        ...,
        help="Path to a versioned Skyjo replay dataset directory.",
    ),
    resume_checkpoint: pathlib.Path | None = typer.Option(
        None,
        "--checkpoint",
        help="Optional versioned checkpoint to resume before training.",
    ),
    output_checkpoint: pathlib.Path | None = typer.Option(
        None,
        "--output-checkpoint",
        help="Optional path for the resumable post-training checkpoint.",
    ),
    optimizer_steps: int = typer.Option(
        DEFAULT_OPTIMIZER_STEPS,
        "--steps",
        min=0,
        help="Cumulative optimizer-step target, including resumed steps.",
    ),
    seed: int = typer.Option(
        DEFAULT_SEED,
        "--seed",
        help="Seed for model initialization and dataset sampling.",
    ),
    split_seed: int = typer.Option(
        DEFAULT_SEED,
        "--split-seed",
        help="Seed used for the deterministic game-level validation split.",
    ),
    validation_fraction: float = typer.Option(
        DEFAULT_VALIDATION_FRACTION,
        "--validation-fraction",
        min=0.0,
        max=1.0,
        help="Fraction of complete games reserved for validation; zero disables validation.",
    ),
    game_indices: list[int] | None = typer.Option(
        None,
        "--game-index",
        help="Persisted game ID to include; repeat to select multiple games.",
    ),
    device_name: str = typer.Option(
        "cpu",
        "--device",
        help="Torch device, for example cpu, cuda, or mps.",
    ),
    batch_size: int = typer.Option(
        DEFAULT_BATCH_SIZE,
        "--batch-size",
        min=1,
        help="Training and evaluation batch size.",
    ),
    learn_rate: float = typer.Option(
        DEFAULT_LEARN_RATE,
        "--learn-rate",
        help="Adam learning rate.",
    ),
    embedding_dimensions: int = typer.Option(
        DEFAULT_EMBEDDING_DIMENSIONS,
        "--embedding-dimensions",
        help="EquivariantSkyNet embedding_dimensions.",
    ),
    global_state_embedding_dimensions: int = typer.Option(
        DEFAULT_GLOBAL_STATE_EMBEDDING_DIMENSIONS,
        "--global-state-embedding-dimensions",
        help="EquivariantSkyNet global_state_embedding_dimensions.",
    ),
    num_heads: int = typer.Option(
        DEFAULT_NUM_HEADS,
        "--num-heads",
        help="EquivariantSkyNet num_heads.",
    ),
    value_scale: float = typer.Option(
        1.0,
        "--value-scale",
        help="Scale for the value loss term.",
    ),
    policy_scale: float = typer.Option(
        1.0,
        "--policy-scale",
        help="Scale for the policy loss term.",
    ),
    experiment_arm: str = typer.Option(
        "control",
        "--experiment-arm",
        help="Auxiliary experiment preset: control, score, clear, or combined.",
    ),
) -> None:
    """Train for an exact cumulative step budget and report evaluation loss."""
    device = torch.device(device_name)
    set_seed(seed)

    load_start = time.perf_counter()
    complete_buffer = buffer.ReplayBuffer.load(dataset_path)
    try:
        experiment = train_utils.get_auxiliary_experiment_preset(experiment_arm)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="--experiment-arm") from error
    has_auxiliary_targets = (
        train_utils.ROUND_SCORE_TARGET_NAME in complete_buffer.target_buffers
        and train_utils.FUTURE_CLEAR_TARGET_NAME in complete_buffer.target_buffers
    )
    if (
        experiment.round_score_scale > 0 or experiment.future_clear_scale > 0
    ) and not has_auxiliary_targets:
        raise typer.BadParameter(
            "selected experiment arm requires round_score and future_clear targets",
            param_hint="--experiment-arm",
        )
    complete_buffer = _select_game_indices(complete_buffer, game_indices)
    if validation_fraction == 0.0:
        training_buffer = complete_buffer
        validation_buffer = None
    else:
        training_buffer, validation_buffer = complete_buffer.split_by_game(
            validation_fraction,
            seed=split_seed,
        )
    load_seconds = time.perf_counter() - load_start

    model = build_model(
        complete_buffer,
        device=device,
        embedding_dimensions=embedding_dimensions,
        global_state_embedding_dimensions=global_state_embedding_dimensions,
        num_heads=num_heads,
    )
    optimizer = train.make_optimizer(model, learn_rate)
    if has_auxiliary_targets:
        loss_function = functools.partial(
            train_utils.outcome_policy_auxiliary_loss,
            value_scale=value_scale,
            policy_scale=policy_scale,
            round_score_scale=experiment.round_score_scale,
            future_clear_scale=experiment.future_clear_scale,
            clear_positive_weight=experiment.clear_positive_weight,
        )
    else:
        loss_function = functools.partial(
            train_utils.base_loss,
            value_scale=value_scale,
            policy_scale=policy_scale,
        )
    dataset_configuration = {
        "dataset_id": complete_buffer.dataset_id,
        "validation_fraction": validation_fraction,
        "split_seed": split_seed,
    }
    if game_indices:
        dataset_configuration["game_indices"] = list(complete_buffer.game_indices)
    resume_configuration = {
        "model": {
            "name": getattr(model, "architecture_name", type(model).__name__),
            "embedding_dimensions": embedding_dimensions,
            "global_state_embedding_dimensions": (
                global_state_embedding_dimensions
            ),
            "num_heads": num_heads,
        },
        "training": {
            "optimizer": "adam",
            "batch_size": batch_size,
            "learn_rate": learn_rate,
            "loss": {
                "name": "auxiliary" if has_auxiliary_targets else "base",
                "value_scale": value_scale,
                "policy_scale": policy_scale,
                **(
                    {
                        "experiment_arm": experiment.name,
                        "round_score_scale": experiment.round_score_scale,
                        "future_clear_scale": experiment.future_clear_scale,
                        "clear_positive_weight": experiment.clear_positive_weight,
                    }
                    if has_auxiliary_targets
                    else {}
                ),
            },
        },
        "search": {
            "score_utility_weight": experiment.score_utility_weight,
        },
        "dataset": dataset_configuration,
    }
    progress = checkpoint.TrainingProgress()
    if resume_checkpoint is not None:
        progress = checkpoint.load_checkpoint(
            resume_checkpoint,
            model=model,
            optimizer=optimizer,
            expected_configuration=resume_configuration,
            map_location=device,
        )
    if optimizer_steps < progress.optimizer_steps:
        raise typer.BadParameter(
            f"--steps {optimizer_steps} is below checkpoint progress "
            f"{progress.optimizer_steps}"
        )
    remaining_steps = optimizer_steps - progress.optimizer_steps

    train_start = time.perf_counter()
    train.train_steps(
        model,
        training_buffer,
        training_batch_size=batch_size,
        optimizer_steps=remaining_steps,
        optimizer=optimizer,
        loss_function=loss_function,
    )
    train_seconds = time.perf_counter() - train_start

    validation_start = time.perf_counter()
    training_loss = train.evaluate_loss(
        model,
        training_buffer,
        evaluation_batch_size=batch_size,
        loss_function=loss_function,
    )
    validation_loss = (
        train.evaluate_loss(
            model,
            validation_buffer,
            evaluation_batch_size=batch_size,
            loss_function=loss_function,
        )
        if validation_buffer is not None
        else None
    )
    validation_seconds = time.perf_counter() - validation_start

    sampled_positions = remaining_steps * batch_size
    progress = dataclasses.replace(
        progress,
        optimizer_steps=optimizer_steps,
        sampled_positions=progress.sampled_positions + sampled_positions,
        trained_positions=progress.trained_positions + sampled_positions,
    )
    save_seconds = 0.0
    if output_checkpoint is not None:
        save_start = time.perf_counter()
        checkpoint.save_checkpoint(
            output_checkpoint,
            model=model,
            optimizer=optimizer,
            configuration=resume_configuration,
            progress=progress,
        )
        save_seconds = time.perf_counter() - save_start

    typer.echo(f"dataset_games: {complete_buffer.game_count}")
    typer.echo(f"dataset_positions: {len(complete_buffer)}")
    typer.echo(
        "selected_game_indices: "
        + ",".join(str(game_index) for game_index in complete_buffer.game_indices)
    )
    typer.echo(f"training_games: {training_buffer.game_count}")
    typer.echo(f"training_positions: {len(training_buffer)}")
    if validation_buffer is not None:
        typer.echo(f"validation_games: {validation_buffer.game_count}")
        typer.echo(f"validation_positions: {len(validation_buffer)}")
    typer.echo(f"optimizer_steps: {progress.optimizer_steps}")
    typer.echo(f"sampled_position_exposures: {progress.sampled_positions}")
    typer.echo(f"load_seconds: {load_seconds:.6f}")
    typer.echo(f"train_seconds: {train_seconds:.6f}")
    if validation_buffer is not None:
        typer.echo(f"validation_seconds: {validation_seconds:.6f}")
    typer.echo(f"save_seconds: {save_seconds:.6f}")
    _print_loss("train", training_loss)
    if validation_loss is not None:
        _print_loss("validation", validation_loss)


if __name__ == "__main__":
    typer.run(main)
