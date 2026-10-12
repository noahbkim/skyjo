"""Single-dataset training and exact resume using the shared learner."""

from __future__ import annotations

import dataclasses
import pathlib
import time

import torch

from skyjo.learning import buffer, checkpoint, objectives, replay_io
from skyjo.learning.learner import Learner


@dataclasses.dataclass(frozen=True)
class OfflineTrainingResult:
    progress: checkpoint.TrainingProgress
    checkpoint: pathlib.Path | None
    metrics: dict


def _select_game_indices(
    replay_buffer: buffer.ReplayBuffer,
    game_indices: list[int] | None,
) -> buffer.ReplayBuffer:
    """Select games by their persisted IDs, preserving dataset order."""
    if not game_indices:
        return replay_buffer
    if len(set(game_indices)) != len(game_indices):
        raise ValueError("--game-index values must be unique")

    requested = set(game_indices)
    missing = requested.difference(replay_buffer.game_indices)
    if missing:
        missing_text = ", ".join(str(game_index) for game_index in sorted(missing))
        raise ValueError(f"unknown --game-index value(s): {missing_text}")
    selected_positions = [
        position
        for position, game_index in enumerate(replay_buffer.game_indices)
        if game_index in requested
    ]
    return replay_buffer.select_games(selected_positions)


def run(
    dataset_path: pathlib.Path,
    *,
    resume_checkpoint: pathlib.Path | None = None,
    output_checkpoint: pathlib.Path | None = None,
    optimizer_steps: int = 1,
    seed: int = 0,
    split_seed: int = 0,
    validation_fraction: float = 0.1,
    game_indices: list[int] | None = None,
    device_name: str = "cpu",
    batch_size: int = 256,
    learn_rate: float = 0.001,
    embedding_dimensions: int = 32,
    global_state_embedding_dimensions: int = 64,
    num_heads: int = 2,
    value_scale: float = 1.0,
    policy_scale: float = 1.0,
    auxiliary_objectives=None,
) -> OfflineTrainingResult:
    device = torch.device(device_name)

    load_start = time.perf_counter()
    complete_buffer = replay_io.load(dataset_path)
    selected_objectives = objectives.resolve(auxiliary_objectives).weights
    missing = selected_objectives.keys() - complete_buffer.target_buffers.keys()
    if missing:
        raise ValueError(
            f"Missing required auxiliary targets: {sorted(missing)}",
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

    learner = Learner.create(
        {
            "embedding_dimensions": embedding_dimensions,
            "global_state_embedding_dimensions": global_state_embedding_dimensions,
            "num_heads": num_heads,
        },
        players=complete_buffer.spatial_input_buffer.shape[1],
        device=device,
        auxiliary_objectives=selected_objectives,
        learn_rate=learn_rate,
        seed=seed,
        value_scale=value_scale,
        policy_scale=policy_scale,
    )
    model, optimizer = learner.model, learner.optimizer
    dataset_configuration = {
        "dataset_id": complete_buffer.dataset_id,
        "validation_fraction": validation_fraction,
        "split_seed": split_seed,
    }
    if game_indices:
        dataset_configuration["game_indices"] = list(complete_buffer.game_indices)
    resume_configuration = {
        "players": model.players,
        "model": {
            "name": model.architecture_name,
            "embedding_dimensions": embedding_dimensions,
            "global_state_embedding_dimensions": global_state_embedding_dimensions,
            "num_heads": num_heads,
        },
        "auxiliary_objectives": selected_objectives,
        "training": {
            "optimizer": "adam",
            "batch_size": batch_size,
            "learn_rate": learn_rate,
            "loss": {
                "name": "configured",
                "auxiliary_objectives": selected_objectives,
                "value_scale": value_scale,
                "policy_scale": policy_scale,
            },
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
            sampling_rng=learner.sampling_rng,
        )
    if optimizer_steps < progress.optimizer_steps:
        raise ValueError(
            f"--steps {optimizer_steps} is below checkpoint progress "
            f"{progress.optimizer_steps}"
        )
    remaining_steps = optimizer_steps - progress.optimizer_steps

    train_start = time.perf_counter()
    learner.fit(training_buffer, batch_size=batch_size, steps=remaining_steps)
    train_seconds = time.perf_counter() - train_start

    validation_start = time.perf_counter()
    training_loss = learner.evaluate(training_buffer, batch_size=batch_size)
    validation_loss = (
        learner.evaluate(validation_buffer, batch_size=batch_size)
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
            sampling_rng=learner.sampling_rng,
        )
        save_seconds = time.perf_counter() - save_start

    metrics = {
        "dataset_games": complete_buffer.game_count,
        "dataset_positions": len(complete_buffer),
        "selected_game_indices": ",".join(map(str, complete_buffer.game_indices)),
        "training_games": training_buffer.game_count,
        "training_positions": len(training_buffer),
        "optimizer_steps": progress.optimizer_steps,
        "sampled_position_exposures": progress.sampled_positions,
        "load_seconds": load_seconds,
        "train_seconds": train_seconds,
        "save_seconds": save_seconds,
        **{f"train_{key}": value for key, value in training_loss.items()},
    }
    if validation_buffer is not None:
        metrics.update(
            validation_games=validation_buffer.game_count,
            validation_positions=len(validation_buffer),
            validation_seconds=validation_seconds,
        )
        metrics.update(
            {f"validation_{key}": value for key, value in validation_loss.items()}
        )
    return OfflineTrainingResult(progress, output_checkpoint, metrics)
