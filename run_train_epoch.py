"""Run one Skyjo backpropagation epoch from a saved ReplayBuffer.

This is a narrow benchmark/debug entrypoint for isolating the training step
without self-play, queues, predictors, or model factory bookkeeping.
"""

from __future__ import annotations

import pathlib
import random
import time
import typing

import numpy as np
import torch
import typer

from skyjo import buffer, skynet, train, train_utils

DEFAULT_SEED = 0
DEFAULT_BATCH_SIZE = 256
DEFAULT_LEARN_RATE = 1e-3
DEFAULT_EMBEDDING_DIMENSIONS = 32
DEFAULT_GLOBAL_STATE_EMBEDDING_DIMENSIONS = 64
DEFAULT_NUM_HEADS = 2


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_state_dict(
    weights_path: pathlib.Path,
    device: torch.device,
) -> dict[str, typing.Any]:
    loaded = torch.load(weights_path, map_location=device, weights_only=True)
    if isinstance(loaded, dict) and "state_dict" in loaded:
        return loaded["state_dict"]
    if isinstance(loaded, dict) and "model_state_dict" in loaded:
        return loaded["model_state_dict"]
    return loaded


def build_model(
    training_data_buffer: buffer.ReplayBuffer,
    device: torch.device,
    embedding_dimensions: int,
    global_state_embedding_dimensions: int,
    num_heads: int,
) -> skynet.EquivariantSkyNet:
    return skynet.EquivariantSkyNet(
        spatial_input_shape=training_data_buffer.spatial_input_buffer.shape[1:],
        non_spatial_input_shape=training_data_buffer.non_spatial_input_buffer.shape[1:],
        value_output_shape=training_data_buffer.outcome_target_buffer.shape[1:],
        policy_output_shape=training_data_buffer.policy_target_buffer.shape[1:],
        device=device,
        embedding_dimensions=embedding_dimensions,
        global_state_embedding_dimensions=global_state_embedding_dimensions,
        num_heads=num_heads,
    )


def main(
    buffer_path: pathlib.Path = typer.Argument(
        ...,
        help="Path to a pickled skyjo.buffer.ReplayBuffer.",
    ),
    weights: pathlib.Path | None = typer.Option(
        None,
        "--weights",
        help="Optional .pth model state_dict to load before training.",
    ),
    output_weights: pathlib.Path | None = typer.Option(
        None,
        "--output-weights",
        help="Optional path where the post-epoch model state_dict should be saved.",
    ),
    seed: int = typer.Option(
        DEFAULT_SEED,
        "--seed",
        help="Seed used when initializing a model without --weights.",
    ),
    device_name: str = typer.Option(
        "cpu",
        "--device",
        help="Torch device to train on, for example cpu, cuda, or mps.",
    ),
    batch_size: int = typer.Option(
        DEFAULT_BATCH_SIZE,
        "--batch-size",
        help="Training batch size.",
    ),
    learn_rate: float = typer.Option(
        DEFAULT_LEARN_RATE,
        "--learn-rate",
        help="Adam learning rate passed to train.train_epoch.",
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
    cleared_columns_scale: float = typer.Option(
        0.1,
        "--cleared-columns-scale",
        help="Scale for the cleared-columns loss term.",
    ),
) -> None:
    """Run one training epoch from a saved ReplayBuffer."""
    device = torch.device(device_name)

    set_seed(seed)

    load_start = time.perf_counter()
    training_data_buffer = buffer.ReplayBuffer.load(buffer_path)
    load_seconds = time.perf_counter() - load_start

    buffer_size = len(training_data_buffer)
    if buffer_size == 0:
        raise typer.BadParameter(f"ReplayBuffer at {buffer_path} is empty.")
    if batch_size > buffer_size:
        raise ValueError(
            f"Batch size {batch_size} cannot be larger than buffer size {buffer_size}."
        )

    model = build_model(
        training_data_buffer,
        device=device,
        embedding_dimensions=embedding_dimensions,
        global_state_embedding_dimensions=global_state_embedding_dimensions,
        num_heads=num_heads,
    )

    if weights is not None:
        model.load_state_dict(load_state_dict(weights, device=device))

    batch_count = buffer_size // batch_size + 1
    optimizer = train.make_optimizer(model, learn_rate)
    train_start = time.perf_counter()
    loss_details = train.train_epoch(
        model,
        training_data_buffer,
        training_batch_size=batch_size,
        optimizer=optimizer,
        loss_function=lambda model_outputs, targets: train_utils.base_loss(
            model_outputs,
            targets,
            value_scale=value_scale,
            policy_scale=policy_scale,
            cleared_columns_scale=cleared_columns_scale,
        ),
    )
    train_seconds = time.perf_counter() - train_start

    if output_weights is not None:
        output_weights.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output_weights)

    typer.echo(f"buffer_path: {buffer_path}")
    typer.echo(f"buffer_size: {buffer_size}")
    typer.echo(f"batch_size: {batch_size}")
    typer.echo(f"batches: {batch_count}")
    typer.echo(f"device: {device}")
    typer.echo(f"weights: {weights if weights is not None else 'initialized'}")
    typer.echo(f"seed: {seed}")
    typer.echo(f"buffer_load_seconds: {load_seconds:.6f}")
    typer.echo(f"train_epoch_seconds: {train_seconds:.6f}")
    typer.echo(f"seconds_per_batch: {train_seconds / batch_count:.6f}")
    typer.echo("loss_summary:")
    typer.echo(train_utils.loss_details_summary(loss_details).to_string())
    if output_weights is not None:
        typer.echo(f"output_weights: {output_weights}")


if __name__ == "__main__":
    typer.run(main)
