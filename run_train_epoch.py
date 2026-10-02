"""Status: manual
Purpose: Run one backpropagation epoch from saved replay, with configurable auxiliaries.
Promote when: A shared training CLI replaces the experiment entrypoints.
"""

from __future__ import annotations

import functools
import json
import pathlib
import random
import time
from typing import Annotated

import numpy as np
import torch
import typer

from skyjo import buffer, objectives, skynet, train, train_utils


def build_model(replay, device, embedding_dimensions, global_state_embedding_dimensions,
                num_heads, auxiliary_objectives=None):
    return skynet.EquivariantSkyNet(
        spatial_input_shape=replay.spatial_input_buffer.shape[1:],
        non_spatial_input_shape=replay.non_spatial_input_buffer.shape[1:],
        value_output_shape=replay.target_buffers["value"].shape[1:],
        policy_output_shape=replay.target_buffers["policy"].shape[1:],
        device=device, embedding_dimensions=embedding_dimensions,
        global_state_embedding_dimensions=global_state_embedding_dimensions,
        num_heads=num_heads, auxiliary_objectives=auxiliary_objectives,
    )


def run_epoch(
    buffer_path: Annotated[pathlib.Path, typer.Argument(help="Saved ReplayBuffer")],
    weights: pathlib.Path | None = None,
    output_weights: pathlib.Path | None = None,
    auxiliary_objectives: Annotated[str | None, typer.Option(help="JSON mapping of objective names to loss weights; defaults to checkpoint config or {}.")] = None,
    seed: int = 0,
    device_name: Annotated[str, typer.Option("--device")] = "cpu",
    batch_size: int = 256,
    learn_rate: float = 1e-3,
    embedding_dimensions: int = 32,
    global_state_embedding_dimensions: int = 64,
    num_heads: int = 2,
    value_scale: float = 1.0 / skynet.SCORE_DIFFERENTIAL_CAP**2,
    policy_scale: float = 1.0,
) -> None:
    device = torch.device(device_name)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    config = None if auxiliary_objectives is None else objectives.resolve(json.loads(auxiliary_objectives))
    replay = buffer.ReplayBuffer.load(buffer_path)
    if batch_size < 1 or len(replay) < batch_size:
        raise ValueError("Batch size must be positive and no larger than replay")
    if weights is not None and weights.with_suffix(".json").exists():
        model = skynet.EquivariantSkyNet.from_checkpoint(weights, device)
        if config is not None and model.objectives != config:
            raise ValueError("Requested objectives differ from checkpoint; start a fresh ablation run")
    else:
        model = build_model(replay, device, embedding_dimensions,
                            global_state_embedding_dimensions, num_heads, config)
        if weights is not None:
            model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    replay.validate_objectives(model.objectives)
    typer.echo(f"auxiliary_objectives: {json.dumps(model.objectives.weights, sort_keys=True)}")
    optimizer = train.make_optimizer(model, learn_rate)
    start = time.perf_counter()
    details = train.train_epoch(
        model, replay, batch_size, optimizer,
        functools.partial(train_utils.base_loss, value_scale=value_scale, policy_scale=policy_scale),
    )
    typer.echo(f"train_epoch_seconds: {time.perf_counter() - start:.6f}")
    typer.echo(train_utils.loss_details_summary(details).to_string())
    if output_weights is not None:
        model.save_checkpoint(output_weights)
        typer.echo(f"output_weights: {output_weights}")


def main() -> None:
    typer.run(run_epoch)


if __name__ == "__main__":
    main()
