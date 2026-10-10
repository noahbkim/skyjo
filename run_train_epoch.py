"""Status: manual
Purpose: Train on a saved replay dataset with a cumulative optimizer-step budget.
Promote when: Reusable work belongs in skyjo.experiments.offline_training.
"""

import json
import pathlib
from typing import Annotated

import typer

from skyjo.experiments import offline_training

DEFAULT_SEED = 0
DEFAULT_BATCH_SIZE = 256
DEFAULT_OPTIMIZER_STEPS = 1
DEFAULT_LEARN_RATE = 1e-3
DEFAULT_EMBEDDING_DIMENSIONS = 32
DEFAULT_GLOBAL_STATE_EMBEDDING_DIMENSIONS = 64
DEFAULT_NUM_HEADS = 2
DEFAULT_VALIDATION_FRACTION = 0.1


def main(
    dataset_path: Annotated[
        pathlib.Path,
        typer.Argument(help="Path to a versioned Skyjo replay dataset directory."),
    ],
    resume_checkpoint: Annotated[
        pathlib.Path | None,
        typer.Option(
            "--checkpoint",
            help="Optional versioned checkpoint to resume before training.",
        ),
    ] = None,
    output_checkpoint: Annotated[
        pathlib.Path | None,
        typer.Option(
            "--output-checkpoint",
            help="Optional path for the resumable post-training checkpoint.",
        ),
    ] = None,
    optimizer_steps: Annotated[
        int,
        typer.Option(
            "--steps",
            min=0,
            help="Cumulative optimizer-step target, including resumed steps.",
        ),
    ] = DEFAULT_OPTIMIZER_STEPS,
    seed: Annotated[
        int,
        typer.Option(
            "--seed", help="Seed for model initialization and dataset sampling."
        ),
    ] = DEFAULT_SEED,
    split_seed: Annotated[
        int,
        typer.Option(
            "--split-seed",
            help="Seed used for the deterministic game-level validation split.",
        ),
    ] = DEFAULT_SEED,
    validation_fraction: Annotated[
        float,
        typer.Option(
            "--validation-fraction",
            min=0.0,
            max=1.0,
            help="Fraction of complete games reserved for validation; zero disables validation.",
        ),
    ] = DEFAULT_VALIDATION_FRACTION,
    game_indices: Annotated[
        list[int] | None,
        typer.Option(
            "--game-index",
            help="Persisted game ID to include; repeat to select multiple games.",
        ),
    ] = None,
    device_name: Annotated[
        str,
        typer.Option("--device", help="Torch device, for example cpu, cuda, or mps."),
    ] = "cpu",
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", min=1, help="Training and evaluation batch size."),
    ] = DEFAULT_BATCH_SIZE,
    learn_rate: Annotated[
        float, typer.Option("--learn-rate", help="Adam learning rate.")
    ] = DEFAULT_LEARN_RATE,
    embedding_dimensions: Annotated[
        int,
        typer.Option(
            "--embedding-dimensions", help="EquivariantSkyNet embedding_dimensions."
        ),
    ] = DEFAULT_EMBEDDING_DIMENSIONS,
    global_state_embedding_dimensions: Annotated[
        int,
        typer.Option(
            "--global-state-embedding-dimensions",
            help="EquivariantSkyNet global_state_embedding_dimensions.",
        ),
    ] = DEFAULT_GLOBAL_STATE_EMBEDDING_DIMENSIONS,
    num_heads: Annotated[
        int, typer.Option("--num-heads", help="EquivariantSkyNet num_heads.")
    ] = DEFAULT_NUM_HEADS,
    value_scale: Annotated[
        float, typer.Option("--value-scale", help="Scale for the value loss term.")
    ] = 1.0,
    policy_scale: Annotated[
        float, typer.Option("--policy-scale", help="Scale for the policy loss term.")
    ] = 1.0,
    auxiliary_objectives: Annotated[
        str | None,
        typer.Option(
            "--auxiliary-objectives",
            help="JSON mapping of round objective names to weights; requires matching replay labels.",
        ),
    ] = None,
) -> None:
    try:
        result = offline_training.run(
            dataset_path,
            resume_checkpoint=resume_checkpoint,
            output_checkpoint=output_checkpoint,
            optimizer_steps=optimizer_steps,
            seed=seed,
            split_seed=split_seed,
            validation_fraction=validation_fraction,
            game_indices=game_indices,
            device_name=device_name,
            batch_size=batch_size,
            learn_rate=learn_rate,
            embedding_dimensions=embedding_dimensions,
            global_state_embedding_dimensions=global_state_embedding_dimensions,
            num_heads=num_heads,
            value_scale=value_scale,
            policy_scale=policy_scale,
            auxiliary_objectives=json.loads(auxiliary_objectives)
            if auxiliary_objectives
            else None,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error

    for name, value in result.metrics.items():
        typer.echo(f"{name}: {value}")


if __name__ == "__main__":
    typer.run(main)
