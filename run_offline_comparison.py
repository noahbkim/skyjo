"""Status: manual
Purpose: Compare ordinary run configs on a frozen replay dataset.
Promote when: Additional orchestration belongs in skyjo.offline_comparison.
"""

from pathlib import Path
from typing import Annotated

import typer

from skyjo.experiments.offline_comparison import launch_comparison


def compare(
    config: Annotated[
        list[Path],
        typer.Option("--config", help="Ordinary run config; repeat for every variant."),
    ],
    dataset: Annotated[Path, typer.Option("--dataset")],
    runs_dir: Annotated[Path, typer.Option("--runs-dir")] = Path(".runs"),
    seeds: Annotated[str, typer.Option("--seeds")] = "0,1,2",
    steps: Annotated[int, typer.Option("--steps", min=0)] = 2000,
    validation_fraction: Annotated[float, typer.Option("--validation-fraction")] = 0.1,
    split_seed: Annotated[int, typer.Option("--split-seed")] = 0,
    evaluation_interval: Annotated[
        int, typer.Option("--evaluation-interval", min=1)
    ] = 200,
    allow_dirty: Annotated[bool, typer.Option("--allow-dirty")] = False,
) -> Path:
    """Train fresh models with paired seeds and a common optimizer-step budget."""
    try:
        parsed_seeds = tuple(int(seed.strip()) for seed in seeds.split(","))
    except ValueError as error:
        raise typer.BadParameter("--seeds must be comma-separated integers") from error
    result = launch_comparison(
        config,
        dataset,
        runs_dir,
        seeds=parsed_seeds,
        steps=steps,
        validation_fraction=validation_fraction,
        split_seed=split_seed,
        evaluation_interval=evaluation_interval,
        allow_dirty=allow_dirty,
    )
    typer.echo(f"Comparison: {result}")
    return result


def main():
    typer.run(compare)


if __name__ == "__main__":
    main()
