"""Status: experimental
Purpose: Compare score-only boundary win prediction on recorded full games.
Promote when: Boundary evaluation is selected for integration with search.
"""

import os
from pathlib import Path
from typing import Annotated

import typer


def experiment(
    source_run: Annotated[Path, typer.Option("--source-run")],
    config: Annotated[Path, typer.Option("--config")] = Path(
        "configs/boundary_value.toml"
    ),
    runs_dir: Annotated[Path, typer.Option("--runs-dir")] = Path(".runs"),
    allow_dirty: Annotated[bool, typer.Option("--allow-dirty")] = False,
) -> Path:
    """Fit logistic and neural boundary values; leave gameplay unchanged."""
    from skyjo.boundary_experiment import launch_experiment

    result = launch_experiment(source_run, config, runs_dir, allow_dirty=allow_dirty)
    typer.echo(f"Boundary experiment: {result}")
    return result


def main() -> None:
    # Configure OpenMP before importing Torch, including before provenance forks.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    typer.run(experiment)


if __name__ == "__main__":
    main()
