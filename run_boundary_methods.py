"""Status: experimental
Purpose: Compare frozen round-boundary valuation methods on later held-out games.
Promote when: This becomes the standard boundary evaluator benchmark.
"""

import os
from pathlib import Path
from typing import Annotated

import typer


def compare(
    round_log: Annotated[list[Path], typer.Option("--round-log")],
    checkpoint: Annotated[Path, typer.Option("--checkpoint")],
    boundary_run: Annotated[Path, typer.Option("--boundary-run")],
    runs_dir: Annotated[Path, typer.Option("--runs-dir")] = Path(".runs"),
    repeats: Annotated[int, typer.Option("--repeats", min=1)] = 3,
    seed: Annotated[int, typer.Option("--seed", min=0)] = 20261006,
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = 256,
    allow_dirty: Annotated[bool, typer.Option("--allow-dirty")] = False,
) -> Path:
    """Evaluate settled score states using score values or 1/10/100 next deals."""
    from skyjo.experiments.boundary_methods import launch_comparison

    result = launch_comparison(
        round_log,
        checkpoint,
        boundary_run,
        runs_dir,
        repeats=repeats,
        seed=seed,
        batch_size=batch_size,
        allow_dirty=allow_dirty,
    )
    typer.echo(f"Comparison: {result}")
    return result


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    typer.run(compare)


if __name__ == "__main__":
    main()
