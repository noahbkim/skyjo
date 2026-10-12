"""Status: manual
Purpose: Measure symmetric-action MCTS speed on frozen public replay positions.
Promote when: This benchmark becomes a recurring performance check.
"""

import os
from pathlib import Path
from typing import Annotated

import typer


def benchmark(
    control_replay: Annotated[Path, typer.Option("--control-replay")],
    variant_replay: Annotated[Path, typer.Option("--variant-replay")],
    checkpoint: Annotated[Path, typer.Option("--checkpoint")],
    boundary_checkpoint: Annotated[Path, typer.Option("--boundary-checkpoint")],
    runs_dir: Annotated[Path, typer.Option("--runs-dir")] = Path(".runs"),
    positions: Annotated[int, typer.Option("--positions")] = 128,
    iterations: Annotated[list[int] | None, typer.Option("--iterations")] = None,
    sweeps: Annotated[int, typer.Option("--sweeps")] = 3,
    warmup: Annotated[int, typer.Option("--warmup")] = 2,
    seed: Annotated[int, typer.Option("--seed")] = 20261009,
    allow_dirty: Annotated[bool, typer.Option("--allow-dirty")] = False,
) -> Path:
    """Compare pooling off/on; paths identify immutable replay snapshots and weights."""
    from skyjo.experiments.mcts_benchmark import Settings, run_benchmark

    result = run_benchmark(
        control_replay=control_replay,
        variant_replay=variant_replay,
        checkpoint=checkpoint,
        boundary_checkpoint=boundary_checkpoint,
        runs_dir=runs_dir,
        settings=Settings(
            positions=positions,
            iterations=tuple(iterations) if iterations is not None else (32, 128),
            sweeps=sweeps,
            warmup=warmup,
            seed=seed,
        ),
        allow_dirty=allow_dirty,
    )
    typer.echo(f"MCTS benchmark: {result}")
    return result


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    typer.run(benchmark)


if __name__ == "__main__":
    main()
