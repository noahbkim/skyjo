"""Status: experimental
Purpose: Profile frozen search workloads and compare isolated performance PoCs.
Promote when: A measured alternative is selected for production implementation.
"""

import os
from pathlib import Path
from typing import Annotated

import typer


def benchmark(
    stage: Annotated[
        str, typer.Option(help="prepare, searches, profile, batches, or games")
    ],
    output: Annotated[Path, typer.Option()],
    inputs: Annotated[Path, typer.Option()] = Path(".runs/search-performance-inputs"),
    checkpoint: Annotated[Path | None, typer.Option()] = None,
    config: Annotated[Path | None, typer.Option()] = None,
    replay: Annotated[Path | None, typer.Option()] = None,
    variant: Annotated[list[str] | None, typer.Option()] = None,
    budget: Annotated[list[int] | None, typer.Option()] = None,
    workers: Annotated[list[int] | None, typer.Option()] = None,
    sweeps: int = 3,
    positions: int = 128,
    game_count: int = 8,
    seed: int = 20261017,
    quick: Annotated[
        bool,
        typer.Option(help="Short search comparison on the first 16 frozen positions"),
    ] = False,
) -> Path:
    from skyjo.experiments.search_performance import run

    if stage == "prepare" and any(p is None for p in (checkpoint, config, replay)):
        raise ValueError("prepare requires checkpoint, config, and replay")
    if quick:
        if stage != "searches":
            raise ValueError("--quick is for the searches stage")
        positions, sweeps = min(positions, 16), min(sweeps, 2)
        budget = budget or [128]
    return run(
        stage=stage,
        output=output,
        inputs=inputs,
        checkpoint=checkpoint,
        config=config,
        replay=replay,
        variants=tuple(variant)
        if variant
        else ("baseline", "inference", "snapshots", "combined"),
        budgets=tuple(budget) if budget else (32, 128, 256),
        workers=tuple(workers) if workers else (1, 4, 8),
        sweeps=sweeps,
        positions=positions,
        game_count=game_count,
        seed=seed,
    )


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    typer.run(benchmark)


if __name__ == "__main__":
    main()
