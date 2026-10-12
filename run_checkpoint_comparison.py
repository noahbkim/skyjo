"""Status: manual
Purpose: Compare two checkpoint contestants with balanced seats.
Promote when: Reusable execution belongs in skyjo.experiments.evaluation.
"""

import logging
from pathlib import Path
from typing import Annotated

import typer

from skyjo.experiments import contestants, evaluation


def compare(
    control: Annotated[Path, typer.Option(help="Control checkpoint.")],
    variant: Annotated[
        Path | None, typer.Option(help="Variant checkpoint; defaults to control.")
    ] = None,
    control_settings: Annotated[
        Path | None, typer.Option(help="Optional contestant TOML.")
    ] = None,
    variant_settings: Annotated[
        Path | None, typer.Option(help="Optional contestant TOML.")
    ] = None,
    iterations: Annotated[
        int, typer.Option(min=1, help="Default search budget for either contestant.")
    ] = 128,
    seed_count: Annotated[int, typer.Option(min=1)] = 32,
    seed: int = 0,
    workers: Annotated[int, typer.Option(min=1)] = 1,
    threads: Annotated[int, typer.Option(min=1)] = 1,
    runs_dir: Path = Path(".runs"),
    allow_dirty: bool = False,
) -> Path:
    settings = evaluation.EvaluationConfig(
        contestants.from_file(control, control_settings, iterations=iterations),
        contestants.from_file(
            variant or control, variant_settings, iterations=iterations
        ),
        seed_count=seed_count,
        seed=seed,
        workers=workers,
        threads=threads,
    )
    path = evaluation.launch_comparison(
        settings,
        runs_dir,
        repository=Path(__file__).resolve().parent,
        allow_dirty=allow_dirty,
    )
    typer.echo(f"Report: {path / 'comparison.json'}")
    return path


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    typer.run(compare)


if __name__ == "__main__":
    main()
