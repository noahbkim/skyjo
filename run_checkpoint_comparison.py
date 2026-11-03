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
    first_round: Annotated[
        bool,
        typer.Option(help="Stop after the first round from a fresh zero-score game."),
    ] = False,
    mode: Annotated[
        str | None,
        typer.Option(help="Shared shortcut: search, policy (sampling), or value."),
    ] = None,
    temperature: Annotated[
        float | None,
        typer.Option(
            min=0, help="Shared action temperature; policy shortcut defaults to 1."
        ),
    ] = None,
    value_chance: Annotated[
        str | None,
        typer.Option(
            help="Value-only ordinary chance: enumerated (default) or sampled."
        ),
    ] = None,
    value_samples: Annotated[
        int | None,
        typer.Option(
            min=1, help="Ordinary samples per random action in sampled value mode."
        ),
    ] = None,
    boundary_samples: Annotated[
        int | None,
        typer.Option(
            min=1, help="Round-ending samples per action in value mode (default 10)."
        ),
    ] = None,
) -> Path:
    shortcuts = any(
        value is not None
        for value in (mode, temperature, value_chance, value_samples, boundary_samples)
    )
    if shortcuts and (control_settings is not None or variant_settings is not None):
        raise typer.BadParameter(
            "Shared shortcuts cannot be combined with contestant settings files. "
            "Use --control-settings and --variant-settings for asymmetric comparisons."
        )
    try:
        if shortcuts:
            players = [
                contestants.from_shortcut(
                    path,
                    mode=mode if mode is not None else "search",
                    iterations=iterations,
                    temperature=temperature,
                    value_chance=value_chance,
                    value_samples=value_samples,
                    boundary_samples=boundary_samples,
                )
                for path in (control, variant or control)
            ]
        else:
            players = [
                contestants.from_file(path, settings_path, iterations=iterations)
                for path, settings_path in (
                    (control, control_settings),
                    (variant or control, variant_settings),
                )
            ]
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    settings = evaluation.EvaluationConfig(
        *players,
        seed_count=seed_count,
        seed=seed,
        workers=workers,
        threads=threads,
        first_round=first_round,
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
