"""Status: manual
Purpose: Curate public positions and compare tactical search behavior.
Promote when: The suite becomes a scheduled research regression check.
"""

import os
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(no_args_is_help=True)


@app.command()
def run(
    checkpoint: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()],
    suite: Annotated[Path, typer.Option()] = Path("data/search-positions"),
    plan: Annotated[Path, typer.Option()] = Path("configs/search-diagnostics.toml"),
    allow_dirty: Annotated[bool, typer.Option()] = False,
) -> Path:
    """Run a fixed suite × methods × budgets × paired seeds."""
    from skyjo.experiments.search_diagnostics import run_suite

    result = run_suite(
        suite=suite,
        checkpoint=checkpoint,
        plan=plan,
        output=output,
        allow_dirty=allow_dirty,
    )
    typer.echo(f"Search diagnostics: {result / 'report.md'}")
    return result


@app.command()
def validate(
    suite: Annotated[Path, typer.Option()] = Path("data/search-positions"),
) -> None:
    """Validate states, labels and deterministic witness prefixes without weights."""
    from skyjo.experiments.search_positions import load_suite

    for position in load_suite(suite):
        typer.echo(
            f"{position.id}: {position.expectation.kind}; {len(position.witnesses)} witnesses"
        )


@app.command("add")
def add_position(
    recording: Annotated[Path, typer.Option()],
    frame: Annotated[int, typer.Option(help="Zero-based recording frame")],
    output: Annotated[Path, typer.Option()],
    id: Annotated[str, typer.Option()],
    description: Annotated[str, typer.Option()],
    rationale: Annotated[str, typer.Option()],
    kind: Annotated[str, typer.Option()] = "unscored",
    action: Annotated[
        list[int] | None, typer.Option(help="Expected/avoided action; repeatable")
    ] = None,
    witness: Annotated[
        list[int] | None,
        typer.Option(help="Actions of one tactical line, in order; repeatable"),
    ] = None,
    tag: Annotated[list[str] | None, typer.Option()] = None,
) -> Path:
    """Export one portable annotated snapshot; never overwrite an existing case."""
    from skyjo.experiments.search_positions import Expectation, Witness, export_position

    result = export_position(
        recording,
        frame,
        output,
        id=id,
        description=description,
        expectation=Expectation(kind, tuple(action or ()), rationale),
        witnesses=(Witness("tactic", tuple(witness)),) if witness else (),
        tags=tuple(tag or ()),
    )
    typer.echo(result)
    return result


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    app()


if __name__ == "__main__":
    main()
