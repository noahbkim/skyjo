"""Launch independently trained, recorded runs from a named configuration suite."""

from pathlib import Path
from typing import Annotated

import typer

from skyjo.experiments import launch_suite


def compare(
    config: Annotated[Path, typer.Option("--config")],
    runs_dir: Annotated[Path, typer.Option("--runs-dir")] = Path(".runs"),
    allow_dirty: Annotated[bool, typer.Option("--allow-dirty")] = False,
) -> None:
    typer.echo(launch_suite(config, runs_dir, allow_dirty=allow_dirty))


if __name__ == "__main__":
    typer.run(compare)
