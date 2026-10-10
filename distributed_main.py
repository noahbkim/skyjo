"""Launch a recorded self-play training run from a configuration file."""

import pathlib
import typing

import typer

from skyjo.experiments import selfplay_training


def launch(
    config: typing.Annotated[pathlib.Path, typer.Option("--config")],
    runs_dir: typing.Annotated[pathlib.Path, typer.Option("--runs-dir")] = pathlib.Path(
        ".runs"
    ),
    allow_dirty: typing.Annotated[bool, typer.Option("--allow-dirty")] = False,
) -> None:
    result = selfplay_training.launch(
        config,
        runs_dir,
        repository=pathlib.Path(__file__).resolve().parent,
        allow_dirty=allow_dirty,
    )
    typer.echo(f"Run directory: {result.path}")


if __name__ == "__main__":
    typer.run(launch)
