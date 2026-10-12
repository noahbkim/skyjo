"""Status: manual. Purpose: launch terminal games and saved-game playback."""

from pathlib import Path
from typing import Annotated

import typer


def launch(
    preset: Annotated[
        Path | None, typer.Option(help="Load a matchup TOML preset")
    ] = None,
    recording: Annotated[
        Path | None, typer.Option(help="Open a game recording")
    ] = None,
):
    from skyjo.ui.app import SkyjoApp

    SkyjoApp(preset=preset, recording=recording).run()


def main():
    typer.run(launch)


if __name__ == "__main__":
    main()
