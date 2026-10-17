"""Status: experimental
Purpose: Diagnose frozen search targets and fixed-target learning before more self-play.
Promote when: Repeated experiments establish a useful standard diagnostic set.
"""

from pathlib import Path
from typing import Annotated

import typer


def diagnose(
    plan: Annotated[Path, typer.Option(help="JSON plan with frozen checkpoint paths.")],
    output: Annotated[Path, typer.Option(help="New output directory.")],
    allow_dirty: bool = False,
) -> Path:
    from skyjo.experiments.learning_loop_diagnostic import run

    return run(plan, output, allow_dirty=allow_dirty)


def main():
    typer.run(diagnose)


if __name__ == "__main__":
    main()
