"""Status: manual
Purpose: Compare frozen checkpoints or search budgets in seat-balanced full games.
Promote when: Additional evaluation logic belongs in skyjo.evaluation.
"""

import dataclasses
import json
from pathlib import Path
from typing import Annotated

import torch
import typer

from skyjo import evaluation, runs


def compare(
    control: Annotated[Path, typer.Option("--control", exists=True, dir_okay=False)],
    variant: Annotated[
        Path | None,
        typer.Option(
            "--variant",
            exists=True,
            dir_okay=False,
            help="Defaults to the control checkpoint for a search-budget comparison.",
        ),
    ] = None,
    iterations: Annotated[int, typer.Option("--iterations", min=1)] = 128,
    control_iterations: Annotated[
        int | None, typer.Option("--control-iterations", min=1)
    ] = None,
    variant_iterations: Annotated[
        int | None, typer.Option("--variant-iterations", min=1)
    ] = None,
    seed_count: Annotated[
        int, typer.Option("--seed-count", min=1, help="Each seed plays both seats.")
    ] = 16,
    seed: Annotated[int, typer.Option("--seed", min=0, max=2**32 - 1)] = 0,
    threads: Annotated[int, typer.Option("--threads", min=1)] = 1,
    runs_dir: Annotated[Path, typer.Option("--runs-dir")] = Path(".runs"),
    allow_dirty: Annotated[bool, typer.Option("--allow-dirty")] = False,
) -> Path:
    """Evaluate without training; positive score margins favor the variant."""
    control = control.resolve()
    variant = control if variant is None else variant.resolve()
    for path in (control, variant):
        if not path.is_file():
            raise FileNotFoundError(path)
    if type(threads) is not int or threads < 1:
        raise ValueError("threads must be a positive integer")
    settings = evaluation.EvaluationConfig(
        seed_count=seed_count,
        seed=seed,
        iterations=iterations,
        control_iterations=control_iterations,
        variant_iterations=variant_iterations,
    )
    configuration = {
        "name": "checkpoint-comparison",
        "control": str(control),
        "variant": str(variant),
        "seed": seed,
        "evaluation": dataclasses.asdict(settings),
        "execution": {"device": "cpu", "threads": threads},
    }
    recorder = runs.RunRecorder.create(
        root=runs_dir,
        repository=Path(__file__).resolve().parent,
        input_path=Path("evaluation.json"),
        input_bytes=json.dumps(configuration, indent=2).encode(),
        configuration=configuration,
        entrypoint="run_checkpoint_comparison.py:compare",
        invocation=[
            "--control",
            str(control),
            "--variant",
            str(variant),
            "--iterations",
            str(iterations),
            "--control-iterations",
            str(control_iterations or iterations),
            "--variant-iterations",
            str(variant_iterations or iterations),
            "--seed-count",
            str(seed_count),
            "--seed",
            str(seed),
            "--threads",
            str(threads),
            "--runs-dir",
            str(runs_dir.resolve()),
        ]
        + (["--allow-dirty"] if allow_dirty else []),
        allow_dirty=allow_dirty,
    )
    typer.echo(f"Run: {recorder.path}")
    completed = 0
    win_credit = 0.0

    def record_game(record):
        nonlocal completed, win_credit
        completed += 1
        win_credit += record["variant_win_credit"]
        recorder.record_event(
            "evaluation_game_completed",
            progress={"evaluated_games": completed},
            context=record,
        )
        typer.echo(
            f"Game {completed}/{2 * seed_count}: seats={record['seats']} "
            f"scores={record['cumulative_scores']}; "
            f"variant win credit={win_credit / completed:.1%}"
        )

    with recorder:
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(threads)
            report = evaluation.evaluate_checkpoints(
                control, variant, settings, on_game=record_game
            )
        finally:
            torch.set_num_threads(previous_threads)
        path = recorder.path / "comparison.json"
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        recorder.register_artifact(
            path, kind="checkpoint_comparison", progress={"evaluated_games": completed}
        )
        typer.echo(
            f"Variant win credit: {report['variant_win_fraction']:.1%}; "
            f"control-minus-variant margin: {report['control_minus_variant_margin']:+.2f}; "
            f"evaluation: {report['evaluation_seconds']:.1f}s\nReport: {path}"
        )
    return recorder.path


def main():
    typer.run(compare)


if __name__ == "__main__":
    main()
