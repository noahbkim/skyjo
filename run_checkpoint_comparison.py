"""Status: manual
Purpose: Compare frozen checkpoints or search settings in seat-balanced full games.
Promote when: Additional evaluation logic belongs in skyjo.evaluation.
"""

import dataclasses
import json
from pathlib import Path
from typing import Annotated

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
    control_boundary_samples: Annotated[
        int, typer.Option("--control-boundary-samples", min=1)
    ] = 1,
    variant_boundary_samples: Annotated[
        int, typer.Option("--variant-boundary-samples", min=1)
    ] = 1,
    control_boundary_value_checkpoint: Annotated[
        Path | None,
        typer.Option("--control-boundary-value-checkpoint", exists=True, dir_okay=False),
    ] = None,
    variant_boundary_value_checkpoint: Annotated[
        Path | None,
        typer.Option("--variant-boundary-value-checkpoint", exists=True, dir_okay=False),
    ] = None,
    seed_count: Annotated[
        int, typer.Option("--seed-count", min=1, help="Each seed plays both seats.")
    ] = 16,
    seed: Annotated[int, typer.Option("--seed", min=0, max=2**32 - 1)] = 0,
    threads: Annotated[
        int, typer.Option("--threads", min=1, help="PyTorch CPU threads per worker.")
    ] = 1,
    runs_dir: Annotated[Path, typer.Option("--runs-dir")] = Path(".runs"),
    allow_dirty: Annotated[bool, typer.Option("--allow-dirty")] = False,
    workers: Annotated[
        int, typer.Option("--workers", min=1, help="Parallel game worker processes.")
    ] = 1,
    control_policy_only: Annotated[
        bool, typer.Option("--control-policy-only", help="Control plays its highest-scored legal policy action without search.")
    ] = False,
    variant_policy_only: Annotated[
        bool, typer.Option("--variant-policy-only", help="Variant plays its highest-scored legal policy action without search.")
    ] = False,
    control_merge_symmetric_actions: Annotated[
        bool,
        typer.Option(
            "--control-merge-symmetric-actions/--no-control-merge-symmetric-actions",
            help="Share search statistics for equivalent control actions when safe.",
        ),
    ] = True,
    variant_merge_symmetric_actions: Annotated[
        bool,
        typer.Option(
            "--variant-merge-symmetric-actions/--no-variant-merge-symmetric-actions",
            help="Share search statistics for equivalent variant actions when safe.",
        ),
    ] = True,
) -> Path:
    """Evaluate without training; positive score margins favor the variant."""
    control = control.resolve()
    variant = control if variant is None else variant.resolve()
    for path in (control, variant):
        if not path.is_file():
            raise FileNotFoundError(path)
    if type(threads) is not int or threads < 1:
        raise ValueError("threads must be a positive integer")
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    settings = evaluation.EvaluationConfig(
        seed_count=seed_count,
        seed=seed,
        iterations=iterations,
        control_iterations=control_iterations,
        variant_iterations=variant_iterations,
        control_boundary_samples=control_boundary_samples,
        variant_boundary_samples=variant_boundary_samples,
        control_boundary_value_checkpoint=(
            str(control_boundary_value_checkpoint.resolve())
            if control_boundary_value_checkpoint is not None else None
        ),
        variant_boundary_value_checkpoint=(
            str(variant_boundary_value_checkpoint.resolve())
            if variant_boundary_value_checkpoint is not None else None
        ),
        control_policy_only=control_policy_only,
        variant_policy_only=variant_policy_only,
        control_merge_symmetric_actions=control_merge_symmetric_actions,
        variant_merge_symmetric_actions=variant_merge_symmetric_actions,
    )
    configuration = {
        "name": "checkpoint-comparison",
        "control": str(control),
        "variant": str(variant),
        "seed": seed,
        "evaluation": dataclasses.asdict(settings),
        "execution": {"device": "cpu", "threads": threads, "workers": workers},
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
            "--control-boundary-samples",
            str(control_boundary_samples),
            "--variant-boundary-samples",
            str(variant_boundary_samples),
            "--seed-count",
            str(seed_count),
            "--seed",
            str(seed),
            "--threads",
            str(threads),
            "--workers",
            str(workers),
            "--runs-dir",
            str(runs_dir.resolve()),
        ] + [
            argument
            for name in ("control", "variant")
            if (path := getattr(settings, f"{name}_boundary_value_checkpoint")) is not None
            for argument in (f"--{name}-boundary-value-checkpoint", path)
        ]
        + [
            (f"--{name}-merge-symmetric-actions"
             if getattr(settings, f"{name}_merge_symmetric_actions")
             else f"--no-{name}-merge-symmetric-actions")
            for name in ("control", "variant")
        ]
        + (["--control-policy-only"] if control_policy_only else [])
        + (["--variant-policy-only"] if variant_policy_only else [])
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
        report = evaluation.evaluate_checkpoints(
            control, variant, settings, on_game=record_game,
            workers=workers, threads=threads,
        )
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
