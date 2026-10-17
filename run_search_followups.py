"""Status: experimental
Purpose: Run the loss-weight sweep and matched visit-versus-Q faceoffs.
Promote when: These ablations become recurring training validation.
"""

import dataclasses
import json
import logging
from pathlib import Path
from typing import Annotated

import typer


def followups(
    plan: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()],
    allow_dirty: bool = False,
):
    from skyjo.experiments import contestants, evaluation, runs
    from skyjo.experiments.learning_loop_diagnostic import BASE
    from skyjo.experiments.loss_weight_sweep import run_sweep

    settings = json.loads(plan.read_text())
    if any(
        settings[key] < 2 or settings[key] % 2 for key in ("full_games", "round_games")
    ):
        raise ValueError(
            "Faceoffs require positive even game counts for balanced seats"
        )
    root = Path(__file__).resolve().parent
    provenance = runs.git_provenance(root, allow_dirty=allow_dirty)
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_text(json.dumps(settings, indent=2) + "\n")
    manifest = {
        "status": "running",
        "started_at": runs.utc_now(),
        "git": provenance,
        "checkpoint_hashes": {
            name: runs.file_digest(Path(settings[name]))
            for name in ("fit_checkpoint", "acting_checkpoint")
        },
        "faceoffs": [],
    }

    def save():
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    try:
        manifest["stage"] = "loss_weight_sweep"
        save()
        run_sweep(settings, output / "loss-sweep")
        control = contestants.ContestantConfig(
            checkpoint=settings["acting_checkpoint"],
            iterations=settings["search_iterations"],
            search=BASE,
        )
        variant = dataclasses.replace(
            control, search=dataclasses.replace(BASE, action_strategy="value")
        )
        for name, first_round, count in (
            ("full-game", False, settings["full_games"]),
            ("first-round", True, settings["round_games"]),
        ):
            manifest["stage"] = name
            save()
            configuration = evaluation.EvaluationConfig(
                control,
                variant,
                seed_count=count // 2,
                seed=settings["faceoff_seed"],
                workers=settings["workers"],
                threads=1,
                first_round=first_round,
            )
            path = evaluation.launch_comparison(
                configuration, output / name, repository=root, allow_dirty=allow_dirty
            )
            manifest["faceoffs"].append(
                {"name": name, "report": str(path / "comparison.json")}
            )
            save()
        manifest.update(status="completed", completed_at=runs.utc_now())
    except BaseException as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        save()
    return output


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    typer.run(followups)


if __name__ == "__main__":
    main()
