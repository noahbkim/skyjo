"""Status: experimental
Purpose: Run two capped continuations and compare with a historical control.
Promote when: This pilot becomes a recurring experiment suite.
"""

import itertools
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

from skyjo.experiments import (
    budget_comparison,
    contestants,
    evaluation,
    experiment_config,
    runs,
)
from skyjo.learning import continuation
from skyjo.learning.learner import Learner
from skyjo.search.configuration import SearchConfig

ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def compare(
    parent_run: Annotated[Path, typer.Option()],
    historical_run: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()],
    seconds_per_arm: Annotated[float, typer.Option(min=1)] = 3000,
    workers: Annotated[int, typer.Option(min=1)] = 8,
    games_per_iteration: Annotated[int, typer.Option(min=1)] = 256,
    evaluation_seeds: Annotated[int, typer.Option(min=1)] = 32,
    prepare_only: bool = False,
) -> None:
    parent_run, historical_run, output = (
        p.resolve() for p in (parent_run, historical_run, output)
    )
    provenance = runs.git_provenance(ROOT, allow_dirty=False)
    # Two replay snapshots plus one atomic replacement, with working room.
    if shutil.disk_usage(output.parent).free < 14 * 1024**3:
        raise ValueError("This full-replay pilot needs at least 14 GiB free disk")
    output.mkdir(exist_ok=False)
    source = parent_run / "checkpoints/checkpoint_000119.pth"
    adapted = output / "parent-adapted.pth"
    adaptation = budget_comparison.adapt_parent(source, adapted, sampling_seed=0)
    group_initial = output / "initial-groups.pth"
    budget_comparison.group_initial_checkpoint(adapted, group_initial)
    historical_config = json.loads(
        (historical_run / "resolved-config.json").read_text()
    )
    common = experiment_config.overlay(
        historical_config,
        {
            "initial_checkpoint": str(adapted),
            "model": {"policy_normalization": "groups"},
            "replay": {"initial_dataset": str(parent_run / "data/replay")},
            "selfplay": {
                "games_per_iteration": games_per_iteration,
                "games_per_task": 1,
            },
            "search": {
                "group_root_noise": True,
                "merge_symmetric_actions": True,
                "uniform_group_priors": False,
                "target_strategy": "visits",
                "valuation": "current_mixture",
                "fpu_mode": "parent",
            },
            "budget": {
                "iterations": 0,
                "max_seconds": seconds_per_arm,
                "checkpoint_interval": 1,
            },
            "execution": {"workers": workers, "threads_per_worker": 1},
            "validation": {"concept_interval": 0},
            "logging": {"progress_interval_seconds": 60.0},
        },
    )
    common.pop("derived", None)
    configurations = {}
    for name, full, fast, probability in [
        ("fixed", 32, 0, 1.0),
        ("randomized", 128, 16, 0.25),
    ]:
        config = experiment_config.overlay(
            common,
            {
                "name": f"group-{name}-budget-from119",
                "description": f"Group-aware {name} search budget continuation.",
                "notes": "See parent experiment plan.json for historical-control limitations and RNG adaptation.",
                "search": {"iterations": full},
                "selfplay": {
                    "fast_search_iterations": fast,
                    "full_search_probability": probability,
                },
            },
        )
        config = experiment_config.resolve_configuration(config, base_directory=output)
        # Validate parent parameters AND Adam moments before either long run.
        parent = continuation.load(adapted, config)
        learner = Learner.from_configuration(config, config["seed"])
        parent.restore(
            learner.model, learner.optimizer, sampling_rng=learner.sampling_rng
        )
        path = output / f"{name}.json"
        write_json(path, config)
        configurations[name] = path
    plan = {
        "status": "prepared",
        "git": provenance,
        "adaptation": adaptation,
        "parent_run": str(parent_run),
        "historical_run": str(historical_run),
        "seconds_per_arm": seconds_per_arm,
        "workers": workers,
        "evaluation_seeds": evaluation_seeds,
        "evaluation_iterations": 32,
        "limitations": [
            "One continuation seed; historical arm differs in code revision and update cadence.",
            "Historical checkpoint selected nearest mean actual elapsed time of the new arms, without strength results.",
            "Time caps are soft: complete the active 256-game iteration and save.",
            "Inherited replay contains old targets; randomized search refreshes fewer rows and receives fewer optimizer updates at replay ratio four.",
            "Group-aware uses learned priors, not fully uniform search priors.",
        ],
    }
    write_json(output / "plan.json", plan)
    if prepare_only:
        return
    results = {}
    try:
        for name, config in configurations.items():
            plan.update(status="training", active_arm=name)
            write_json(output / "plan.json", plan)
            with (output / f"{name}.log").open("w") as log:
                subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "distributed_main.py"),
                        "--config",
                        str(config),
                        "--runs-dir",
                        str(output / name),
                    ],
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            result_path = next((output / name).glob("*/result.json"))
            results[name] = json.loads(result_path.read_text())
        mean_seconds = (
            sum(r["timings"]["time/run_seconds"] for r in results.values()) / 2
        )
        historical = budget_comparison.historical_checkpoint(
            historical_run, mean_seconds
        )
        plan.update(status="evaluating", historical=historical, results=results)
        write_json(output / "plan.json", plan)
        checkpoints = {
            "historical": historical["checkpoint"],
            **{
                name: result["checkpoint"]["absolute_path"]
                for name, result in results.items()
            },
        }
        pairs = [
            (a, b, checkpoints[a], checkpoints[b])
            for a, b in itertools.combinations(checkpoints, 2)
        ]
        pairs += [
            (
                "initial",
                name,
                str(source if name == "historical" else group_initial),
                path,
            )
            for name, path in checkpoints.items()
        ]
        reports = []
        for a, b, control, variant in pairs:
            settings = {
                "iterations": 32,
                "temperature": 0.0,
                "search": SearchConfig(
                    c_puct=1.0, fpu_reduction=0.0, dirichlet_epsilon=0.0
                ),
            }
            report = evaluation.launch_comparison(
                evaluation.EvaluationConfig(
                    contestants.ContestantConfig(checkpoint=control, **settings),
                    contestants.ContestantConfig(checkpoint=variant, **settings),
                    seed_count=evaluation_seeds,
                    seed=20261016,
                    workers=workers,
                    threads=1,
                ),
                output / "faceoffs",
                repository=ROOT,
            )
            reports.append(
                {"control": a, "variant": b, "report": str(report / "comparison.json")}
            )
            write_json(output / "comparisons.json", reports)
        plan.update(status="completed", comparisons=reports)
        write_json(output / "plan.json", plan)
    except BaseException as error:
        plan.update(status="failed", error=repr(error))
        write_json(output / "plan.json", plan)
        raise


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    typer.run(compare)


if __name__ == "__main__":
    main()
