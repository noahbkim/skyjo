"""Thin complete-run suite orchestration; all training belongs to the ordinary runner."""

from __future__ import annotations

import copy
import json
import pathlib
import time
import tomllib

import numpy as np

from skyjo.experiments import evaluation, contestants
from skyjo.experiments import experiment_config
from skyjo.experiments import runs

overlay = experiment_config.overlay


def load_suite(path: pathlib.Path) -> dict:
    """Resolve and validate every child before recording or starting any training."""
    path = path.resolve()
    supplied = tomllib.loads(path.read_text())
    unknown = supplied.keys() - {
        "name",
        "baseline",
        "variants",
        "seeds",
        "evaluation",
        "control",
    }
    if unknown:
        raise ValueError(f"Unknown suite settings: {sorted(unknown)}")
    name, seeds, variants = supplied["name"], supplied["seeds"], supplied["variants"]
    if (
        not isinstance(name, str)
        or not name
        or not isinstance(variants, dict)
        or not variants
    ):
        raise ValueError("Suite requires a name and named variants")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(type(seed) is not int for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("Suite seeds must be distinct integers")
    control = supplied.get("control", "control")
    if control not in variants:
        raise ValueError("Control variant is missing")
    evaluation_settings = supplied.get("evaluation", {})
    if not isinstance(evaluation_settings, dict):
        raise ValueError("Suite evaluation settings must be a table")
    unknown = evaluation_settings.keys() - {"seed_count", "seed", "iterations"}
    if unknown:
        raise ValueError(f"Unknown suite evaluation settings: {sorted(unknown)}")
    settings = {"seed_count": 32, "seed": 0, "iterations": 128, **evaluation_settings}
    if any(
        type(settings[k]) is not int or settings[k] < 1
        for k in ("seed_count", "iterations")
    ):
        raise ValueError("Evaluation seed count and iterations must be positive")
    if type(settings["seed"]) is not int or not 0 <= settings["seed"] <= 2**32 - 1:
        raise ValueError("Evaluation seed must fit a uint32")
    baseline_path = (path.parent / supplied["baseline"]).resolve()
    _, baseline = experiment_config.load_configuration(baseline_path)
    baseline.pop("derived")
    baseline = experiment_config.resolve_paths(baseline, baseline_path.parent)
    children = []
    for seed in seeds:
        for variant, overrides in variants.items():
            if not isinstance(overrides, dict):
                raise ValueError(f"Variant {variant!r} overrides must be a table")
            config = overlay(
                baseline, experiment_config.resolve_paths(overrides, path.parent)
            )
            config.update(seed=seed, name=f"{name}-{variant}-seed-{seed}")
            resolved = experiment_config.resolve_configuration(
                config, base_directory=path.parent
            )
            if resolved["players"] != 2:
                raise ValueError(
                    "Suite checkpoint evaluation currently requires two players"
                )
            children.append(
                {"variant": variant, "seed": seed, "configuration": resolved}
            )
    return {
        "name": name,
        "baseline": str(baseline_path),
        "control": control,
        "seeds": seeds,
        "evaluation": settings,
        "children": children,
    }


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def launch_suite(
    config: pathlib.Path,
    runs_dir: pathlib.Path,
    *,
    repository: pathlib.Path,
    allow_dirty=False,
) -> pathlib.Path:
    from skyjo.experiments.selfplay_training import launch

    resolved = load_suite(config)
    recorder = runs.RunRecorder.create(
        root=runs_dir,
        repository=repository,
        input_path=config,
        input_bytes=config.read_bytes(),
        configuration=resolved,
        entrypoint="run_auxiliary_ablation.py:compare",
        invocation=["--config", str(config.resolve()), "--runs-dir", str(runs_dir)],
        allow_dirty=allow_dirty,
    )
    with recorder:
        configs = recorder.path / "configs"
        configs.mkdir()
        completed = []
        for index, child in enumerate(resolved["children"]):
            configuration = copy.deepcopy(child["configuration"])
            membership = {
                "name": resolved["name"],
                "variant": child["variant"],
                "suite_run_id": recorder.manifest["run_id"],
            }
            configuration["experiment"] = membership
            path = configs / f"{index:03d}.json"
            _json(path, configuration)
            recorder.register_artifact(path, kind="child_configuration", progress={})
            recorder.record_event(
                "child_started", context={**membership, "seed": child["seed"]}
            )
            started = time.perf_counter()
            child_result = launch(
                path,
                recorder.path / "runs",
                allow_dirty=allow_dirty,
                repository=repository,
            )
            result = {
                **child,
                "configuration": configuration,
                **child_result.to_dict(),
                "training_elapsed_seconds": time.perf_counter() - started,
            }
            completed.append(result)
            recorder.record_event("child_completed", context=result)
        comparisons = []
        for child in completed:
            if child["variant"] == resolved["control"]:
                continue
            control = next(
                c
                for c in completed
                if c["seed"] == child["seed"] and c["variant"] == resolved["control"]
            )
            options = resolved["evaluation"]
            report = evaluation.evaluate_checkpoints(
                evaluation.EvaluationConfig(
                    contestants.ContestantConfig(
                        checkpoint=control["checkpoint"]["absolute_path"],
                        iterations=options["iterations"],
                    ),
                    contestants.ContestantConfig(
                        checkpoint=child["checkpoint"]["absolute_path"],
                        iterations=options["iterations"],
                    ),
                    seed_count=options["seed_count"],
                    seed=options["seed"],
                )
            ).to_dict()
            report.update(
                variant=child["variant"],
                training_seed=child["seed"],
                control_run_id=control["run_id"],
                variant_run_id=child["run_id"],
                control_checkpoint_artifact_id=control["checkpoint"]["artifact_id"],
                variant_checkpoint_artifact_id=child["checkpoint"]["artifact_id"],
            )
            comparisons.append(report)
            path = recorder.path / f"comparison-{len(comparisons):03d}.json"
            _json(path, report)
            recorder.register_artifact(path, kind="checkpoint_comparison", progress={})
            recorder.record_event(
                "comparison_completed",
                context={"path": str(path)},
                metrics={
                    key: report[key]
                    for key in (
                        "variant_win_fraction",
                        "control_minus_variant_margin",
                        "evaluation_seconds",
                    )
                },
            )
        averages = {
            variant: {
                key: float(
                    np.mean([c[key] for c in comparisons if c["variant"] == variant])
                )
                for key in ("variant_win_fraction", "control_minus_variant_margin")
            }
            for variant in dict.fromkeys(c["variant"] for c in comparisons)
        }
        path = recorder.path / "comparison.json"
        _json(
            path,
            {
                "experiment": resolved["name"],
                "runs": completed,
                "per_seed": comparisons,
                "averages": averages,
                "interpretation": "Report actual training volume and elapsed cost separately; equal iterations do not match compute or sampled positions.",
            },
        )
        recorder.register_artifact(path, kind="experiment_comparison", progress={})
    return recorder.path
