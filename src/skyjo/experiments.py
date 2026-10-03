"""Thin complete-run suite orchestration; all training belongs to the ordinary runner."""

from __future__ import annotations

import copy
import dataclasses
import json
import pathlib
import time
import tomllib

import numpy as np

from . import evaluation, experiment_config, runs


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
    settings = evaluation.EvaluationConfig(**supplied.get("evaluation", {}))
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
        "evaluation": dataclasses.asdict(settings),
        "children": children,
    }


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _child_result(path: pathlib.Path) -> dict:
    manifest = json.loads((path / "run.json").read_text())
    artifacts = [
        json.loads(line) for line in (path / "artifacts.jsonl").read_text().splitlines()
    ]
    final = next(
        a
        for a in reversed(artifacts)
        if a["kind"] == "registered" and a["metadata"].get("role") == "final"
    )
    events = [
        json.loads(line)
        for line in (path / "trajectory.jsonl").read_text().splitlines()
    ]
    last_iteration = next(
        e for e in reversed(events) if e["kind"] == "iteration_completed"
    )
    timings = {
        key: sum(
            e["metrics"].get(key, 0)
            for e in events
            if e["kind"] == "iteration_completed"
        )
        for key in last_iteration["metrics"]
        if key.startswith("time/")
    }
    return {
        "run_id": manifest["run_id"],
        "path": str(path),
        "checkpoint": {**final, "absolute_path": str(path / final["path"])},
        "progress": last_iteration["progress"],
        "timings": timings,
    }


def launch_suite(
    config: pathlib.Path, runs_dir: pathlib.Path, *, allow_dirty=False
) -> pathlib.Path:
    from distributed_main import launch

    resolved = load_suite(config)
    recorder = runs.RunRecorder.create(
        root=runs_dir,
        repository=pathlib.Path(__file__).resolve().parents[2],
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
            child_path = launch(path, recorder.path / "runs", allow_dirty)
            result = {
                **child,
                "configuration": configuration,
                **_child_result(child_path),
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
            report = evaluation.evaluate_checkpoints(
                pathlib.Path(control["checkpoint"]["absolute_path"]),
                pathlib.Path(child["checkpoint"]["absolute_path"]),
                evaluation.EvaluationConfig(**resolved["evaluation"]),
            )
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
