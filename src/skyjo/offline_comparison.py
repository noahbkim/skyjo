"""Recorded comparisons of ordinary configurations on a single frozen dataset."""

import copy
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import buffer, checkpoint, experiment_config, offline, runs

REPOSITORY = Path(__file__).resolve().parents[2]


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _load_variant(path, index):
    supplied, sources = experiment_config.configuration_sources(path)
    # Offline data and seed are supplied by the comparison, not the self-play recipe.
    supplied = experiment_config.overlay(
        supplied, {"seed": 0, "replay": {"initial_dataset": None, "dataset_id": None}}
    )
    resolved = experiment_config.resolve_configuration(
        supplied, base_directory=path.parent
    )
    return {
        "variant": f"{index:02d}-{path.stem}",
        "configuration": resolved,
        "sources": sources,
    }


def _validate_dataset(replay, configuration):
    expected = configuration["derived"]
    for key, actual in (
        ("spatial_input_shape", replay.spatial_input_buffer.shape[1:]),
        ("non_spatial_input_shape", replay.non_spatial_input_buffer.shape[1:]),
        ("action_mask_shape", replay.action_masks.shape[1:]),
    ):
        if list(actual) != expected[key]:
            raise ValueError(f"Replay {key} does not match configuration")
    for spec in expected["target_specs"]:
        target = replay.target_buffers.get(spec["name"])
        if target is None or list(target.shape[1:]) != spec["shape"]:
            raise ValueError(f"Missing or incompatible replay target: {spec['name']}")


def _record_curve(parent, child, row):
    child.record_event(
        "evaluation",
        progress={
            "optimizer_steps": row["optimizer_steps"],
            "sampled_positions": row["sampled_positions"],
        },
        metrics=row["metrics"],
        context={"split": row["split"]},
    )
    with (parent.path / "curves.jsonl").open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
    csv_path = parent.path / "curves.csv"
    new = not csv_path.exists()
    with csv_path.open("a", newline="") as stream:
        writer = csv.writer(stream)
        fields = (
            "variant",
            "seed",
            "run_id",
            "split",
            "optimizer_steps",
            "sampled_positions",
            "training_seconds",
            "evaluation_seconds",
        )
        if new:
            writer.writerow((*fields, "metric", "value"))
        for key, value in row["metrics"].items():
            writer.writerow((*[row[field] for field in fields], key, value))


def _train_child(
    parent, variant, seed, training, validation, probe, settings, allow_dirty
):
    config = copy.deepcopy(variant["configuration"])
    config.update(seed=seed, name=f"offline-{variant['variant']}-seed-{seed}")
    config["offline"] = settings
    config["experiment"] = {
        "name": "offline-comparison",
        "variant": variant["variant"],
        "suite_run_id": parent.manifest["run_id"],
    }
    last_source = variant["sources"][-1]
    recorder = runs.RunRecorder.create(
        root=parent.path / "runs",
        repository=REPOSITORY,
        input_path=Path(last_source["path"]),
        input_bytes=last_source["content"].encode(),
        configuration=config,
        entrypoint="run_offline_comparison.py:compare",
        invocation=parent.manifest["invocation"],
        allow_dirty=allow_dirty,
    )
    with recorder:
        recorder.record_event(
            "experiment_membership", context={**config["experiment"], "seed": seed}
        )
        _write_json(recorder.path / "input-sources.json", variant["sources"])
        recorder.register_artifact(
            recorder.path / "input-sources.json",
            kind="configuration_sources",
            progress={},
        )
        torch.set_num_threads(config["execution"]["threads_per_worker"])
        trainer = offline.OfflineTrainer.from_configuration(config, seed)
        parameters = sum(p.numel() for p in trainer.model.parameters())
        recorder.record_event("model_created", metrics={"parameter_count": parameters})
        batch_size = config["training"]["batch_size"]
        completed = 0
        training_seconds = evaluation_seconds = diagnostic_seconds = 0.0
        final_metrics = {}

        def evaluate(split, replay, indices=None):
            nonlocal evaluation_seconds
            started = time.perf_counter()
            metrics = trainer.evaluate(
                replay, batch_size=batch_size, indices=indices, diagnostics=True
            )
            evaluation_seconds += time.perf_counter() - started
            _record_curve(
                parent,
                recorder,
                {
                    "variant": variant["variant"],
                    "seed": seed,
                    "run_id": recorder.manifest["run_id"],
                    "split": split,
                    "optimizer_steps": completed,
                    "sampled_positions": completed * batch_size,
                    "training_seconds": training_seconds,
                    "evaluation_seconds": evaluation_seconds,
                    "metrics": metrics,
                },
            )
            return metrics

        while True:
            evaluate("train_probe", training, probe)
            final_metrics = evaluate("validation", validation)
            if completed == settings["steps"]:
                break
            count = min(settings["evaluation_interval"], settings["steps"] - completed)
            gradients = (
                {}
                if completed == 0 and config["training"]["gradient_diagnostic"]
                else None
            )
            started = time.perf_counter()
            trainer.fit(
                training, steps=count, batch_size=batch_size, gradient_scales=gradients
            )
            elapsed = time.perf_counter() - started
            if gradients:
                diagnostic_seconds += gradients["seconds"]
                elapsed -= gradients["seconds"]
                recorder.record_event(
                    "gradient_diagnostic",
                    progress={"optimizer_steps": 1},
                    metrics=gradients,
                )
            training_seconds += elapsed
            completed += count
        full_training = evaluate("train_full", training)
        progress = checkpoint.TrainingProgress(
            optimizer_steps=completed,
            sampled_positions=completed * batch_size,
            trained_positions=completed * batch_size,
        )
        path = recorder.path / "checkpoints" / "final.pth"
        checkpoint.save_checkpoint(
            path,
            model=trainer.model,
            optimizer=trainer.optimizer,
            configuration=config,
            progress=progress,
            sampling_rng=trainer.sampling_rng,
        )
        artifact_id = recorder.register_artifact(
            path,
            kind="checkpoint",
            progress={"optimizer_steps": completed},
            metadata={"role": "final"},
        )
        result = {
            "variant": variant["variant"],
            "seed": seed,
            "run_id": recorder.manifest["run_id"],
            "path": str(recorder.path),
            "configuration": config,
            "parameter_count": parameters,
            "optimizer_steps": completed,
            "sampled_positions": completed * batch_size,
            "training_seconds": training_seconds,
            "evaluation_seconds": evaluation_seconds,
            "gradient_diagnostic_seconds": diagnostic_seconds,
            "elapsed_seconds": time.perf_counter() - recorder.started,
            "checkpoint": {"path": str(path), "artifact_id": artifact_id},
            "validation": final_metrics,
            "train_full": full_training,
        }
        _write_json(recorder.path / "result.json", result)
        recorder.register_artifact(
            recorder.path / "result.json",
            kind="offline_result",
            progress={"optimizer_steps": completed},
        )
        recorder.record_event(
            "training_completed",
            progress={
                "optimizer_steps": completed,
                "sampled_positions": completed * batch_size,
            },
            metrics={
                "training_seconds": training_seconds,
                "evaluation_seconds": evaluation_seconds,
            },
        )
        print(
            f"{variant['variant']} seed={seed}: {completed} steps, {completed * batch_size} positions, "
            f"train={training_seconds:.2f}s eval={evaluation_seconds:.2f}s; validation "
            f"value={final_metrics['outcome_value_loss']:.5f} policy={final_metrics['policy_loss']:.5f}",
            flush=True,
        )
        return result


def summarize(completed, control):
    pairs = []
    for child in completed:
        if child["variant"] == control:
            continue
        baseline = next(
            c
            for c in completed
            if c["variant"] == control and c["seed"] == child["seed"]
        )
        # Only common, unweighted errors are comparable across loss definitions.
        shared = child["validation"].keys() & baseline["validation"].keys()
        keys = [
            k
            for k in sorted(shared)
            if k != "total_loss"
            and "weighted" not in k
            and (k.endswith(("_loss", "_mae_points")) or k == "policy_kl")
        ]
        delta = {
            k: child["validation"][k] - baseline["validation"][k]
            for k in keys
            if child["validation"][k] is not None
            and baseline["validation"][k] is not None
        }
        pairs.append(
            {
                "variant": child["variant"],
                "seed": child["seed"],
                "control_run_id": baseline["run_id"],
                "variant_run_id": child["run_id"],
                "variant_minus_control": delta,
            }
        )
    averages = {}
    for variant in dict.fromkeys(c["variant"] for c in completed):
        selected = [c for c in completed if c["variant"] == variant]
        metrics = {}
        for key in sorted(set().union(*(c["validation"].keys() for c in selected))):
            values = [
                c["validation"][key]
                for c in selected
                if c["validation"].get(key) is not None
            ]
            if values:
                metrics[key] = {
                    "mean": float(np.mean(values)),
                    "seed_count": len(values),
                }
        paired = [p["variant_minus_control"] for p in pairs if p["variant"] == variant]
        deltas = (
            {
                key: float(np.mean([p[key] for p in paired if key in p]))
                for key in set().union(*paired)
            }
            if paired
            else {}
        )
        averages[variant] = {
            "validation": metrics,
            "paired_differences": deltas,
            **{
                key: float(np.mean([c[key] for c in selected]))
                for key in (
                    "training_seconds",
                    "evaluation_seconds",
                    "elapsed_seconds",
                    "sampled_positions",
                    "parameter_count",
                    "optimizer_steps",
                )
            },
        }
    return {
        "runs": completed,
        "paired_differences": pairs,
        "averages": averages,
        "interpretation": "Fixed-replay learning, not playing strength. Steps are matched; exposure and compute may differ. Weighted totals across different objectives are not rankings.",
    }


def launch_comparison(
    configs,
    dataset,
    runs_dir=Path(".runs"),
    *,
    seeds=(0, 1, 2),
    steps=2000,
    validation_fraction=0.1,
    split_seed=0,
    evaluation_interval=200,
    allow_dirty=False,
):
    configs = [Path(path).resolve() for path in configs]
    if len(configs) < 2:
        raise ValueError("At least two --config arguments are required")
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)
    ):
        raise ValueError("Seeds must be distinct uint32 integers")
    if (
        type(steps) is not int
        or type(evaluation_interval) is not int
        or type(split_seed) is not int
        or steps < 0
        or evaluation_interval < 1
        or not 0 < validation_fraction < 1
        or not 0 <= split_seed < 2**32
    ):
        raise ValueError(
            "Invalid step budget, evaluation interval, validation fraction, or split seed"
        )
    variants = [_load_variant(path, i) for i, path in enumerate(configs)]
    dataset = Path(dataset).resolve()
    manifest_path = dataset / buffer.MANIFEST_FILE
    before = manifest_path.read_bytes()
    replay = buffer.ReplayBuffer.load(dataset)
    if before != manifest_path.read_bytes():
        raise ValueError("Replay changed while loading; retry with a stable dataset")
    source_id = replay.dataset_id
    for variant in variants:
        _validate_dataset(replay, variant["configuration"])
    training, validation = replay.split_by_game(validation_fraction, seed=split_seed)
    probe = np.random.default_rng(
        np.random.SeedSequence([split_seed, 0x50524F42])
    ).choice(len(training), min(8192, len(training)), replace=False)
    settings = {
        "steps": steps,
        "seeds": list(seeds),
        "validation_fraction": validation_fraction,
        "split_seed": split_seed,
        "evaluation_interval": evaluation_interval,
        "source_dataset": str(dataset),
        "source_dataset_id": source_id,
    }
    parent = runs.RunRecorder.create(
        root=Path(runs_dir),
        repository=REPOSITORY,
        input_path=Path("comparison.json"),
        input_bytes=json.dumps(
            {"configs": [str(p) for p in configs], **settings}
        ).encode(),
        configuration={"name": "offline-comparison", "variants": variants, **settings},
        entrypoint="run_offline_comparison.py:compare",
        invocation=[arg for path in configs for arg in ("--config", str(path))]
        + [
            "--dataset",
            str(dataset),
            "--steps",
            str(steps),
            "--seeds",
            ",".join(map(str, seeds)),
            "--validation-fraction",
            str(validation_fraction),
            "--split-seed",
            str(split_seed),
            "--evaluation-interval",
            str(evaluation_interval),
            "--runs-dir",
            str(Path(runs_dir).resolve()),
            *(["--allow-dirty"] if allow_dirty else []),
        ],
        allow_dirty=allow_dirty,
    )
    previous_threads = torch.get_num_threads()
    with parent, offline.preserve_rng():
        try:
            snapshot = replay.save(
                parent.path / "data" / "replay",
                generation_metadata={
                    "source_dataset_id": source_id,
                    "source_path": str(dataset),
                },
            )
            snapshot_artifact = parent.register_artifact(
                snapshot, kind="replay_dataset", progress={}
            )
            split = {
                "training_game_indices": list(training.game_indices),
                "validation_game_indices": list(validation.game_indices),
                "training_probe_indices": probe.tolist(),
                "source_manifest": json.loads(before),
            }
            _write_json(parent.path / "split.json", split)
            split_artifact = parent.register_artifact(
                parent.path / "split.json", kind="dataset_split", progress={}
            )
            settings.update(
                snapshot_path=str(snapshot),
                snapshot_dataset_id=replay.dataset_id,
                snapshot_artifact_id=snapshot_artifact,
                split_artifact_id=split_artifact,
            )
            completed = []
            for seed in seeds:
                for variant in variants:
                    parent.record_event(
                        "child_started",
                        context={"variant": variant["variant"], "seed": seed},
                    )
                    result = _train_child(
                        parent,
                        variant,
                        seed,
                        training,
                        validation,
                        probe,
                        settings,
                        allow_dirty,
                    )
                    completed.append(result)
                    parent.record_event(
                        "child_completed",
                        context={
                            "run_id": result["run_id"],
                            "variant": result["variant"],
                            "seed": seed,
                        },
                    )
                    _write_json(
                        parent.path / "comparison.json",
                        summarize(completed, variants[0]["variant"]),
                    )
            for filename, kind in (
                ("comparison.json", "offline_comparison"),
                ("curves.jsonl", "learning_curves"),
                ("curves.csv", "learning_curves"),
            ):
                parent.register_artifact(parent.path / filename, kind=kind, progress={})
        finally:
            torch.set_num_threads(previous_threads)
    return parent.path
