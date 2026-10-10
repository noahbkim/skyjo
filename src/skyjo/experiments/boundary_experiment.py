"""Recorded offline comparison of logistic and neural round-boundary values."""

from __future__ import annotations

from skyjo.analytics.reports import boundary_value_report

import dataclasses
import json
import math
import shutil
import tomllib
from pathlib import Path

import numpy as np
import torch

from skyjo.learning import boundary_data
from skyjo.learning import boundary_value
from skyjo.experiments import runs

REPOSITORY = Path(__file__).resolve().parents[3]
KINDS = ("logistic", "mlp")


@dataclasses.dataclass(frozen=True)
class Settings:
    seeds: tuple[int, ...] = (0, 1, 2)
    split_seed: int = 20261006
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    epochs: int = 200
    batch_size: int = 256
    learn_rate: float = 0.003
    weight_decay: float = 0.0001
    hidden_width: int = 32
    bootstrap_samples: int = 2000

    def __post_init__(self):
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Provide distinct training seeds")
        if any(
            type(seed) is not int or not 0 <= seed < 2**32
            for seed in (*self.seeds, self.split_seed)
        ):
            raise ValueError("Seeds must be uint32 integers")
        for name in ("epochs", "batch_size", "hidden_width", "bootstrap_samples"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not (
            0 < self.validation_fraction < 1
            and 0 < self.test_fraction < 1
            and self.validation_fraction + self.test_fraction < 1
        ):
            raise ValueError("Validation and test fractions must leave training games")
        if not math.isfinite(self.learn_rate) or self.learn_rate <= 0:
            raise ValueError("learn_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def paired_game_interval(
    differences: np.ndarray, game_ids: np.ndarray, *, samples: int, seed: int
) -> dict:
    """Bootstrap whole games, preserving the boundary-weighted point estimate."""
    _, groups = np.unique(game_ids, return_inverse=True)
    counts = np.bincount(groups)
    totals = np.bincount(groups, weights=differences)
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples)
    for index in range(samples):
        selected = rng.integers(0, len(counts), size=len(counts))
        estimates[index] = totals[selected].sum() / counts[selected].sum()
    return {
        "mean": float(np.mean(differences)),
        "ci95": np.quantile(estimates, [0.025, 0.975]).tolist(),
        "bootstrap_games": len(counts),
        "bootstrap_samples": samples,
        "interpretation": "Paired game bootstrap on seed-mean per-boundary errors; conditional on this split and these fitted models, not independent training replications.",
    }


def launch_experiment(
    source_run: Path,
    config: Path,
    runs_dir: Path = Path(".runs"),
    *,
    allow_dirty: bool = False,
) -> Path:
    """Snapshot observed boundaries and compare two fitted probability models."""
    source_run, config = Path(source_run).resolve(), Path(config).resolve()
    config_bytes = config.read_bytes()
    settings = Settings(**tomllib.loads(config_bytes.decode()))
    dataset = boundary_data.load_boundaries(
        sorted((source_run / "metrics").glob("rounds-*.jsonl"))
    )
    splits = boundary_data.split_by_game(
        dataset,
        validation_fraction=settings.validation_fraction,
        test_fraction=settings.test_fraction,
        seed=settings.split_seed,
    )
    configuration = {
        "name": "boundary-value",
        "source_run": str(source_run),
        "settings": dataclasses.asdict(settings),
        "execution": {"device": "cpu", "threads": 1},
        "players": int(dataset.scores.shape[1]),
        "features": "cumulative scores / 100, in next-starter order",
        "target": "observed eventual winner credit, ties shared",
        "selection": "minimum validation value MSE, earliest epoch on ties",
    }
    recorder = runs.RunRecorder.create(
        root=Path(runs_dir),
        repository=REPOSITORY,
        input_path=config,
        input_bytes=config_bytes,
        configuration=configuration,
        entrypoint="run_boundary_value_experiment.py:experiment",
        invocation=[
            "--source-run",
            str(source_run),
            "--config",
            str(config),
            "--runs-dir",
            str(Path(runs_dir).resolve()),
            *(["--allow-dirty"] if allow_dirty else []),
        ],
        allow_dirty=allow_dirty,
    )
    previous_threads = torch.get_num_threads()
    with recorder:
        try:
            torch.set_num_threads(1)
            return _run_comparison(recorder, dataset, splits, settings)
        finally:
            torch.set_num_threads(previous_threads)


def _run_comparison(recorder, dataset, splits, settings) -> Path:
    def artifact(name, kind):
        recorder.register_artifact(recorder.path / name, kind=kind, progress={})

    np.savez_compressed(
        recorder.path / "data/boundaries.npz",
        scores=dataset.scores,
        targets=dataset.targets,
        game_ids=dataset.game_ids,
        round_numbers=dataset.round_numbers,
        iterations=dataset.iterations,
    )
    artifact("data/boundaries.npz", "boundary_dataset")
    _write_json(recorder.path / "sources.json", dataset.sources)
    artifact("sources.json", "source_provenance")
    split_manifest = {
        name: {
            "row_indices": indices.tolist(),
            "game_ids": np.unique(dataset.game_ids[indices]).tolist(),
        }
        for name, indices in splits.items()
    }
    _write_json(recorder.path / "split.json", split_manifest)
    artifact("split.json", "dataset_split")
    for relative in (
        "run_boundary_value_experiment.py",
        "src/skyjo/analytics/reports.py",
        "src/skyjo/learning/boundary_data.py",
        "src/skyjo/learning/boundary_value.py",
        "src/skyjo/experiments/boundary_experiment.py",
    ):
        destination = recorder.path / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPOSITORY / relative, destination)
        artifact(str(destination.relative_to(recorder.path)), "experiment_source")
    test = splits["test"]
    test_predictions = {}
    results = []
    print(f"Boundary experiment: {recorder.path}", flush=True)
    for seed in settings.seeds:
        for kind in KINDS:
            fit = boundary_value.fit_model(
                dataset.scores,
                dataset.targets,
                splits["train"],
                splits["validation"],
                kind=kind,
                seed=seed,
                epochs=settings.epochs,
                batch_size=settings.batch_size,
                learn_rate=settings.learn_rate,
                weight_decay=settings.weight_decay,
                hidden_width=settings.hidden_width,
            )
            metrics = {}
            for name, indices in splits.items():
                predictions = boundary_value.predict(fit.model, dataset.scores[indices])
                metrics[name] = boundary_value.probability_metrics(
                    predictions, dataset.targets[indices], dataset.scores[indices]
                )
                if name == "test":
                    test_predictions[f"{kind}_{seed}"] = predictions
            filename = f"checkpoints/{kind}-{seed}.pth"
            torch.save(
                {
                    "format": "skyjo.boundary-value",
                    "version": 1,
                    "kind": kind,
                    "players": int(dataset.scores.shape[1]),
                    "hidden_width": settings.hidden_width,
                    "score_scale": 100.0,
                    "input_order": "next starter first, then cyclic seat order",
                    "model_state_dict": fit.model.state_dict(),
                    "best_epoch": fit.best_epoch,
                    "seed": seed,
                },
                recorder.path / filename,
            )
            artifact(filename, "boundary_checkpoint")
            with (recorder.path / "curves.jsonl").open("a") as stream:
                for row in fit.history:
                    stream.write(
                        json.dumps({"kind": kind, "seed": seed, **row}, allow_nan=False)
                        + "\n"
                    )
            result = {
                "kind": kind,
                "seed": seed,
                "checkpoint": filename,
                "best_epoch": fit.best_epoch,
                "optimizer_steps": fit.optimizer_steps,
                "training_seconds": fit.training_seconds,
                "parameter_count": sum(p.numel() for p in fit.model.parameters()),
                "metrics": metrics,
            }
            results.append(result)
            recorder.record_event(
                "model_completed",
                context={"kind": kind, "seed": seed},
                metrics={
                    "best_epoch": fit.best_epoch,
                    "test_value_mse": metrics["test"]["value_mse"],
                },
            )
            print(
                f"{kind} seed={seed}: best epoch={fit.best_epoch}, test MSE={metrics['test']['value_mse']:.6f}",
                flush=True,
            )
    artifact("curves.jsonl", "learning_curves")
    np.savez_compressed(
        recorder.path / "predictions.npz",
        scores=dataset.scores[test],
        targets=dataset.targets[test],
        game_ids=dataset.game_ids[test],
        **test_predictions,
    )
    artifact("predictions.npz", "held_out_predictions")
    metric_names = ("value_mse", "brier_score", "cross_entropy", "ece")
    averages = {
        kind: {
            name: float(
                np.mean(
                    [r["metrics"]["test"][name] for r in results if r["kind"] == kind]
                )
            )
            for name in metric_names
        }
        for kind in KINDS
    }
    uniform = boundary_value.probability_metrics(
        np.full_like(dataset.targets[test], 1 / dataset.targets.shape[1]),
        dataset.targets[test],
        dataset.scores[test],
    )
    averages = {"uniform": {name: uniform[name] for name in metric_names}, **averages}
    differences = np.mean(
        [
            np.mean(
                (test_predictions[f"mlp_{seed}"] - dataset.targets[test]) ** 2, axis=1
            )
            - np.mean(
                (test_predictions[f"logistic_{seed}"] - dataset.targets[test]) ** 2,
                axis=1,
            )
            for seed in settings.seeds
        ],
        axis=0,
    )
    report = {
        "dataset": dataset.summary,
        "splits": {
            name: {
                "games": len(info["game_ids"]),
                "boundaries": len(info["row_indices"]),
            }
            for name, info in split_manifest.items()
        },
        "models": results,
        "test_averages": averages,
        "mlp_minus_logistic_test_mse": paired_game_interval(
            differences,
            dataset.game_ids[test],
            samples=settings.bootstrap_samples,
            seed=settings.split_seed,
        ),
    }
    _write_json(recorder.path / "results.json", report)
    artifact("results.json", "boundary_comparison")
    (recorder.path / "report.md").write_text(boundary_value_report(report))
    artifact("report.md", "experiment_report")
    return recorder.path
