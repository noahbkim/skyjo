"""One recorded smoke run and the uncertainty calculation's weighting contract."""

import json

import numpy as np
import pytest
import torch
from typer.testing import CliRunner
import typer

from run_boundary_value_experiment import experiment
from skyjo.experiments.boundary_experiment import paired_game_interval
from skyjo.learning.boundary_value import BoundaryValueModel, predict


def test_recorded_experiment_is_directly_callable_and_keeps_source_logs(tmp_path):
    source = tmp_path / "source"
    metrics = source / "metrics"
    metrics.mkdir(parents=True)
    records = []
    for game in range(24):
        before = [10 + game, 45 - game]
        final = [100 + game, 80] if game % 2 else [70, 100 + game]
        for number, cumulative in enumerate((before, final), start=1):
            records.append(
                {
                    "run_id": "source",
                    "iteration": 1,
                    "game_index": game,
                    "play_seed": game,
                    "round_number": number,
                    "cumulative_scores": cumulative,
                    "scores": cumulative
                    if number == 1
                    else [a - b for a, b in zip(final, before)],
                    "ending_player": game % 2,
                    "partial_start": False,
                }
            )
    log = metrics / "rounds-000001.jsonl"
    log.write_text("".join(json.dumps(row) + "\n" for row in records))
    original = log.read_bytes()
    config = tmp_path / "tiny.toml"
    config.write_text(
        "seeds = [0]\nepochs = 2\nbatch_size = 8\nbootstrap_samples = 20\n"
    )
    threads = torch.get_num_threads()
    result = experiment(source, config, tmp_path / "runs", allow_dirty=True)

    assert log.read_bytes() == original
    assert torch.get_num_threads() == threads
    assert json.loads((result / "run.json").read_text())["status"] == "completed"
    report = json.loads((result / "results.json").read_text())
    assert report["dataset"]["boundaries"] == 24
    assert (
        report["mlp_minus_logistic_test_mse"]["bootstrap_games"]
        == report["splits"]["test"]["games"]
    )
    snapshot = np.load(result / "data/boundaries.npz", allow_pickle=False)
    np.testing.assert_array_equal(snapshot["targets"].sum(axis=1), np.ones(24))
    predictions = np.load(result / "predictions.npz", allow_pickle=False)
    for model in report["models"]:
        saved = torch.load(result / model["checkpoint"], weights_only=True)
        restored = BoundaryValueModel(
            saved["kind"], saved["players"], saved["hidden_width"]
        )
        restored.load_state_dict(saved["model_state_dict"])
        np.testing.assert_array_equal(
            predict(restored, predictions["scores"]),
            predictions[f"{model['kind']}_{model['seed']}"],
        )
    assert (result / "report.md").is_file()
    assert (result / "source/src/skyjo/learning/boundary_value.py").is_file()
    app = typer.Typer()
    app.command()(experiment)
    help_result = CliRunner().invoke(app, ["--help"])
    assert help_result.exit_code == 0
    assert "source-run" in help_result.stdout


def test_game_bootstrap_preserves_boundary_weighting_and_paired_differences():
    # One game has three boundaries; the point estimate must retain its weight.
    result = paired_game_interval(
        np.array([1.0, 1.0, 1.0, -1.0]),
        np.array(["a", "a", "a", "b"]),
        samples=100,
        seed=0,
    )
    assert result["mean"] == pytest.approx(0.5)
    assert result["bootstrap_games"] == 2
    assert result["ci95"] == [-1.0, 1.0]
