import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import typer
from typer.testing import CliRunner

import run_checkpoint_comparison as cli
from skyjo import checkpoint, evaluation, models
from skyjo.boundary_value import BoundaryValueModel


@pytest.mark.parametrize(
    "mode,boundary_side",
    [("direct", None), ("cli", None), ("interrupted", None),
     ("direct", "variant"), ("cli", "control")],
)
def test_recorded_comparison(tmp_path, monkeypatch, mode, boundary_side):
    settings = {"embedding_dimensions": 4, "global_state_embedding_dimensions": 8}
    model = models.build(settings, players=2, device="cpu")
    source = checkpoint.save_checkpoint(
        tmp_path / "model.pth",
        model=model,
        optimizer=None,
        configuration={"model": settings, "players": 2},
    )
    boundary_options = {}
    boundary_identities = {"control": None, "variant": None}
    if boundary_side is not None:
        boundary_model = BoundaryValueModel("logistic", players=2)
        boundary_path = tmp_path / "boundary.pth"
        torch.save(
            {
                "format": "skyjo.boundary-value", "version": 1,
                "kind": "logistic", "players": 2, "hidden_width": 32,
                "score_scale": 100.0,
                "input_order": "next starter first, then cyclic seat order",
                "model_state_dict": boundary_model.state_dict(),
            },
            boundary_path,
        )
        boundary_options = {
            f"{boundary_side}_boundary_samples": 10,
            f"{boundary_side}_boundary_value_checkpoint": boundary_path,
        }
        boundary_identities[boundary_side] = {
            "path": str(boundary_path),
            "sha256": hashlib.sha256(boundary_path.read_bytes()).hexdigest(),
        }
    calls = 0

    def play_game(agents):
        nonlocal calls
        calls += 1
        seats = ("control", "variant") if calls % 2 else ("variant", "control")
        for side, agent in zip(seats, agents, strict=True):
            assert agent.mcts_iterations == (1 if side == "control" else 2)
            assert agent.mcts_boundary_samples == (10 if side == boundary_side else 1)
            identity = boundary_identities[side]
            assert agent.mcts_boundary_value_checkpoint == (
                None if identity is None else identity["path"]
            )
        if mode == "interrupted" and calls == 2:
            raise KeyboardInterrupt()
        return SimpleNamespace(final_scores=(100, 100), winners=(0, 1))

    monkeypatch.setattr(evaluation.play, "play_game", play_game)
    previous_threads = torch.get_num_threads()
    if mode == "cli":
        app = typer.Typer()
        app.command()(cli.compare)
        result = CliRunner().invoke(
            app,
            [
                "--control",
                str(source),
                "--control-iterations",
                "1",
                "--variant-iterations",
                "2",
                "--seed-count",
                "1",
                "--runs-dir",
                str(tmp_path / "runs"),
                "--allow-dirty",
            ] + [
                argument
                for name, value in boundary_options.items()
                for argument in (f"--{name.replace('_', '-')}", str(value))
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Game 2/2" in result.output
    else:

        def run():
            return cli.compare(
                source,
                control_iterations=1,
                variant_iterations=2,
                seed_count=1,
                runs_dir=tmp_path / "runs",
                allow_dirty=True,
                **boundary_options,
            )

        if mode == "interrupted":
            with pytest.raises(KeyboardInterrupt):
                run()
        else:
            run()
    assert torch.get_num_threads() == previous_threads
    (run_path,) = (tmp_path / "runs").iterdir()
    manifest = json.loads((run_path / "run.json").read_text())
    events = [
        json.loads(line)
        for line in (run_path / "trajectory.jsonl").read_text().splitlines()
    ]
    played = [event for event in events if event["kind"] == "evaluation_game_completed"]
    if mode == "interrupted":
        assert manifest["status"] == "interrupted"
        assert len(played) == 1
        assert not (run_path / "comparison.json").exists()
        return
    assert manifest["status"] == "completed"
    report = json.loads((run_path / "comparison.json").read_text())
    assert report["checkpoints"]["control"] == report["checkpoints"]["variant"]
    assert report["search_by_player"]["control"]["mcts_iterations"] == 1
    assert report["search_by_player"]["variant"]["mcts_iterations"] == 2
    assert report["boundary_value_checkpoints"] == boundary_identities
    for game in report["games"]:
        assert game["boundary_value_checkpoints"] == boundary_identities
    for side, identity in boundary_identities.items():
        search = report["search_by_player"][side]
        assert search["mcts_boundary_samples"] == (10 if side == boundary_side else 1)
        assert search["mcts_boundary_value_checkpoint"] == (
            None if identity is None else identity["path"]
        )
    assert len(played) == 2
    artifacts = [
        json.loads(line)
        for line in (run_path / "artifacts.jsonl").read_text().splitlines()
    ]
    assert any(a["path"] == "comparison.json" for a in artifacts)


@pytest.mark.parametrize(
    "name",
    ["control_iterations", "variant_iterations",
     "control_boundary_samples", "variant_boundary_samples"],
)
def test_invalid_budget_rejected(name):
    with pytest.raises(ValueError, match="positive integer"):
        evaluation.EvaluationConfig(**{name: 0})


@pytest.mark.parametrize("side", ["control", "variant"])
def test_resampling_requires_corresponding_boundary_model(side):
    with pytest.raises(ValueError, match="requires"):
        evaluation.EvaluationConfig(**{f"{side}_boundary_samples": 10})


@pytest.mark.parametrize("policy_only", [False, True])
def test_parallel_cli_matches_serial_games_and_restores_runtime(tmp_path, policy_only):
    """Worker count must not change the sample, outcomes, or caller's state."""
    settings = {"embedding_dimensions": 4, "global_state_embedding_dimensions": 8}
    torch.manual_seed(17)
    source = checkpoint.save_checkpoint(
        tmp_path / "model.pth",
        model=models.build(settings, players=2, device="cpu"),
        optimizer=None,
        configuration={"model": settings, "players": 2},
    )
    boundary = tmp_path / "boundary.pth"
    torch.save(
        {
            "format": "skyjo.boundary-value", "version": 1,
            "kind": "logistic", "players": 2, "hidden_width": 32,
            "score_scale": 100.0,
            "input_order": "next starter first, then cyclic seat order",
            "model_state_dict": BoundaryValueModel("logistic", players=2).state_dict(),
        }, boundary,
    )
    evaluation_settings = evaluation.EvaluationConfig(
        seed_count=3, seed=97, control_iterations=1, variant_iterations=2,
        variant_boundary_samples=2, variant_boundary_value_checkpoint=str(boundary),
        control_policy_only=policy_only,
    )
    serial = evaluation.evaluate_checkpoints(source, source, evaluation_settings)
    previous_threads = torch.get_num_threads()
    before = checkpoint.capture_rng_state()
    app = typer.Typer()
    app.command()(cli.compare)
    result = CliRunner().invoke(
        app,
        [
            "--control", str(source), "--control-iterations", "1",
            "--variant-iterations", "2", "--variant-boundary-samples", "2",
            "--variant-boundary-value-checkpoint", str(boundary),
            "--seed-count", "3", "--seed", "97", "--workers", "2",
            "--threads", "1", "--runs-dir", str(tmp_path / "runs"), "--allow-dirty",
        ] + (["--control-policy-only"] if policy_only else []),
    )
    assert result.exit_code == 0, result.output
    assert "Game 6/6" in result.output
    (run_path,) = (tmp_path / "runs").iterdir()
    report = json.loads((run_path / "comparison.json").read_text())
    assert report["games"] == serial["games"]
    assert report["variant_win_fraction"] == serial["variant_win_fraction"]
    assert report["control_minus_variant_margin"] == serial["control_minus_variant_margin"]
    assert report["play_mode_by_player"] == {
        "control": "policy" if policy_only else "mcts", "variant": "mcts",
    }
    if policy_only:
        assert report["search_by_player"]["control"] is None
        for game in report["games"]:
            assert game["play_mode_by_player"] == report["play_mode_by_player"]
            assert game["search_by_player"]["control"] is None
    recorded = json.loads((run_path / "resolved-config.json").read_text())
    assert recorded["execution"]["workers"] == 2
    assert torch.get_num_threads() == previous_threads
    assert random.getstate() == before["python"]
    np.testing.assert_equal(np.random.get_state(), before["numpy"])
    assert torch.equal(torch.get_rng_state(), before["torch_cpu"])
    def stop_after_game(record):
        raise RuntimeError("stop after a completed game")

    with pytest.raises(RuntimeError, match="stop after a completed game"):
        evaluation.evaluate_checkpoints(
            source, source, evaluation_settings, workers=2, on_game=stop_after_game,
        )
    assert torch.get_num_threads() == previous_threads
    assert random.getstate() == before["python"]
    np.testing.assert_equal(np.random.get_state(), before["numpy"])
    assert torch.equal(torch.get_rng_state(), before["torch_cpu"])


@pytest.mark.parametrize("side", ["control", "variant"])
def test_policy_only_rejects_unused_boundary_settings(tmp_path, side):
    boundary = tmp_path / "unused.pth"
    boundary.touch()
    with pytest.raises(ValueError, match="policy"):
        evaluation.EvaluationConfig(
            **{f"{side}_policy_only": True, f"{side}_boundary_value_checkpoint": boundary}
        )


def test_abrupt_parallel_worker_exit_fails_recorded_run(tmp_path):
    settings = {"embedding_dimensions": 4, "global_state_embedding_dimensions": 8}
    source = checkpoint.save_checkpoint(
        tmp_path / "model.pth",
        model=models.build(settings, players=2, device="cpu"),
        optimizer=None,
        configuration={"model": settings, "players": 2},
    )
    # Run the crash in an isolated driver: a lost worker must fail, not hang.
    driver = tmp_path / "crash_worker.py"
    driver.write_text(
        "import os\n"
        "from pathlib import Path\n"
        "import sys\n"
        "from skyjo import evaluation\n"
        f"sys.path.insert(0, {str(Path(cli.__file__).resolve().parent)!r})\n"
        "import run_checkpoint_comparison as cli\n"
        "def abort_game(job):\n"
        "    os._exit(17)\n"
        "if __name__ == '__main__':\n"
        "    evaluation._play_worker_game = abort_game\n"
        "    cli.compare(Path(sys.argv[1]), seed_count=1, iterations=1,\n"
        "                workers=2, runs_dir=Path(sys.argv[2]), allow_dirty=True)\n"
    )
    completed = subprocess.run(
        [sys.executable, str(driver), str(source), str(tmp_path / "runs")],
        capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode != 0
    assert "BrokenProcessPool" in completed.stderr
    (run_path,) = (tmp_path / "runs").iterdir()
    assert json.loads((run_path / "run.json").read_text())["status"] == "failed"
    assert not (run_path / "comparison.json").exists()
