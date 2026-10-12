"""Balanced-seat evaluation, recorded failures, and serial/spawn reproducibility."""

import dataclasses
import json
import os
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
from skyjo.experiments import contestants, evaluation
from skyjo.experiments.contestants import ContestantConfig
from skyjo.learning import checkpoint, models
from skyjo.learning.boundary_value import BoundaryValueModel
from skyjo.search.mcts import SearchConfig

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def model_checkpoint(tmp_path):
    settings = {"embedding_dimensions": 4, "global_state_embedding_dimensions": 8}
    with torch.random.fork_rng():
        torch.manual_seed(17)
        model = models.build(settings, players=2, device="cpu")
    return checkpoint.save_checkpoint(
        tmp_path / "model.pth",
        model=model,
        optimizer=None,
        configuration={"model": settings, "players": 2},
    )


def assert_runtime_unchanged(saved, threads):
    assert torch.get_num_threads() == threads
    assert random.getstate() == saved["python"]
    np.testing.assert_equal(np.random.get_state(), saved["numpy"])
    assert torch.equal(torch.get_rng_state(), saved["torch_cpu"])


def test_balanced_seats_share_ties_and_restore_runtime_on_callback_failure(
    tmp_path, monkeypatch
):
    paths = [tmp_path / "control.pth", tmp_path / "variant.pth"]
    for path in paths:
        path.write_bytes(path.stem.encode())
    settings = evaluation.EvaluationConfig(
        ContestantConfig(checkpoint=str(paths[0]), iterations=1),
        ContestantConfig(checkpoint=str(paths[1]), iterations=2),
        seed_count=2,
    )
    monkeypatch.setattr(
        contestants,
        "prepare",
        lambda config: SimpleNamespace(
            evaluator=SimpleNamespace(model=SimpleNamespace(players=2)),
            player=lambda rng: Path(config.checkpoint).stem,
        ),
    )
    played = []

    def play_game(players, *, environment_rng, action_rng):
        played.append((players, environment_rng.random(), action_rng.random()))
        # Exercise restoration even when a supplied player consumes global RNGs.
        random.random()
        np.random.random()
        torch.rand(1)
        if len(played) <= 2:
            return SimpleNamespace(final_scores=(100, 100), winners=(0, 1))
        seat = players.index("variant")
        return SimpleNamespace(
            final_scores=(80, 100) if seat == 0 else (100, 80),
            winners=(seat,),
        )

    monkeypatch.setattr(evaluation.play, "play_game", play_game)
    saved, threads = checkpoint.capture_rng_state(), torch.get_num_threads()
    streamed = []
    result = evaluation.evaluate_checkpoints(settings, on_game=streamed.append)
    report = result.to_dict()
    assert streamed == list(result.games)
    assert [game["seats"] for game in result.games] == [
        ["control", "variant"],
        ["variant", "control"],
    ] * 2
    assert played[0][1:] == played[1][1:]
    assert played[2][1:] == played[3][1:]
    assert report["variant_win_fraction"] == 0.75
    assert report["control_minus_variant_margin"] == 10
    assert_runtime_unchanged(saved, threads)

    def stop(record):
        raise RuntimeError("stop after a completed game")

    with pytest.raises(RuntimeError, match="stop after a completed game"):
        evaluation.evaluate_checkpoints(settings, on_game=stop)
    assert_runtime_unchanged(saved, threads)


def test_recording_keeps_completed_game_when_interrupted(
    tmp_path, model_checkpoint, monkeypatch
):
    calls = 0

    def play_game(players, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt()
        return SimpleNamespace(final_scores=(100, 100), winners=(0, 1))

    monkeypatch.setattr(evaluation.play, "play_game", play_game)
    contestant = ContestantConfig(checkpoint=str(model_checkpoint), iterations=1)
    settings = evaluation.EvaluationConfig(contestant, contestant, seed_count=1)
    with pytest.raises(KeyboardInterrupt):
        evaluation.launch_comparison(
            settings, tmp_path / "runs", repository=ROOT, allow_dirty=True
        )
    (run_path,) = (tmp_path / "runs").iterdir()
    assert json.loads((run_path / "run.json").read_text())["status"] == "interrupted"
    events = [
        json.loads(line)
        for line in (run_path / "trajectory.jsonl").read_text().splitlines()
    ]
    assert sum(event["kind"] == "evaluation_game_completed" for event in events) == 1
    assert not (run_path / "comparison.json").exists()


def test_parallel_cli_matches_serial_contestants_and_restores_runtime(
    tmp_path, model_checkpoint
):
    """Real games protect settings wiring, seat order, and worker-independent RNGs."""
    boundary = tmp_path / "boundary.pth"
    torch.save(
        {
            "format": "skyjo.boundary-value",
            "version": 1,
            "kind": "logistic",
            "players": 2,
            "hidden_width": 32,
            "score_scale": 100.0,
            "input_order": "next starter first, then cyclic seat order",
            "model_state_dict": BoundaryValueModel("logistic", players=2).state_dict(),
        },
        boundary,
    )
    control_file = tmp_path / "policy.toml"
    control_file.write_text('mode = "policy"\n')
    variant_file = tmp_path / "search.toml"
    variant_file.write_text("""iterations = 2
boundary_checkpoint = "boundary.pth"
[search]
boundary_samples = 2
after_state_evaluate_all_children = true
merge_symmetric_actions = false
""")
    settings = evaluation.EvaluationConfig(
        ContestantConfig(checkpoint=str(model_checkpoint), mode="policy", iterations=1),
        ContestantConfig(
            checkpoint=str(model_checkpoint),
            iterations=2,
            boundary_checkpoint=str(boundary),
            search=SearchConfig(
                boundary_samples=2,
                after_state_evaluate_all_children=True,
                merge_symmetric_actions=False,
            ),
        ),
        seed_count=2,
        seed=97,
    )
    saved, threads = checkpoint.capture_rng_state(), torch.get_num_threads()
    serial = evaluation.evaluate_checkpoints(settings).to_dict()
    assert_runtime_unchanged(saved, threads)
    app = typer.Typer()
    app.command()(cli.compare)
    result = CliRunner().invoke(
        app,
        [
            "--control",
            str(model_checkpoint),
            "--control-settings",
            str(control_file),
            "--variant-settings",
            str(variant_file),
            "--iterations",
            "1",
            "--seed-count",
            "2",
            "--seed",
            "97",
            "--workers",
            "2",
            "--threads",
            "1",
            "--runs-dir",
            str(tmp_path / "runs"),
            "--allow-dirty",
        ],
    )
    assert result.exit_code == 0, result.output
    (run_path,) = (tmp_path / "runs").iterdir()
    report = json.loads((run_path / "comparison.json").read_text())
    assert report["games"] == serial["games"]
    assert report["variant_win_fraction"] == serial["variant_win_fraction"]
    assert (
        report["control_minus_variant_margin"] == serial["control_minus_variant_margin"]
    )
    assert report["settings"] == dataclasses.asdict(
        dataclasses.replace(settings, workers=2)
    )
    assert report["boundary_value_checkpoints"]["variant"]["path"] == str(boundary)
    assert report["execution"]["effective_workers"] == 2
    assert json.loads((run_path / "run.json").read_text())["status"] == "completed"
    assert_runtime_unchanged(saved, threads)


def test_abrupt_parallel_worker_exit_fails_recorded_run(tmp_path, model_checkpoint):
    # The worker entrypoint is the deliberate fault-injection seam; isolate the
    # real process death so a lost worker must fail the recording, not hang pytest.
    driver = tmp_path / "crash_worker.py"
    driver.write_text(
        "import os\nfrom pathlib import Path\nimport sys\n"
        "from skyjo.experiments import evaluation\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "import run_checkpoint_comparison as cli\n"
        "def abort_game(job):\n    os._exit(17)\n"
        "if __name__ == '__main__':\n"
        "    evaluation._play_worker_game = abort_game\n"
        "    cli.compare(Path(sys.argv[1]), seed_count=1, iterations=1,\n"
        "                workers=2, runs_dir=Path(sys.argv[2]), allow_dirty=True)\n"
    )
    completed = subprocess.run(
        [sys.executable, str(driver), str(model_checkpoint), str(tmp_path / "runs")],
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        check=False,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert completed.returncode != 0
    assert "BrokenProcessPool" in completed.stderr
    (run_path,) = (tmp_path / "runs").iterdir()
    assert json.loads((run_path / "run.json").read_text())["status"] == "failed"
    assert not (run_path / "comparison.json").exists()
