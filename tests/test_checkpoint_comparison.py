import json
from types import SimpleNamespace

import pytest
import torch
import typer
from typer.testing import CliRunner

import run_checkpoint_comparison as cli
from skyjo import checkpoint, evaluation, models


@pytest.mark.parametrize("mode", ["direct", "cli", "interrupted"])
def test_recorded_comparison(tmp_path, monkeypatch, mode):
    settings = {"embedding_dimensions": 4, "global_state_embedding_dimensions": 8}
    model = models.build(settings, players=2, device="cpu")
    source = checkpoint.save_checkpoint(
        tmp_path / "model.pth",
        model=model,
        optimizer=None,
        configuration={"model": settings, "players": 2},
    )
    calls = 0

    def play_game(agents):
        nonlocal calls
        calls += 1
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
    assert len(played) == 2
    artifacts = [
        json.loads(line)
        for line in (run_path / "artifacts.jsonl").read_text().splitlines()
    ]
    assert any(a["path"] == "comparison.json" for a in artifacts)


@pytest.mark.parametrize("name", ["control_iterations", "variant_iterations"])
def test_invalid_budget_rejected(name):
    with pytest.raises(ValueError, match="positive integer"):
        evaluation.EvaluationConfig(**{name: 0})
