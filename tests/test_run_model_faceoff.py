from __future__ import annotations

import pathlib

import pytest
import torch
import typer
from typer.testing import CliRunner

from skyjo import faceoff_cli


def test_run_faceoff_requires_paired_games() -> None:
    with pytest.raises(ValueError, match="positive even number"):
        faceoff_cli.run_faceoff(
            candidate_checkpoint=pathlib.Path("candidate.pth"),
            champion_checkpoint=pathlib.Path("champion.pth"),
            games=3,
            seed=0,
            device_name="cpu",
            mcts_iterations=1,
            terminal_state_rollouts=1,
            embedding_dimensions=None,
            global_state_embedding_dimensions=None,
            num_heads=None,
            game_completed_callback=None,
            workers=1,
        )


def test_faceoff_cli_reports_results(tmp_path, monkeypatch) -> None:
    candidate = tmp_path / "candidate.pth"
    champion = tmp_path / "champion.pth"
    candidate.touch()
    champion.touch()
    def fake_run_faceoff(**kwargs):
        for _ in range(kwargs["games"]):
            kwargs["game_completed_callback"]()
        return 3, 1

    monkeypatch.setattr(faceoff_cli, "run_faceoff", fake_run_faceoff)

    app = typer.Typer()
    app.command()(faceoff_cli.faceoff_checkpoints)
    result = CliRunner().invoke(
        app,
        [str(candidate), str(champion), "--games", "4"],
    )

    assert result.exit_code == 0
    assert "candidate_wins: 3" in result.stdout
    assert "champion_wins: 1" in result.stdout
    assert "candidate_win_rate: 75.0%" in result.stdout
    assert "winner: candidate" in result.stdout
    assert "4/4" in result.output
    assert "workers: 1" in result.stdout


def test_resolve_model_parameter_rejects_mismatch() -> None:
    with pytest.raises(ValueError, match="settings disagree"):
        faceoff_cli._resolve_model_parameter(
            name="num_heads",
            override=None,
            candidate_configuration={"num_heads": 1},
            champion_configuration={"num_heads": 2},
            default=2,
        )


def test_run_faceoff_rejects_legacy_model_architecture(tmp_path) -> None:
    candidate = tmp_path / "candidate.pth"
    champion = tmp_path / "champion.pth"
    legacy_payload = {"configuration": {"model": {"name": "equivariant"}}}
    torch.save(legacy_payload, candidate)
    torch.save(legacy_payload, champion)

    with pytest.raises(ValueError, match="Legacy EquivariantSkyNet checkpoints"):
        faceoff_cli.run_faceoff(
            candidate_checkpoint=candidate,
            champion_checkpoint=champion,
            games=2,
            seed=0,
            device_name="cpu",
            mcts_iterations=1,
            terminal_state_rollouts=1,
            embedding_dimensions=None,
            global_state_embedding_dimensions=None,
            num_heads=None,
            game_completed_callback=None,
            workers=1,
        )


def test_parallel_faceoff_combines_completed_pairs(monkeypatch) -> None:
    submitted_seeds = []

    class FakeExecutor:
        def __init__(self, **kwargs):
            assert kwargs["max_workers"] == 2

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def submit(self, function, pair_seed):
            del function
            submitted_seeds.append(pair_seed)
            future = concurrent.futures.Future()
            future.set_result((2, 0) if pair_seed % 2 == 0 else (0, 2))
            return future

    import concurrent.futures

    monkeypatch.setattr(
        faceoff_cli.concurrent.futures,
        "ProcessPoolExecutor",
        FakeExecutor,
    )
    completed_games = []
    wins = faceoff_cli._run_parallel_faceoff(
        candidate_checkpoint=pathlib.Path("candidate.pth"),
        champion_checkpoint=pathlib.Path("champion.pth"),
        paired_rounds=3,
        seed=10,
        workers=2,
        device_name="cpu",
        model_parameters={
            "embedding_dimensions": 32,
            "global_state_embedding_dimensions": 64,
            "num_heads": 2,
        },
        model_player_config=faceoff_cli._model_player_config(
            mcts_iterations=1,
            terminal_state_rollouts=1,
        ),
        game_completed_callback=lambda: completed_games.append(None),
    )

    assert submitted_seeds == [10, 11, 12]
    assert wins == (4, 2)
    assert len(completed_games) == 6


def test_missing_architecture_metadata_defers_to_strict_model_loading() -> None:
    faceoff_cli._validate_checkpoint_architecture(
        pathlib.Path("checkpoint-without-configuration.pth"),
        {},
    )
