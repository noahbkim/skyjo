import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from skyjo import checkpoint, evaluation, experiments, runs, selfplay_training

ROOT = Path(__file__).resolve().parents[1]


def test_suite_expands_supported_overrides_and_records_isolated_paired_runs(
    tmp_path, monkeypatch
):
    source = tmp_path / "suite.toml"
    source.write_text(f'''name = "test-suite"
baseline = "{ROOT / "configs/smoke.toml"}"
seeds = [7, 8]
[variants.control]
[variants.aux.model]
num_heads = 2
[variants.aux.training]
replay_ratio = 0.25
[variants.aux.auxiliary_objectives]
round_raw_score = 0.1
[evaluation]
seed_count = 1
iterations = 1
''')
    children = []

    def ordinary_run(config, directory, *, allow_dirty, repository):
        # Exercise the runner's real resolver/recorder; generation is covered by
        # the CLI smoke, so this test focuses on suite isolation and provenance.
        _, resolved = selfplay_training.experiment_config.load_configuration(config)
        recorder = runs.RunRecorder.create(
            root=directory,
            repository=ROOT,
            input_path=config,
            input_bytes=config.read_bytes(),
            configuration=resolved,
            entrypoint="test",
            invocation=[],
            allow_dirty=allow_dirty,
        )
        with recorder:
            path = recorder.path / "checkpoints/final.pth"
            path.write_bytes(b"checkpoint")
            recorder.register_artifact(
                path, kind="checkpoint", progress={}, metadata={"role": "final"}
            )
            recorder.record_event(
                "iteration_completed",
                progress={"generated_games": 2},
                metrics={"time/training_seconds": 0.1},
            )
        children.append((resolved, recorder.path))
        return recorder.path

    monkeypatch.setattr(selfplay_training, "launch", ordinary_run)
    calls = []

    def compare(control, variant, settings):
        calls.append((control, variant, settings))
        return {
            "variant_win_fraction": 0.5,
            "control_minus_variant_margin": 1.0,
            "evaluation_seconds": 0.01,
        }

    monkeypatch.setattr(evaluation, "evaluate_checkpoints", compare)
    path = experiments.launch_suite(
        source, tmp_path / "runs", allow_dirty=True, repository=ROOT
    )
    report = json.loads((path / "comparison.json").read_text())
    assert [(c["variant"], c["seed"]) for c in report["runs"]] == [
        ("control", 7),
        ("aux", 7),
        ("control", 8),
        ("aux", 8),
    ]
    assert len(set(p for _, p in children)) == 4
    for configuration, child_path in children:
        assert configuration["experiment"]["suite_run_id"] == path.name
        assert child_path.parent == path / "runs"
        if configuration["experiment"]["variant"] == "aux":
            assert configuration["model"]["num_heads"] == 2
            assert configuration["training"]["replay_ratio"] == 0.25
    assert report["averages"]["aux"]["variant_win_fraction"] == 0.5
    assert [c["training_seed"] for c in report["per_seed"]] == [7, 8]
    for comparison, call in zip(report["per_seed"], calls, strict=True):
        assert comparison["control_run_id"] == call[0].parents[1].name
        assert comparison["variant_run_id"] == call[1].parents[1].name
        assert call[2].seed_count == 1
    # Invalid later variants fail before any additional runner invocation.
    source.write_text(
        source.read_text() + "\n[variants.bad.training]\nunsupported = true\n"
    )
    with pytest.raises(ValueError, match="Unknown settings"):
        experiments.launch_suite(
            source, tmp_path / "invalid", allow_dirty=True, repository=ROOT
        )
    assert not (tmp_path / "invalid").exists()
    assert len(children) == 4


def test_suite_failure_keeps_completed_children_and_does_not_retry(
    tmp_path, monkeypatch
):
    # Reuse recorded smoke configurations; a failing ordinary runner is propagated.
    source = ROOT / "configs/round_objectives_smoke.toml"
    invoked = []

    def fail(config, directory, *, allow_dirty, repository):
        configuration = json.loads(config.read_text())
        invoked.append(configuration)
        if len(invoked) == 2:
            raise RuntimeError("training failed")
        child = runs.RunRecorder.create(
            root=directory,
            repository=ROOT,
            input_path=config,
            input_bytes=config.read_bytes(),
            configuration=configuration,
            entrypoint="test",
            invocation=[],
            allow_dirty=allow_dirty,
        )
        with child:
            final = child.path / "checkpoints/final.pth"
            final.write_bytes(b"completed checkpoint")
            child.register_artifact(
                final, kind="checkpoint", progress={}, metadata={"role": "final"}
            )
            child.record_event("iteration_completed", metrics={})
        return child.path

    monkeypatch.setattr(selfplay_training, "launch", fail)
    with pytest.raises(RuntimeError, match="training failed"):
        experiments.launch_suite(source, tmp_path, allow_dirty=True, repository=ROOT)
    assert len(invoked) == 2
    manifest = json.loads(next(tmp_path.glob("*/run.json")).read_text())
    assert manifest["status"] == "failed"
    assert next(tmp_path.glob("*/configs/000.json")).exists()
    child = next(tmp_path.glob("*/runs/*/run.json"))
    assert json.loads(child.read_text())["status"] == "completed"
    assert (
        child.parent / "checkpoints/final.pth"
    ).read_bytes() == b"completed checkpoint"


def test_suite_paths_resolve_at_declaring_config_and_two_player_validation(
    tmp_path, monkeypatch
):
    # Path resolution is tested without inventing a replay dataset manifest.
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    base = {"players": 2, "replay": {"initial_dataset": "initial"}}
    monkeypatch.setattr(
        experiments.experiment_config,
        "load_configuration",
        lambda path: (b"", {**base, "derived": {}}),
    )
    configs = []

    def resolve(config, **kwargs):
        configs.append(config)
        return config

    monkeypatch.setattr(experiments.experiment_config, "resolve_configuration", resolve)
    source = tmp_path / "suite.toml"
    source.write_text("""name="paths"
baseline="base/train.toml"
seeds=[0]
[variants.control]
[variants.aux.replay]
initial_dataset="variant-data"
""")
    experiments.load_suite(source)
    assert configs[0]["replay"]["initial_dataset"] == str(base_dir / "initial")
    assert configs[1]["replay"]["initial_dataset"] == str(tmp_path / "variant-data")
    source.write_text(source.read_text() + "\n[variants.aux]\nplayers=3\n")
    with pytest.raises(ValueError, match="two players"):
        experiments.load_suite(source)


def test_evaluation_balances_seats_shares_ties_and_restores_rng(tmp_path, monkeypatch):
    control, variant = tmp_path / "control.pth", tmp_path / "variant.pth"
    control.write_bytes(b"control")
    variant.write_bytes(b"variant")
    monkeypatch.setattr(evaluation, "load_model", lambda path: path.stem)
    monkeypatch.setattr(
        evaluation.predictor, "LocalPredictor", lambda model, **kwargs: model
    )
    settings_seen = []

    def agent(client, **kwargs):
        settings_seen.append(kwargs)
        return client

    monkeypatch.setattr(evaluation.player, "ModelPlayer", agent)
    games = []

    def play_game(seats):
        games.append((seats, random.random(), np.random.random(), torch.rand(1).item()))
        # First seed ties; second seed has variant win by 20 in each seat.
        if len(games) <= 2:
            return SimpleNamespace(final_scores=(100, 100), winners=(0, 1))
        seat = seats.index("variant")
        return SimpleNamespace(
            final_scores=(80, 100) if seat == 0 else (100, 80), winners=(seat,)
        )

    monkeypatch.setattr(evaluation.play, "play_game", play_game)
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    saved = checkpoint.capture_rng_state()
    report = evaluation.evaluate_checkpoints(
        control, variant, evaluation.EvaluationConfig(seed_count=2, iterations=1)
    )
    assert random.getstate() == saved["python"]
    np.testing.assert_equal(np.random.get_state(), saved["numpy"])
    assert torch.equal(torch.get_rng_state(), saved["torch_cpu"])
    assert [g["seats"] for g in report["games"]] == [
        ["control", "variant"],
        ["variant", "control"],
    ] * 2
    assert games[0][1:] == games[1][1:] and games[2][1:] == games[3][1:]
    assert report["variant_win_fraction"] == 0.75
    assert report["control_minus_variant_margin"] == 10
    assert report["checkpoints"]["control"]["sha256"] == runs.file_digest(control)
    assert all(
        s["action_softmax_temperature"] == s["mcts_dirichlet_epsilon"] == 0
        for s in settings_seen
    )


def test_package_suite_validates_config_outside_checkout(tmp_path):
    import subprocess
    import sys

    config = tmp_path / "suite.toml"
    config.write_text(
        'name = "invalid"\nbaseline = "missing.toml"\nseeds = []\n[variants.control]\n'
    )
    script = """
from pathlib import Path
from skyjo.experiments import launch_suite
import sys
try:
    launch_suite(Path(sys.argv[1]), Path("runs"), repository=Path(sys.argv[2]))
except ValueError as error:
    assert "seeds" in str(error)
else:
    raise AssertionError("invalid suite was accepted")
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(config), str(ROOT)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "runs").exists()
