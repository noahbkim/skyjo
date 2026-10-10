"""Suite isolation and consumption of typed training results."""

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from skyjo.experiments import (
    evaluation,
    experiment_config,
    runs,
    selfplay_training,
    suites,
    training_setup,
)
from skyjo.experiments.settings import SelfPlayRunConfig
from skyjo.experiments.state import Snapshot, TrainingRunResult
from skyjo.learning import checkpoint, continuation

ROOT = Path(__file__).resolve().parents[1]


def recorded_child(config, directory, *, allow_dirty, repository):
    """A completed training boundary with artifacts but no iteration log events."""
    _, resolved = experiment_config.load_configuration(config)
    recorder = runs.RunRecorder.create(
        root=directory,
        repository=repository,
        input_path=config,
        input_bytes=config.read_bytes(),
        configuration=resolved,
        entrypoint="test",
        invocation=[],
        allow_dirty=allow_dirty,
    )
    with recorder:
        final = recorder.path / "checkpoints/final.pth"
        final.write_bytes(b"completed checkpoint")
        artifact_id = recorder.register_artifact(final, kind="checkpoint", progress={})
    return TrainingRunResult(
        recorder.manifest["run_id"],
        recorder.path,
        Snapshot(final, artifact_id),
        progress={"generated_games": 2},
        timings={"training_seconds": 0.1},
    )


def test_suite_records_isolated_paired_children_and_consumes_returned_results(
    tmp_path, monkeypatch
):
    source = tmp_path / "suite.toml"
    source.write_text(f'''name = "test-suite"
baseline = "{ROOT / "configs/smoke.toml"}"
seeds = [7, 8]
[variants.control]
[variants.aux.training]
replay_ratio = 0.25
[variants.aux.auxiliary_objectives]
round_raw_score = 0.1
[evaluation]
seed_count = 1
iterations = 1
''')
    children = []

    def train(config, directory, **kwargs):
        result = recorded_child(config, directory, **kwargs)
        children.append((json.loads(config.read_text()), result))
        return result

    monkeypatch.setattr(selfplay_training, "launch", train)
    matches = []

    def compare(settings):
        matches.append(settings)
        return evaluation.MatchResult(
            settings,
            checkpoints={},
            boundary_checkpoints={},
            games=({"variant_win_credit": 0.5, "control_minus_variant": 1.0},),
            seconds=0.01,
            effective_workers=1,
        )

    monkeypatch.setattr(evaluation, "evaluate_checkpoints", compare)
    path = suites.launch_suite(
        source, tmp_path / "runs", allow_dirty=True, repository=ROOT
    )
    report = json.loads((path / "comparison.json").read_text())
    assert [(child["variant"], child["seed"]) for child in report["runs"]] == [
        ("control", 7),
        ("aux", 7),
        ("control", 8),
        ("aux", 8),
    ]
    assert len({result.path for _, result in children}) == 4
    for (config, result), recorded in zip(children, report["runs"], strict=True):
        assert result.path.parent == path / "runs"
        assert config["experiment"]["suite_run_id"] == path.name
        assert recorded["progress"] == result.progress
        assert recorded["timings"] == result.timings
        assert recorded["checkpoint"] == result.to_dict()["checkpoint"]
        if config["experiment"]["variant"] == "aux":
            assert config["training"]["replay_ratio"] == 0.25
            assert config["auxiliary_objectives"] == {"round_raw_score": 0.1}
    assert report["averages"]["aux"]["variant_win_fraction"] == 0.5
    assert [row["training_seed"] for row in report["per_seed"]] == [7, 8]
    for comparison, settings in zip(report["per_seed"], matches, strict=True):
        assert (
            comparison["control_run_id"]
            == Path(settings.control.checkpoint).parents[1].name
        )
        assert (
            comparison["variant_run_id"]
            == Path(settings.variant.checkpoint).parents[1].name
        )
        assert settings.seed_count == 1

    # Validate every child before starting any work, even a later invalid arm.
    source.write_text(
        source.read_text() + "\n[variants.bad.training]\nunsupported = true\n"
    )
    with pytest.raises(ValueError, match="Unknown settings"):
        suites.launch_suite(
            source, tmp_path / "invalid", allow_dirty=True, repository=ROOT
        )
    assert not (tmp_path / "invalid").exists()
    assert len(children) == 4


def test_suite_failure_keeps_completed_child_without_retry(tmp_path, monkeypatch):
    calls = 0

    def fail_second(config, directory, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("training failed")
        return recorded_child(config, directory, **kwargs)

    monkeypatch.setattr(selfplay_training, "launch", fail_second)
    with pytest.raises(RuntimeError, match="training failed"):
        suites.launch_suite(
            ROOT / "configs/round_objectives_smoke.toml",
            tmp_path,
            allow_dirty=True,
            repository=ROOT,
        )
    assert calls == 2
    manifest = json.loads(next(tmp_path.glob("*/run.json")).read_text())
    assert manifest["status"] == "failed"
    child = next(tmp_path.glob("*/runs/*/run.json"))
    assert json.loads(child.read_text())["status"] == "completed"
    assert (
        child.parent / "checkpoints/final.pth"
    ).read_bytes() == b"completed checkpoint"


@pytest.mark.parametrize(
    "evaluation_settings", ["iteratons = 3", "seed = -1", "seed = 0.5"]
)
def test_suite_rejects_invalid_evaluation_before_any_training(
    tmp_path, monkeypatch, evaluation_settings
):
    source = tmp_path / "suite.toml"
    source.write_text(f'''name = "invalid-evaluation"
baseline = "{ROOT / "configs/smoke.toml"}"
seeds = [0]
[variants.control]
[variants.variant]
[evaluation]
{evaluation_settings}
''')
    monkeypatch.setattr(
        selfplay_training,
        "launch",
        lambda *args, **kwargs: pytest.fail("Training started"),
    )
    with pytest.raises(ValueError, match="[Ee]valuation"):
        suites.launch_suite(
            source, tmp_path / "runs", allow_dirty=True, repository=ROOT
        )
    assert not (tmp_path / "runs").exists()


def test_continuation_reports_only_requested_setting_changes(tmp_path):
    resolved = experiment_config.load_configuration(ROOT / "configs/smoke.toml")[1]
    settings = SelfPlayRunConfig.from_resolved(resolved)
    requested = checkpoint.normalize_configuration(settings)
    # Runtime artifact paths belong in execution records, not requested-setting deltas.
    runtime_settings = dataclasses.replace(
        settings,
        contestant=dataclasses.replace(
            settings.contestant,
            boundary_checkpoint=str(tmp_path / "frozen-boundary.pth"),
        ),
    )
    session = training_setup.prepare_training(
        runtime_settings,
        tmp_path / "checkpoints",
        parent=None,
        run_id="parent",
        requested_settings=requested,
    )
    path = checkpoint.save_checkpoint(
        tmp_path / "parent.pth",
        model=session.learner.model,
        optimizer=session.learner.optimizer,
        configuration=session.checkpoint_configuration,
        sampling_rng=session.learner.sampling_rng,
    )
    unchanged = continuation.load(path, resolved, requested_settings=requested)
    assert unchanged.provenance["configuration_changes"] == []
    assert unchanged.payload["configuration"]["requested"] == requested
    changed_settings = dataclasses.replace(
        settings,
        contestant=dataclasses.replace(settings.contestant, iterations=2),
        training=dataclasses.replace(settings.training, replay_ratio=0.5),
    )
    changed = continuation.load(
        path,
        resolved,
        requested_settings=checkpoint.normalize_configuration(changed_settings),
    )
    assert changed.provenance["configuration_changes"] == [
        {"setting": "contestant.iterations", "parent": 1, "child": 2},
        {"setting": "training.replay_ratio", "parent": 0.1, "child": 0.5},
    ]


def test_suite_resolves_each_path_at_its_declaring_config(tmp_path, monkeypatch):
    # Test path ownership without creating a synthetic replay dataset.
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    monkeypatch.setattr(
        experiment_config,
        "load_configuration",
        lambda path: (
            b"",
            {"players": 2, "replay": {"initial_dataset": "initial"}, "derived": {}},
        ),
    )
    monkeypatch.setattr(
        experiment_config, "resolve_configuration", lambda config, **kwargs: config
    )
    source = tmp_path / "suite.toml"
    source.write_text("""name="paths"
baseline="base/train.toml"
seeds=[0]
[variants.control]
[variants.aux.replay]
initial_dataset="variant-data"
""")
    children = suites.load_suite(source)["children"]
    assert children[0]["configuration"]["replay"]["initial_dataset"] == str(
        base_dir / "initial"
    )
    assert children[1]["configuration"]["replay"]["initial_dataset"] == str(
        tmp_path / "variant-data"
    )
    source.write_text(source.read_text() + "\n[variants.aux]\nplayers=3\n")
    with pytest.raises(ValueError, match="two players"):
        suites.load_suite(source)


def test_package_suite_validates_before_recording_outside_checkout(tmp_path):
    config = tmp_path / "suite.toml"
    config.write_text(
        'name="invalid"\nbaseline="missing.toml"\nseeds=[]\n[variants.control]\n'
    )
    script = """
from pathlib import Path
from skyjo.experiments.suites import launch_suite
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
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "runs").exists()
