"""Continuation state and completed-iteration budget boundaries."""

import copy
import json
import os
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from helpers import NaiveQuickFinishPlayer

from skyjo import (
    buffer,
    continuation,
    experiment_config,
    experiments,
    models,
    play,
    runs,
    training_budget,
)
from skyjo import selfplay_training as runner

ROOT = Path(__file__).resolve().parents[1]


def records(path, name):
    return [json.loads(line) for line in (path / name).read_text().splitlines()]


def final_checkpoint(path):
    artifact = next(
        a
        for a in reversed(records(path, "artifacts.jsonl"))
        if a.get("artifact_kind") == "checkpoint" and a["metadata"]["role"] == "final"
    )
    file = path / artifact["path"]
    return file, torch.load(file, weights_only=False)


def smoke_config():
    config = experiment_config.load_configuration(ROOT / "configs/smoke.toml")[1]
    config["selfplay"]["games_per_iteration"] = 1
    return config


def launch(tmp_path, config):
    file = tmp_path / "input.json"
    file.write_text(json.dumps(config))
    return runner.launch(file, tmp_path / "runs", repository=ROOT, allow_dirty=True)


@pytest.fixture(scope="module")
def parent_run(tmp_path_factory):
    return launch(tmp_path_factory.mktemp("parent"), smoke_config())


def child_config(parent):
    config = smoke_config()
    config["initial_checkpoint"] = str(final_checkpoint(parent)[0])
    config["replay"]["initial_dataset"] = str(parent / "data/replay")
    return config


@pytest.mark.parametrize(
    "limits", [(0, 0), (-1, 1), (1.5, 0), (1, -1), (1, float("inf")), (1, float("nan"))]
)
def test_invalid_budgets(limits, tmp_path):
    with pytest.raises(ValueError):
        experiment_config.resolve_configuration(
            {"budget": dict(zip(("iterations", "max_seconds"), limits))},
            base_directory=tmp_path,
        )


def test_budget_counts_invocation_time_and_either_cap(monkeypatch):
    clock = SimpleNamespace(perf_counter=lambda: 109.0)
    monkeypatch.setattr(training_budget, "time", clock)
    budget = training_budget.TrainingBudget(2, 10, started=100)
    assert budget.stop_reason(1) is None
    assert budget.stop_reason(2) == "iteration_limit"
    clock.perf_counter = lambda: 112.0
    assert budget.stop_reason(1) == "time_limit"
    assert budget.stop_reason(2) == "both"
    assert budget.metrics()["time/budget_overshoot_seconds"] == 2


@pytest.mark.parametrize(
    "stage",
    ["setup", "worker_setup", "generation", "training", "validation", "recording"],
)
def test_deadline_finishes_iteration_and_forces_final_save(
    tmp_path, monkeypatch, stage
):
    config = smoke_config()
    config["budget"].update(iterations=0, max_seconds=10, checkpoint_interval=99)
    config["validation"]["concept_interval"] = 1
    elapsed = [0.0]
    monkeypatch.setattr(
        training_budget.TrainingBudget, "elapsed", lambda self: elapsed[0]
    )
    sample = play.play_game([NaiveQuickFinishPlayer(), NaiveQuickFinishPlayer()])
    calls = []

    def generate(pool, **kwargs):
        calls.append(kwargs["first_game_index"])
        if stage == "generation":
            elapsed[0] = 12
        return [runner.GeneratedGame(kwargs["first_game_index"], 3, sample)]

    def pool(**kwargs):
        if stage == "worker_setup":
            elapsed[0] = 12
        return nullcontext(None)

    monkeypatch.setattr(runner, "generate_iteration", generate)
    monkeypatch.setattr(runner.mp, "Pool", pool)
    target, name = {
        "setup": (runner.models, "build"),
        "training": (runner.train, "train_iteration"),
        "validation": (runner.explain, "evaluate_concepts"),
        "recording": (runner.experiment_training.RecipeRecording, "save_rounds"),
    }.get(stage, (None, None))
    if target is not None:
        original = getattr(target, name)
        count = 0

        def expire(*args, **kwargs):
            nonlocal count
            result = original(*args, **kwargs)
            count += 1
            if stage != "validation" or count == 2:
                elapsed[0] = 12
            return result

        monkeypatch.setattr(target, name, expire)
    path = launch(tmp_path, config)
    expected = 0 if stage in ("setup", "worker_setup") else 1
    assert len(calls) == expected
    _, payload = final_checkpoint(path)
    assert payload["progress"]["iteration"] == expected
    assert (payload["progress"]["optimizer_steps"] > 0) == bool(expected)
    trace = records(path, "trajectory.jsonl")
    finish = next(e for e in trace if e["kind"] == "budget_completed")
    assert finish["context"]["stop_reason"] == "time_limit"
    assert finish["metrics"]["time/budget_overshoot_seconds"] == 2
    assert json.loads((path / "run.json").read_text())["status"] == "completed"
    if expected:
        assert any(e["kind"] == "iteration_completed" for e in trace)
        assert (path / "data/replay/manifest.json").is_file()
    concepts = [
        e["progress"]["iteration"] for e in trace if e["kind"] == "concept_checks"
    ]
    assert concepts == ([0, 1] if expected else [0])


def test_continuations_restore_exact_state_and_reproduce_updates(
    parent_run, tmp_path, monkeypatch
):
    source, parent = final_checkpoint(parent_run)
    protected = [source, *sorted((parent_run / "data/replay").rglob("*"))]
    digests = {p: runs.file_digest(p) for p in protected if p.is_file()}
    config = child_config(parent_run)
    children = [launch(tmp_path, config), launch(tmp_path, config)]
    for child in children:
        entries = records(child, "artifacts.jsonl")
        initial = next(a for a in entries if a.get("artifact_kind") == "checkpoint")
        start = torch.load(child / initial["path"], weights_only=False)
        torch.testing.assert_close(
            start["model_state_dict"], parent["model_state_dict"], rtol=0, atol=0
        )
        torch.testing.assert_close(
            start["optimizer_state_dict"],
            parent["optimizer_state_dict"],
            rtol=0,
            atol=0,
        )
        assert start["rng_state"]["python"] == parent["rng_state"]["python"]
        np.testing.assert_equal(
            start["rng_state"]["numpy"], parent["rng_state"]["numpy"]
        )
        assert torch.equal(
            start["rng_state"]["torch_cpu"], parent["rng_state"]["torch_cpu"]
        )
        _, final = final_checkpoint(child)
        assert final["progress"]["iteration"] == 2
        assert final["progress"]["generated_games"] == 2
        assert final["continuation_state"]["next_game_index"] == 2
        trace = records(child, "trajectory.jsonl")
        lineage = next(
            e["context"] for e in trace if e["kind"] == "continuation_started"
        )
        assert lineage["checkpoint_sha256"] == digests[source]
        assert lineage["run_id"] == parent_run.name
        assert (
            lineage["source_replay_dataset_id"]
            == json.loads((parent_run / "data/replay/manifest.json").read_text())[
                "dataset_id"
            ]
        )
        assert trace[-1]["progress"]["additional_iterations"] == 1
        rounds = [a for a in entries if a.get("artifact_kind") == "round_statistics"]
        assert {r["game_index"] for a in rounds for r in records(child, a["path"])} == {
            1
        }
    first, second = [final_checkpoint(p)[1] for p in children]
    torch.testing.assert_close(
        first["model_state_dict"], second["model_state_dict"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        first["optimizer_state_dict"], second["optimizer_state_dict"], rtol=0, atol=0
    )
    assert {p: runs.file_digest(p) for p in digests} == digests

    # Continuing also matches the second update of an uninterrupted run.
    uninterrupted = smoke_config()
    uninterrupted["budget"]["iterations"] = 2
    _, result = final_checkpoint(launch(tmp_path, uninterrupted))
    torch.testing.assert_close(
        first["model_state_dict"], result["model_state_dict"], rtol=0, atol=0
    )


def test_replay_ratio_switches_after_fill_and_survives_continuation(
    tmp_path, monkeypatch
):
    sample = play.play_game([NaiveQuickFinishPlayer(), NaiveQuickFinishPlayer()])
    positions = len(play.game_result_to_game_data(sample)[0])
    monkeypatch.setattr(runner.mp, "Pool", lambda **kwargs: nullcontext(None))
    monkeypatch.setattr(
        runner,
        "generate_iteration",
        lambda pool, **kwargs: [
            runner.GeneratedGame(kwargs["first_game_index"], 3, sample)
        ],
    )
    config = smoke_config()
    config["validation"]["concept_interval"] = 0
    config["training"].update(
        batch_size=positions, replay_ratio=1.0, replay_ratio_after_fill=2.0
    )
    config["replay"]["capacity"] = 2 * positions + 1
    config["budget"]["iterations"] = 3
    parent = launch(tmp_path, config)
    trace = records(parent, "trajectory.jsonl")
    assert [
        e["metrics"]["training/optimizer_steps"]
        for e in trace if e["kind"] == "training"
    ] == [1, 1, 1]
    changes = [e for e in trace if e["kind"] == "replay_ratio_changed"]
    assert len(changes) == 1
    assert changes[0]["context"]["applies_from_iteration"] == 4

    child = copy.deepcopy(config)
    child["budget"]["iterations"] = 1
    child["initial_checkpoint"] = str(final_checkpoint(parent)[0])
    child["replay"]["initial_dataset"] = str(parent / "data/replay")
    continued = launch(tmp_path, child)
    assert next(
        e["metrics"]["training/optimizer_steps"]
        for e in records(continued, "trajectory.jsonl") if e["kind"] == "training"
    ) == 2

    # An uninterrupted run and a restored run must perform the same updates.
    config["budget"]["iterations"] = 4
    uninterrupted = launch(tmp_path, config)
    torch.testing.assert_close(
        final_checkpoint(continued)[1]["model_state_dict"],
        final_checkpoint(uninterrupted)[1]["model_state_dict"],
        rtol=0, atol=0,
    )
    torch.testing.assert_close(
        final_checkpoint(continued)[1]["optimizer_state_dict"],
        final_checkpoint(uninterrupted)[1]["optimizer_state_dict"],
        rtol=0, atol=0,
    )

    # Enlarging replay again starts a new filling phase, not inherited ratio2.
    child["replay"]["capacity"] = 4 * positions + 1
    expanded = launch(tmp_path, child)
    assert next(
        e["metrics"]["training/optimizer_steps"]
        for e in records(expanded, "trajectory.jsonl") if e["kind"] == "training"
    ) == 1


@pytest.mark.parametrize("ratio", [0, "8", True])
def test_invalid_replay_ratio_after_fill_fails_during_resolution(tmp_path, ratio):
    with pytest.raises(ValueError, match="replay_ratio_after_fill"):
        experiment_config.resolve_configuration(
            {"training": {"replay_ratio_after_fill": ratio}}, base_directory=tmp_path
        )


def test_legacy_seed_provenance_validation_and_optimizer_override(parent_run, tmp_path):
    config = experiment_config.resolve_configuration(
        child_config(parent_run), base_directory=tmp_path
    )
    source, payload = final_checkpoint(parent_run)
    parent = continuation.load(source, config)
    model = models.build(config["model"], players=2, device="cpu")
    optimizer = runner.train.make_optimizer(model, 0.0002)
    parent.restore(model, optimizer)
    assert optimizer.param_groups[0]["lr"] == 0.0002
    original = payload["optimizer_state_dict"]["state"]
    torch.testing.assert_close(
        optimizer.state_dict()["state"], original, rtol=0, atol=0
    )
    # Verify the fallback against real run-record hashes without editing the parent.
    legacy = tmp_path / "legacy"
    (legacy / "checkpoints").mkdir(parents=True)
    legacy_payload = copy.deepcopy(payload)
    legacy_payload["configuration"].pop("seed")
    legacy_payload["configuration"].pop("optimizer")
    legacy_path = legacy / "checkpoints/parent.pth"
    torch.save(legacy_payload, legacy_path)
    for name in ("run.json", "resolved-config.json"):
        (legacy / name).write_bytes((parent_run / name).read_bytes())
    artifact = {
        "artifact_kind": "checkpoint",
        "artifact_id": "parent",
        "path": "checkpoints/parent.pth",
        "sha256": runs.file_digest(legacy_path),
        "progress": {"generated_positions": 123},
    }
    (legacy / "artifacts.jsonl").write_text(json.dumps(artifact) + "\n")
    assert continuation.load(legacy_path, config).provenance["seed"] == 0
    for key, value, message in [
        ("seed", 5, "seed"),
        ("players", 3, "player count"),
        ("auxiliary_objectives", {"round_raw_score": 0.1}, "heads"),
    ]:
        changed = copy.deepcopy(config)
        changed[key] = value
        with pytest.raises(ValueError, match=message):
            continuation.load(source, changed)
    changed = copy.deepcopy(config)
    changed["model"]["embedding_dimensions"] *= 2
    with pytest.raises(ValueError, match="architecture"):
        continuation.load(source, changed)
    legacy_payload["optimizer_state_dict"] = None
    torch.save(legacy_payload, tmp_path / "missing.pth")
    with pytest.raises(ValueError, match="optimizer"):
        continuation.load(tmp_path / "missing.pth", config)
    (legacy / "resolved-config.json").write_text('{"seed": 7}')
    with pytest.raises(ValueError, match="provenance"):
        continuation.load(legacy_path, config)


def test_replay_copy_order_and_concurrent_read_rejected(
    parent_run, tmp_path, monkeypatch
):
    source = parent_run / "data/replay"
    replay = buffer.ReplayBuffer.load(source)
    config = buffer.Config(
        max_size=len(replay),
        spatial_input_shape=replay.spatial_input_buffer.shape[1:],
        non_spatial_input_shape=replay.non_spatial_input_buffer.shape[1:],
        action_mask_shape=replay.action_masks.shape[1:],
        target_specs=replay.target_specs,
        path=tmp_path / "child",
    )
    copied = runner.initialize_training_data_buffer(config, source, replay.dataset_id)
    assert copied.spatial_input_buffer.flags.writeable
    np.testing.assert_equal(
        copied.ordered_batch().spatial_inputs, replay.ordered_batch().spatial_inputs
    )
    assert copied.game_indices == replay.game_indices
    with pytest.raises(ValueError, match="identity"):
        runner.initialize_training_data_buffer(config, source, "different")
    # Mutate only a temporary fixture to simulate an atomic concurrent rewrite.
    temporary = replay.save(tmp_path / "changing")
    original = buffer.ReplayBuffer.from_config_or_load

    def changed(config):
        result = original(config)
        manifest = temporary / "manifest.json"
        manifest.write_text(manifest.read_text() + "\n")
        return result

    monkeypatch.setattr(buffer.ReplayBuffer, "from_config_or_load", changed)
    with pytest.raises(ValueError, match="changed while loading"):
        runner.initialize_training_data_buffer(config, temporary)


def test_checkpoint_path_inheritance_and_required_replay(parent_run, tmp_path):
    config = child_config(parent_run)
    expected = config["initial_checkpoint"]
    config["initial_checkpoint"] = os.path.relpath(expected, tmp_path)
    file = tmp_path / "base.json"
    file.write_text(json.dumps(config))
    child_dir = tmp_path / "nested"
    child_dir.mkdir()
    child = child_dir / "child.toml"
    child.write_text(
        'extends = "../base.json"\n[budget]\niterations = 0\nmax_seconds = 10\n'
    )
    resolved = experiment_config.load_configuration(child)[1]
    assert resolved["initial_checkpoint"] == expected
    config["replay"]["initial_dataset"] = None
    with pytest.raises(ValueError, match="requires replay"):
        experiment_config.resolve_configuration(config, base_directory=tmp_path)


def test_existing_suite_accepts_setup_timeout_final_checkpoint(tmp_path):
    suite = tmp_path / "suite.toml"
    suite.write_text(f'''name = "zero-work-budget-smoke"
baseline = "{ROOT / "configs/smoke.toml"}"
seeds = [0]
[variants.control.budget]
iterations = 0
max_seconds = 0.000000001
''')
    path = experiments.launch_suite(
        suite, tmp_path / "runs", repository=ROOT, allow_dirty=True
    )
    report = json.loads((path / "comparison.json").read_text())
    (child,) = report["runs"]
    assert child["progress"]["iteration"] == 0
    assert child["checkpoint"]["metadata"]["role"] == "final"
    assert child["timings"]["time/run_seconds"] > 0
