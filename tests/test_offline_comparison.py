"""Fixed-data comparison invariants, with one tiny recorded integration run."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from skyjo import (
    buffer,
    checkpoint,
    evaluation,
    experiment_config,
    game,
    models,
    observations,
    offline,
    offline_comparison,
    play,
    skynet,
)


def dataset_at(path):
    replay = buffer.ReplayBuffer(
        max_size=24,
        spatial_input_shape=(2, 3, 4, 17),
        non_spatial_input_shape=observations.get_non_spatial_input_shape(2),
        action_mask_shape=(28,),
        target_specs=experiment_config.target_specs(
            2, {"round_raw_score": 0.1, "round_doubled": 0.1}
        ),
    )
    for index in range(6):
        state = game.new(players=2, top=index)
        # Exact opponent score, with no hidden cards, for the masked diagnostic.
        state.table[1] = 0
        state.table[1, :, :, index + 2] = 1
        mask = game.actions(state).astype(np.float32)
        labels = {
            "value": np.array([index % 2, 1 - index % 2], dtype=np.float32),
            "policy": mask / mask.sum(),
            "round_raw_score": np.array([0.3, index / 12], dtype=np.float32),
            "round_doubled": np.array([0, 1], dtype=np.float32),
        }
        replay.add_game_data(
            [play.GameDataPoint(state, None, labels)] * 2,
            game_index=100 + index,
            play_seed=index,
            target_seed=index + 10,
        )
    replay.save(path)
    return replay


def config_at(path, *, width=4, batch=2, raw=0.1):
    path.write_text(f"""name = "test-offline"
[model]
embedding_dimensions = {width}
global_state_embedding_dimensions = {width * 2}
num_heads = 1
[training]
batch_size = {batch}
gradient_diagnostic = true
[execution]
threads_per_worker = 1
[auxiliary_objectives]
round_raw_score = {raw}
round_doubled = 0.1
""")
    return path


def test_inheritance_uses_declaring_paths_and_validates(tmp_path):
    parent_dir = tmp_path / "parent"
    parent_dir.mkdir()
    replay = dataset_at(parent_dir / "replay")
    base = config_at(parent_dir / "base.toml")
    # Dataset contains a superset of targets; both enabled here match exactly.
    with base.open("a") as stream:
        stream.write('\n[replay]\ninitial_dataset = "replay"\n')
    child = tmp_path / "child.toml"
    child.write_text(
        'extends = "parent/base.toml"\n[model]\nembedding_dimensions = 8\n'
    )
    _, resolved = experiment_config.load_configuration(child)
    assert resolved["model"]["embedding_dimensions"] == 8
    assert resolved["training"]["batch_size"] == 2
    assert resolved["replay"]["initial_dataset"] == str(replay.path)
    assert resolved["auxiliary_objectives"] == {
        "round_raw_score": 0.1,
        "round_doubled": 0.1,
    }
    _, sources = experiment_config.configuration_sources(child)
    assert [Path(s["path"]).name for s in sources] == ["base.toml", "child.toml"]
    child.write_text('extends = "child.toml"\n')
    with pytest.raises(ValueError, match="cycle"):
        experiment_config.load_configuration(child)
    child.write_text('extends = "parent/base.toml"\n[model]\nunsupported = 3\n')
    with pytest.raises(ValueError, match="Unknown model"):
        experiment_config.load_configuration(child)


def test_sampler_grouping_and_global_rng_are_independent(tmp_path):
    replay = dataset_at(tmp_path / "replay")
    a = np.random.default_rng(72)
    expected = replay.sample_batch(12, rng=a).spatial_inputs
    b = np.random.default_rng(72)
    actual = []
    for _ in range(4):
        np.random.random(100)
        torch.rand(30)
        actual.append(replay.sample_batch(3, rng=b).spatial_inputs)
    np.testing.assert_array_equal(expected, np.concatenate(actual))


def test_evaluation_does_not_change_updates_and_sampler_resumes(tmp_path):
    torch.set_num_threads(1)
    replay = dataset_at(tmp_path / "replay")
    config = experiment_config.load_configuration(config_at(tmp_path / "base.toml"))[1]
    first = offline.OfflineTrainer.from_configuration(config, 5)
    second = offline.OfflineTrainer.from_configuration(config, 5)
    first.fit(replay, steps=3, batch_size=2)
    second.fit(replay, steps=1, batch_size=2)
    second.evaluate(replay, batch_size=2, diagnostics=True)
    path = tmp_path / "checkpoint.pth"
    checkpoint.save_checkpoint(
        path,
        model=second.model,
        optimizer=second.optimizer,
        configuration=config,
        sampling_rng=second.sampling_rng,
        progress=checkpoint.TrainingProgress(optimizer_steps=1),
    )
    resumed = offline.OfflineTrainer.from_configuration(config, 99)
    checkpoint.load_checkpoint(
        path,
        model=resumed.model,
        optimizer=resumed.optimizer,
        sampling_rng=resumed.sampling_rng,
    )
    resumed.fit(replay, steps=2, batch_size=2)
    for key, expected in first.model.state_dict().items():
        assert torch.equal(expected, resumed.model.state_dict()[key]), key
    restored = evaluation.load_model(path)
    assert restored.auxiliary_objectives == config["auxiliary_objectives"]


def test_metrics_normalization_and_missing_head(tmp_path):
    replay = dataset_at(tmp_path / "replay")
    config = experiment_config.load_configuration(config_at(tmp_path / "base.toml"))[1]
    trainer = offline.OfflineTrainer.from_configuration(config, 0)
    with torch.no_grad():
        trainer.model.auxiliary_heads["round_raw_score"].weight.zero_()
        trainer.model.auxiliary_heads["round_raw_score"].bias.zero_()
    metrics = trainer.evaluate(replay, batch_size=5, diagnostics=True)
    assert metrics["round_raw_score_known_count"] == 12
    assert metrics["round_raw_score_known_mae_points"] == pytest.approx(30)
    assert metrics["round_raw_score_mae_points"] == pytest.approx((43.2 + 30) / 2)
    assert metrics["round_raw_score_weighted_loss"] == pytest.approx(
        0.1 * metrics["round_raw_score_loss"]
    )
    config["auxiliary_objectives"] = {}
    without = offline.OfflineTrainer.from_configuration(config, 0).evaluate(
        replay, batch_size=5, diagnostics=True
    )
    assert not any(key.startswith("round_") for key in without)


def test_recorded_comparison_inherited_configs_and_different_losses(tmp_path):
    replay = dataset_at(tmp_path / "replay")
    base = config_at(tmp_path / "base.toml")
    wider = tmp_path / "wide.toml"
    wider.write_text(
        'extends = "base.toml"\n[model]\nembedding_dimensions = 8\n[training]\nbatch_size = 3\n[auxiliary_objectives]\nround_raw_score = 0.0\n'
    )
    original_manifest = (replay.path / "manifest.json").read_bytes()
    result = offline_comparison.launch_comparison(
        [base, wider],
        replay.path,
        tmp_path / "runs",
        seeds=(2,),
        steps=2,
        validation_fraction=0.5,
        evaluation_interval=1,
        allow_dirty=True,
    )
    report = json.loads((result / "comparison.json").read_text())
    left, right = report["runs"]
    assert left["optimizer_steps"] == right["optimizer_steps"] == 2
    assert [left["sampled_positions"], right["sampled_positions"]] == [4, 6]
    assert left["parameter_count"] != right["parameter_count"]
    assert left["run_id"] != right["run_id"]
    split = json.loads((result / "split.json").read_text())
    assert not set(split["training_game_indices"]) & set(
        split["validation_game_indices"]
    )
    assert (replay.path / "manifest.json").read_bytes() == original_manifest
    assert (result / "data/replay/manifest.json").exists()
    assert (result / "curves.csv").exists()
    assert "total_loss" not in report["paired_differences"][0]["variant_minus_control"]
    assert "round_raw_score_loss" not in right["validation"]
    for child in report["runs"]:
        assert (
            json.loads((Path(child["path"]) / "run.json").read_text())["status"]
            == "completed"
        )
        assert evaluation.load_model(Path(child["checkpoint"]["path"]))
        events = [
            json.loads(line)
            for line in (Path(child["path"]) / "trajectory.jsonl")
            .read_text()
            .splitlines()
        ]
        assert sum(e["kind"] == "gradient_diagnostic" for e in events) == 1


def test_validation_and_failure_preserve_completed_work(tmp_path, monkeypatch):
    replay = dataset_at(tmp_path / "replay")
    base = config_at(tmp_path / "base.toml")
    missing = tmp_path / "missing.toml"
    missing.write_text('extends="base.toml"\n[auxiliary_objectives]\nround_score=0.1\n')
    with pytest.raises(ValueError, match="round_score"):
        offline_comparison.launch_comparison(
            [base, missing],
            replay.path,
            tmp_path / "invalid",
            steps=0,
            allow_dirty=True,
        )
    assert not (tmp_path / "invalid").exists()
    original = offline.OfflineTrainer.from_configuration.__func__
    calls = 0

    def fail_second(cls, configuration, seed):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("diagnostic failure")
        return original(cls, configuration, seed)

    monkeypatch.setattr(
        offline.OfflineTrainer, "from_configuration", classmethod(fail_second)
    )
    with pytest.raises(RuntimeError, match="diagnostic failure"):
        offline_comparison.launch_comparison(
            [base, base],
            replay.path,
            tmp_path / "runs",
            seeds=(0,),
            steps=0,
            allow_dirty=True,
        )
    suite = next((tmp_path / "runs").iterdir())
    assert json.loads((suite / "run.json").read_text())["status"] == "failed"
    assert len(json.loads((suite / "comparison.json").read_text())["runs"]) == 1
    assert sorted(
        json.loads((p / "run.json").read_text())["status"]
        for p in (suite / "runs").iterdir()
    ) == ["completed", "failed"]


def test_concurrent_snapshot_change_rejected(tmp_path, monkeypatch):
    replay = dataset_at(tmp_path / "replay")
    config = config_at(tmp_path / "base.toml")
    original = buffer.ReplayBuffer.load.__func__

    def changed(cls, path, **kwargs):
        loaded = original(cls, path, **kwargs)
        manifest = path / "manifest.json"
        contents = json.loads(manifest.read_text())
        contents["dataset_id"] = "changed"
        manifest.write_text(json.dumps(contents))
        return loaded

    monkeypatch.setattr(buffer.ReplayBuffer, "load", classmethod(changed))
    with pytest.raises(ValueError, match="changed while loading"):
        offline_comparison.launch_comparison(
            [config, config], replay.path, tmp_path / "runs", steps=0, allow_dirty=True
        )
    assert not (tmp_path / "runs").exists()


def test_registered_architecture_uses_shared_model_builder_and_checkpoint(
    tmp_path, monkeypatch
):
    def resolve_custom(settings):
        if set(settings) != {"width"}:
            raise ValueError("custom model requires width")
        return settings

    def custom_model(*, width, **kwargs):
        return skynet.EquivariantSkyNet(
            embedding_dimensions=width,
            global_state_embedding_dimensions=width * 2,
            num_heads=1,
            **kwargs,
        )

    monkeypatch.setitem(
        models.REGISTRY,
        "test-custom",
        models.ModelDefinition(custom_model, resolve_custom),
    )
    path = tmp_path / "custom.toml"
    path.write_text('[model]\nname="test-custom"\nwidth=4\n')
    configuration = experiment_config.load_configuration(path)[1]
    trainer = offline.OfflineTrainer.from_configuration(configuration, 0)
    saved = tmp_path / "custom.pth"
    checkpoint.save_checkpoint(
        saved, model=trainer.model, optimizer=None, configuration=configuration
    )
    restored = evaluation.load_model(saved)
    for key, value in trainer.model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])
