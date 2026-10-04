import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from skyjo import buffer, checkpoint, experiment_config, selfplay_training, skynet

REPOSITORY = Path(__file__).resolve().parents[1]


def events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_delivered_configs_resolve_and_round_trip(tmp_path):
    for name in ("smoke", "baseline"):
        _, config = experiment_config.load_configuration(
            REPOSITORY / "configs" / f"{name}.toml"
        )
        saved = tmp_path / f"{name}.json"
        saved.write_text(json.dumps(config))
        assert experiment_config.load_configuration(saved)[1] == config
    assert config["selfplay"]["games_per_iteration"] == 256
    assert config["budget"]["iterations"] == 100
    assert config["replay"]["capacity"] == 524288
    assert config["logging"]["progress_interval_seconds"] == 0
    assert config["validation"]["concept_interval"] == 5
    assert config["derived"]["optimizer"]["weight_decay"] == 1e-4


def test_invalid_config_fails_before_creating_run(tmp_path):
    config = tmp_path / "bad.toml"
    config.write_text("[training]\nbatch_size = 0\n")
    with pytest.raises(ValueError, match="training.batch_size must be positive"):
        selfplay_training.launch(
            config,
            tmp_path / "runs",
            allow_dirty=True,
            repository=REPOSITORY,
        )
    assert not (tmp_path / "runs").exists()


def test_real_smoke_cli_and_saved_config_rerun(tmp_path):
    source = REPOSITORY / "configs" / "smoke.toml"
    run_paths = []
    for _ in range(2):
        completed = subprocess.run(
            [
                sys.executable,
                str(REPOSITORY / "distributed_main.py"),
                "--config",
                str(source),
                "--runs-dir",
                str(tmp_path / "runs"),
                "--allow-dirty",
            ],
            cwd=REPOSITORY,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        run_path = Path(completed.stdout.split("Run directory: ", 1)[1].strip())
        run_paths.append(run_path)
        manifest = json.loads((run_path / "run.json").read_text())
        assert manifest["status"] == "completed"
        assert manifest["final_progress"]["optimizer_steps"] > 0
        assert manifest["final_progress"]["generated_games"] == 2
        assert (run_path / "logs/train.log").stat().st_size > 0
        assert "Not suitable" in (run_path / "notes.md").read_text()
        trace = events(run_path / "trajectory.jsonl")
        assert any(
            e["kind"] == "training" and "loss/outcome_value_loss" in e["metrics"]
            for e in trace
        )
        artifacts = events(run_path / "artifacts.jsonl")
        checkpoints = [e for e in artifacts if e.get("artifact_kind") == "checkpoint"]
        assert [e["progress"]["iteration"] for e in checkpoints] == [0, 1]
        assert len(list((run_path / "checkpoints").glob("*.pth"))) == 2
        data_record = next(
            e for e in artifacts if e.get("artifact_kind") == "replay_data"
        )
        round_artifact = next(
            e for e in artifacts if e.get("artifact_kind") == "round_statistics"
        )
        round_records = events(run_path / round_artifact["path"])
        assert {r["game_index"] for r in round_records} == {0, 1}
        assert all(
            r["iteration"] == 1 and r["checkpoint_artifact_id"] for r in round_records
        )
        assert all(r["decisions"] == 2 * r["turns"] + 2 for r in round_records)
        concepts = [e for e in trace if e["kind"] == "concept_checks"]
        assert [e["progress"]["iteration"] for e in concepts] == [0, 1]
        training_metrics = next(e["metrics"] for e in trace if e["kind"] == "training")
        assert (
            training_metrics["policy/all/positions"]
            == training_metrics["training/sampled_positions"]
        )
        iteration_metrics = next(
            e["metrics"] for e in trace if e["kind"] == "iteration_completed"
        )
        assert iteration_metrics["round/count"] == len(round_records)
        assert iteration_metrics["generation/decisions_per_second"] > 0
        log = (run_path / "logs/train.log").read_text()
        assert "games/s" in log and "replay-equivalent passes" in log
        assert "[TARGETS]" not in log and " tasks" not in log
        assert log.count("[SELF-PLAY]") == 1
        assert log.count("[TRAIN]") == 1
        assert log.count("[GAMES]") == 1
        assert "p90=" not in log and "Action rates:" not in log
        assert "Phase timings" not in log and "Mean losses" not in log
        replay = buffer.ReplayBuffer.load(run_path / data_record["path"])
        assert replay.dataset_id == data_record["metadata"]["dataset_id"]
        assert replay.game_count == 2
        assert set(replay.target_names) == {"value", "policy"}
        initial_payload = torch.load(
            run_path / checkpoints[0]["path"], weights_only=False
        )
        final_payload = torch.load(
            run_path / checkpoints[-1]["path"], weights_only=False
        )
        assert any(
            not torch.equal(initial_payload["model_state_dict"][k], v)
            for k, v in final_payload["model_state_dict"].items()
        )
        final = [e for e in artifacts if e.get("artifact_kind") == "checkpoint"][-1]
        model = skynet.EquivariantSkyNet(
            spatial_input_shape=replay.spatial_input_buffer.shape[1:],
            non_spatial_input_shape=replay.non_spatial_input_buffer.shape[1:],
            value_output_shape=(2,),
            policy_output_shape=replay.action_masks.shape[1:],
            device=torch.device("cpu"),
            embedding_dimensions=8,
            global_state_embedding_dimensions=16,
            num_heads=1,
        )
        progress = checkpoint.load_checkpoint(
            run_path / final["path"], model=model, restore_rng=False
        )
        assert progress.optimizer_steps == manifest["final_progress"]["optimizer_steps"]
        source = run_path / "resolved-config.json"
    assert run_paths[0] != run_paths[1]
    assert (run_paths[0] / "resolved-config.json").read_bytes() == source.read_bytes()

    # Initial data paths are resolved relative to the config and bound to the
    # recorded dataset identity, rather than whichever data later occupies a path.
    config = json.loads(source.read_text())
    config["replay"]["initial_dataset"] = str(
        (run_paths[0] / "data/replay").relative_to(tmp_path)
    )
    seeded_input = tmp_path / "seeded.json"
    seeded_input.write_text(json.dumps(config))
    _, seeded = experiment_config.load_configuration(seeded_input)
    assert Path(seeded["replay"]["initial_dataset"]).is_absolute()
    seeded_input.write_text(json.dumps(seeded))
    assert experiment_config.load_configuration(seeded_input)[1] == seeded
    seeded["replay"]["dataset_id"] = "different-snapshot"
    seeded_input.write_text(json.dumps(seeded))
    with pytest.raises(ValueError, match="identity"):
        experiment_config.load_configuration(seeded_input)


@pytest.mark.parametrize("iterations,interval", [(1, 1), (3, 1), (3, 2), (4, 2)])
def test_continuous_training_records_exact_snapshots(
    tmp_path, monkeypatch, iterations, interval
):
    _, config = experiment_config.load_configuration(REPOSITORY / "configs/smoke.toml")
    config["budget"].update(iterations=iterations, checkpoint_interval=interval)
    config["selfplay"]["games_per_iteration"] = 1
    source = tmp_path / "small.json"
    source.write_text(json.dumps(config))

    # Real generation and optimizer updates.
    # Capture training outputs so checkpoint assertions detect any later rollback.
    trained = []
    original_train = selfplay_training.train.train_steps

    def capture_training(model, replay, **kwargs):
        optimizer = kwargs["optimizer"]
        if trained:
            torch.testing.assert_close(
                model.state_dict(), trained[-1][0], rtol=0, atol=0
            )
            torch.testing.assert_close(
                optimizer.state_dict(), trained[-1][1], rtol=0, atol=0
            )
        losses = original_train(model, replay, **kwargs)
        trained.append(copy.deepcopy((model.state_dict(), optimizer.state_dict())))
        return losses

    monkeypatch.setattr(selfplay_training.train, "train_steps", capture_training)
    path = selfplay_training.launch(
        source,
        tmp_path / "runs",
        allow_dirty=True,
        repository=REPOSITORY,
    )
    artifacts = events(path / "artifacts.jsonl")
    boundaries = sorted({*range(interval, iterations + 1, interval), iterations})
    checkpoints = [e for e in artifacts if e.get("artifact_kind") == "checkpoint"]
    assert [e["progress"]["iteration"] for e in checkpoints] == [0, *boundaries]
    assert len(list((path / "checkpoints").glob("*.pth"))) == len(checkpoints)
    checkpoint_by_id = {e["artifact_id"]: e for e in checkpoints}
    initial = torch.load(path / checkpoints[0]["path"], weights_only=False)
    assert initial["configuration"]["model"]["name"]
    assert initial["optimizer_state_dict"]["state"] == {}
    for saved in checkpoints[1:]:
        payload = torch.load(path / saved["path"], weights_only=False)
        iteration = payload["progress"]["iteration"]
        weights, optimizer_state = trained[iteration - 1]
        torch.testing.assert_close(payload["model_state_dict"], weights, rtol=0, atol=0)
        torch.testing.assert_close(
            payload["optimizer_state_dict"], optimizer_state, rtol=0, atol=0
        )
        assert (
            payload["progress"]["optimizer_steps"]
            == saved["progress"]["optimizer_steps"]
        )
        assert all(
            s["step"].item() == payload["progress"]["optimizer_steps"]
            for s in optimizer_state["state"].values()
        )

    replays = [e for e in artifacts if e.get("artifact_kind") == "replay_data"]
    superseded = [e for e in artifacts if e["kind"] == "superseded"]
    assert len(replays) == iterations and len(superseded) == iterations - 1
    for old, replacement, event in zip(replays, replays[1:], superseded):
        assert event["artifact_id"] == old["artifact_id"]
        assert event["sequence"] < replacement["sequence"]
    run_id = json.loads((path / "run.json").read_text())["run_id"]
    for iteration, replay in enumerate(replays, start=1):
        batch = replay["metadata"]["latest_generated_batch"]
        assert batch["run_id"] == run_id
        assert batch["generation_iteration"] == iteration
        if iteration == 1 or (iteration - 1) % interval == 0:
            generation_checkpoint = checkpoint_by_id[batch["checkpoint_artifact_id"]]
            assert generation_checkpoint["progress"]["iteration"] == iteration - 1
        else:
            assert batch["checkpoint_artifact_id"] is None
            assert batch["checkpoint_path"] is None
    manifest = json.loads((path / replays[-1]["path"] / "manifest.json").read_text())
    assert manifest["dataset_id"] == replays[-1]["metadata"]["dataset_id"]
    assert manifest["source_checkpoint"] is None
    assert manifest["game_count"] == iterations


@pytest.mark.parametrize(
    "settings",
    [
        {"players": 9},
        {"players": 1},
        {"search": {"c_puct": 0}},
        {"search": {"c_puct": float("inf")}},
        {"search": {"action_softmax_temperature": -1}},
        {"search": {"action_softmax_temperature": float("nan")}},
    ],
)
def test_invalid_domain_config_fails_before_creating_artifacts(tmp_path, settings):
    config = tmp_path / "invalid.json"
    config.write_text(json.dumps(settings))
    with pytest.raises(ValueError):
        selfplay_training.launch(
            config, tmp_path / "runs", repository=REPOSITORY, allow_dirty=True
        )
    assert not (tmp_path / "runs").exists()
