import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from skyjo import buffer, checkpoint, experiment_config, skynet

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
import distributed_main  # noqa: E402


def events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_delivered_configs_resolve_and_round_trip(tmp_path):
    for name in ("smoke", "score_aux"):
        _, config = experiment_config.load_configuration(
            REPOSITORY / "configs" / f"{name}.toml"
        )
        saved = tmp_path / f"{name}.json"
        saved.write_text(json.dumps(config))
        assert experiment_config.load_configuration(saved)[1] == config
    assert config["selfplay"]["games_per_iteration"] == 1024
    assert config["derived"]["optimizer"]["weight_decay"] == 1e-4


def test_invalid_config_fails_before_creating_run(tmp_path):
    config = tmp_path / "bad.toml"
    config.write_text("[training]\nfuture_clear_scale = 1.0\n")
    with pytest.raises(ValueError, match="Future-clear"):
        distributed_main.launch(config, tmp_path / "runs", allow_dirty=True)
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
            e["kind"] == "training" and "loss/total_loss" in e["metrics"] for e in trace
        )
        artifacts = events(run_path / "artifacts.jsonl")
        data_record = next(
            e for e in artifacts if e.get("artifact_kind") == "replay_data"
        )
        replay = buffer.ReplayBuffer.load(run_path / data_record["path"])
        assert replay.dataset_id == data_record["metadata"]["dataset_id"]
        assert replay.game_count == 2
        final = [e for e in artifacts if e.get("artifact_kind") == "checkpoint"][-1]
        model = skynet.EquivariantSkyNetWithRoundScoreAux(
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


@pytest.mark.parametrize(
    "checkpoint_interval,promotion_interval", [(1, 1), (2, 1), (1, 2)]
)
def test_score_recipe_records_validation_replay_replacement_and_rollback(
    tmp_path, monkeypatch, checkpoint_interval, promotion_interval
):
    _, config = experiment_config.load_configuration(
        REPOSITORY / "configs/score_aux.toml"
    )
    config["budget"]["iterations"] = 2
    config["budget"]["checkpoint_interval"] = checkpoint_interval
    config["selfplay"].update(
        games_per_iteration=1, games_per_task=1, outcome_rollouts=1
    )
    config["search"].update(iterations=1, terminal_state_initial_rollouts=1)
    config["execution"].update(workers=1, threads_per_worker=1)
    config["training"].update(batch_size=16, replay_ratio=0.1)
    config["replay"]["capacity"] = 4096
    config["faceoff"]["paired_rounds"] = 1
    config["faceoff"]["interval"] = promotion_interval
    source = tmp_path / "score-small.json"
    source.write_text(json.dumps(config))
    # Real training, deterministic rejection: this test owns rollback recording,
    # not the separately tested faceoff scoring algorithm.
    monkeypatch.setattr(
        distributed_main,
        "validate_model_faceoff",
        lambda **kwargs: {
            "candidate_wins": 0,
            "champion_wins": 2,
            "promoted": False,
        },
    )
    path = distributed_main.launch(source, tmp_path / "runs", allow_dirty=True)
    trace = events(path / "trajectory.jsonl")
    artifacts = events(path / "artifacts.jsonl")
    validations = [e for e in trace if e["kind"] == "validation"]
    assert len(validations) == 2 and all(
        "value_loss" in e["metrics"] for e in validations
    )
    rollbacks = [e for e in trace if e["kind"] == "rollback"]
    assert len(rollbacks) == 2 // promotion_interval
    assert rollbacks[-1]["progress"]["optimizer_steps"] > 0
    assert rollbacks[-1]["context"]["active_state_progress"]["optimizer_steps"] == 0
    faceoffs = [e for e in trace if e["kind"] == "faceoff"]
    checkpoint_ids = {
        e["artifact_id"] for e in artifacts if e.get("artifact_kind") == "checkpoint"
    }
    assert all(
        e["context"]["candidate_artifact_id"] in checkpoint_ids for e in faceoffs
    )
    replays = [e for e in artifacts if e.get("artifact_kind") == "replay_data"]
    superseded = [e for e in artifacts if e["kind"] == "superseded"]
    assert len(replays) == 2 and len(superseded) == 1
    assert superseded[0]["artifact_id"] == replays[0]["artifact_id"]
    assert superseded[0]["sequence"] < replays[1]["sequence"]
    manifest = json.loads((path / replays[1]["path"] / "manifest.json").read_text())
    assert manifest["dataset_id"] == replays[1]["metadata"]["dataset_id"]
    assert manifest["source_checkpoint"] is None
    assert (
        replays[1]["metadata"]["previous_dataset_id"]
        == replays[0]["metadata"]["dataset_id"]
    )
    assert replays[1]["metadata"]["latest_generated_batch"]["game_count"] == 1
    assert manifest["game_count"] == 2
    # Saving more often cannot promote intermediate model or optimizer updates.
    checkpoints = [e for e in artifacts if e.get("artifact_kind") == "checkpoint"]
    initial = torch.load(path / checkpoints[0]["path"], weights_only=False)
    final = torch.load(path / checkpoints[-1]["path"], weights_only=False)
    assert all(
        torch.equal(value, final["model_state_dict"][key])
        for key, value in initial["model_state_dict"].items()
    )
    assert final["optimizer_state_dict"]["state"] == {}
