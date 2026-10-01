"""Recipe-level state transitions and mixed replay provenance."""

import json
from types import SimpleNamespace

import numpy as np
import torch

from skyjo import buffer, checkpoint, experiment_training, game, skynet


def test_replay_provenance_keeps_initial_buffer_reference_and_latest_batch_separate(
    tmp_path,
):
    replay = buffer.ReplayBuffer(
        max_size=8,
        spatial_input_shape=(2, game.ROW_COUNT, game.COLUMN_COUNT, game.FINGER_SIZE),
        non_spatial_input_shape=skynet.get_non_spatial_input_shape(2),
        action_mask_shape=(game.MASK_SIZE,),
    )
    state = game.new(players=2, top=0)
    mask = game.actions(state).astype(np.float32)
    targets = {
        "value": np.array([1.0, 0.0], dtype=np.float32),
        "policy": mask / mask.sum(),
    }
    replay.add(state, targets, game_index=0)
    original = replay.save(tmp_path / "initial")
    initial_id = replay.dataset_id
    original_manifest = (original / "manifest.json").read_bytes()
    replay.path = tmp_path / "output"
    recording = experiment_training.RecipeRecording(
        None, replay, initial_dataset_path=original
    )
    progress = checkpoint.TrainingProgress()
    training = experiment_training.TrainingState(progress, progress)
    previous_id = initial_id
    for index in (1, 2):
        replay.add(state, targets, game_index=index)
        generation = experiment_training.Snapshot(
            tmp_path / f"model-{index}.pth", f"model-{index}", progress
        )
        recording.save_replay(
            replay,
            state=training,
            generation=generation,
            game_count=1,
            generation_settings={"seed": index},
        )
        manifest = json.loads((replay.path / "manifest.json").read_text())
        metadata = manifest["generation_metadata"]
        assert metadata["initial_buffer"] == {
            "path": str(original),
            "dataset_id": initial_id,
        }
        assert metadata["previous_dataset_id"] == previous_id
        assert (
            metadata["latest_generated_batch"]["checkpoint_artifact_id"]
            == f"model-{index}"
        )
        assert metadata["latest_generated_batch"]["game_count"] == 1
        assert manifest["game_count"] == index + 1
        assert manifest["source_checkpoint"] is None
        assert "game_indices" not in json.dumps(metadata)
        previous_id = manifest["dataset_id"]
    assert (original / "manifest.json").read_bytes() == original_manifest


def test_promotion_rejection_restores_last_accepted_model_and_optimizer(tmp_path):
    model = torch.nn.Linear(1, 1)
    model.device = torch.device("cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    recording = experiment_training.RecipeRecording(
        None, SimpleNamespace(dataset_id=None)
    )
    sequence = 0

    def save_model(model, **kwargs):
        nonlocal sequence
        sequence += 1
        return checkpoint.save_checkpoint(
            tmp_path / f"checkpoint-{sequence}.pth", model=model, **kwargs
        )

    factory = SimpleNamespace(save_model=save_model)
    progress = checkpoint.TrainingProgress()
    state = experiment_training.TrainingState(progress, progress)
    champion = recording.save_snapshot(
        factory, model, optimizer, {}, state, role="champion"
    )
    accepted_weights = None
    for iteration, accept in enumerate((True, False), start=1):
        optimizer.zero_grad()
        model(torch.ones(1, 1)).square().sum().backward()
        optimizer.step()
        state = state.generated(games=1, positions=1).trained(
            iteration=iteration, steps=1, batch_size=1
        )
        result = experiment_training.promote_candidate(
            model=model,
            optimizer=optimizer,
            factory=factory,
            configuration={},
            state=state,
            champion=champion,
            recording=recording,
            protocol={},
            evaluate=lambda: {
                "candidate_wins": int(accept),
                "champion_wins": int(not accept),
                "promoted": accept,
            },
        )
        state, champion = result.state, result.champion
        if accept:
            accepted_weights = {
                key: value.clone() for key, value in model.state_dict().items()
            }
    assert state.work.optimizer_steps == 2 and state.retained.optimizer_steps == 1
    assert champion.retained.optimizer_steps == 1
    assert all(
        torch.equal(value, model.state_dict()[key])
        for key, value in accepted_weights.items()
    )
    assert all(value["step"].item() == 1 for value in optimizer.state.values())
