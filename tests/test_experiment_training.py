"""Recipe-level observational validation and mixed replay provenance."""

import json
import random
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
    training = experiment_training.TrainingState(progress)
    previous_id = initial_id
    for index in (1, 2):
        replay.add(state, targets, game_index=index)
        generation = experiment_training.Snapshot(
            tmp_path / f"model-{index}.pth", f"model-{index}"
        )
        recording.save_replay(
            replay,
            state=training,
            generation=generation,
            generation_iteration=index,
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


def test_validation_preserves_model_mode_weights_and_randomness():
    model = torch.nn.Linear(1, 1)
    weights = {key: value.clone() for key, value in model.state_dict().items()}
    recording = experiment_training.RecipeRecording(
        None, SimpleNamespace(dataset_id=None)
    )
    state = experiment_training.TrainingState(checkpoint.TrainingProgress())
    rng = checkpoint.capture_rng_state()
    expected = (random.random(), np.random.random(), torch.rand(1))
    checkpoint.restore_rng_state(rng)

    def validate(model):
        assert not model.training
        assert not torch.is_grad_enabled()
        random.random()
        np.random.random()
        return {"prediction": model(torch.rand(1)).item()}

    recording.validation(model, validate, state)
    assert model.training
    torch.testing.assert_close(model.state_dict(), weights, rtol=0, atol=0)
    assert random.random() == expected[0]
    assert np.random.random() == expected[1]
    torch.testing.assert_close(torch.rand(1), expected[2], rtol=0, atol=0)
