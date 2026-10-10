"""Recipe-level mixed replay provenance."""

import json

import numpy as np

from skyjo.learning import buffer, batches, replay_io
from skyjo.experiments.artifacts import RunArtifacts
from skyjo.learning import checkpoint
from skyjo.experiments import experiment_training
from skyjo.engine import game
from skyjo.learning import observations


def test_replay_provenance_keeps_initial_buffer_reference_and_latest_batch_separate(
    tmp_path,
):
    replay = buffer.ReplayBuffer(
        max_size=8,
        spatial_input_shape=(2, game.ROW_COUNT, game.COLUMN_COUNT, game.FINGER_SIZE),
        non_spatial_input_shape=observations.get_non_spatial_input_shape(2),
        action_mask_shape=(game.MASK_SIZE,),
    )
    state = game.new(players=2, top=0)
    mask = game.actions(state).astype(np.float32)
    targets = {
        "value": np.array([1.0, 0.0], dtype=np.float32),
        "policy": mask / mask.sum(),
    }
    encoded = batches.states_to_batch([state])
    batch = batches.TrainingBatch(
        encoded.spatial_inputs,
        encoded.non_spatial_inputs,
        encoded.action_masks,
        {name: value[None] for name, value in targets.items()},
    )
    replay.append(batch, buffer.GameProvenance(0))
    original = replay_io.save(replay, tmp_path / "initial")
    initial_id = replay.dataset_id
    original_manifest = (original / "manifest.json").read_bytes()
    output_path = tmp_path / "output"
    recording = RunArtifacts(
        None, output_path, initial_dataset=original, dataset_id=initial_id
    )
    progress = checkpoint.TrainingProgress()
    training = experiment_training.TrainingState(progress)
    previous_id = initial_id
    for index in (1, 2):
        replay.append(batch, buffer.GameProvenance(index))
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
        manifest = json.loads((output_path / "manifest.json").read_text())
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
