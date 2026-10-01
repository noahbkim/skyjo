import pathlib

import numpy as np
import pytest

from skyjo import buffer, skynet, train_utils
from skyjo import game as sj
from skyjo import play


def make_game_data(length: int, marker: float = 0.0) -> play.GameData:
    state = sj.new(players=2, top=0)
    action_mask = sj.actions(state).astype(np.float32)
    policy = action_mask / action_mask.sum()
    return [
        play.GameDataPoint(
            state,
            None,
            {
                train_utils.VALUE_TARGET_NAME: np.array(
                    [marker, 1.0 - marker], dtype=np.float32
                ),
                train_utils.POLICY_TARGET_NAME: policy,
            },
        )
        for _ in range(length)
    ]


def make_replay_buffer(max_size: int = 32) -> buffer.ReplayBuffer:
    return buffer.ReplayBuffer(
        max_size=max_size,
        spatial_input_shape=(2, sj.ROW_COUNT, sj.COLUMN_COUNT, sj.FINGER_SIZE),
        non_spatial_input_shape=skynet.get_non_spatial_input_shape(2),
        action_mask_shape=(sj.MASK_SIZE,),
    )


def test_replay_buffer_defaults_to_core_target_specs():
    replay_buffer = buffer.ReplayBuffer(
        max_size=16,
        spatial_input_shape=(3, 4, 4, 17),
        non_spatial_input_shape=(10,),
        action_mask_shape=(23,),
    )

    assert replay_buffer.target_names == train_utils.CORE_TARGET_NAMES
    assert replay_buffer.target_specs == (
        buffer.TargetShapeSpec(
            name=train_utils.VALUE_TARGET_NAME,
            shape=(3,),
        ),
        buffer.TargetShapeSpec(
            name=train_utils.POLICY_TARGET_NAME,
            shape=(23,),
        ),
    )


def test_replay_buffer_from_config_preserves_path_with_default_target_specs():
    path = pathlib.Path("data/training_data/test-buffer.pkl")
    config = buffer.Config(
        max_size=16,
        spatial_input_shape=(2, 4, 4, 17),
        non_spatial_input_shape=(10,),
        action_mask_shape=(12,),
        path=path,
    )

    replay_buffer = buffer.ReplayBuffer.from_config(config)

    assert replay_buffer.path == path
    assert replay_buffer.target_names == train_utils.CORE_TARGET_NAMES


def test_replay_buffer_accepts_round_score_aux_target_spec():
    replay_buffer = buffer.ReplayBuffer(
        max_size=16,
        spatial_input_shape=(2, 4, 4, 17),
        non_spatial_input_shape=(10,),
        action_mask_shape=(23,),
        target_specs=(
            buffer.TargetShapeSpec(
                name=train_utils.VALUE_TARGET_NAME,
                shape=(2,),
            ),
            buffer.TargetShapeSpec(
                name=train_utils.POLICY_TARGET_NAME,
                shape=(23,),
            ),
            buffer.TargetShapeSpec(
                name=train_utils.ROUND_SCORE_TARGET_NAME,
                shape=(2,),
            ),
        ),
    )

    assert replay_buffer.target_names == (
        train_utils.VALUE_TARGET_NAME,
        train_utils.POLICY_TARGET_NAME,
        train_utils.ROUND_SCORE_TARGET_NAME,
    )


def test_replay_buffer_evicts_complete_games_and_rejects_oversize_game():
    replay_buffer = make_replay_buffer(max_size=6)
    replay_buffer.add_game_data(
        make_game_data(3, 0.1),
        game_index=10,
        play_seed=100,
        target_seed=101,
    )
    replay_buffer.add_game_data(
        make_game_data(4, 0.2),
        game_index=11,
        play_seed=110,
        target_seed=111,
    )

    assert len(replay_buffer) == 4
    assert replay_buffer.game_indices == (11,)

    replay_buffer.add_game_data(
        make_game_data(2, 0.3),
        game_index=12,
        play_seed=120,
        target_seed=121,
    )
    assert len(replay_buffer) == 6
    assert replay_buffer.game_indices == (11, 12)
    assert np.allclose(
        replay_buffer.ordered_batch().value_targets[:, 0],
        [0.2, 0.2, 0.2, 0.2, 0.3, 0.3],
    )
    with pytest.raises(ValueError, match="exceeding replay capacity"):
        replay_buffer.add_game_data(make_game_data(7), game_index=13)


def test_dataset_round_trip_writes_only_populated_rows(tmp_path):
    replay_buffer = make_replay_buffer(max_size=100)
    replay_buffer.add_game_data(
        make_game_data(2, 0.1),
        game_index=20,
        play_seed=200,
        target_seed=201,
    )
    replay_buffer.add_game_data(
        make_game_data(3, 0.2),
        game_index=21,
        play_seed=210,
        target_seed=211,
    )
    expected = replay_buffer.ordered_batch()
    path = tmp_path / "dataset"

    replay_buffer.save(
        path,
        generation_metadata={"mcts_iterations": 4},
        source_checkpoint=pathlib.Path("models/source.pth"),
    )
    loaded = buffer.ReplayBuffer.load(path)

    assert loaded.dataset_id == replay_buffer.dataset_id
    assert len(loaded) == 5
    assert loaded.max_size == 5
    assert loaded.game_indices == (20, 21)
    assert [record.play_seed for record in loaded._games] == [200, 210]
    assert [record.target_seed for record in loaded._games] == [201, 211]
    actual = loaded.ordered_batch()
    assert np.array_equal(actual.spatial_inputs, expected.spatial_inputs)
    assert np.array_equal(actual.non_spatial_inputs, expected.non_spatial_inputs)
    assert np.array_equal(actual.action_masks, expected.action_masks)
    for name in expected.targets:
        assert np.array_equal(actual.targets[name], expected.targets[name])
    assert np.load(path / "spatial_inputs.npy", allow_pickle=False).shape[0] == 5
    assert loaded.dataset_metadata["generation_metadata"] == {
        "mcts_iterations": 4
    }

    replay_buffer.add_game_data(
        make_game_data(1, 0.4),
        game_index=22,
        play_seed=220,
        target_seed=221,
    )
    replay_buffer.save(path, generation_metadata={"mcts_iterations": 5})
    overwritten = buffer.ReplayBuffer.load(path)
    assert overwritten.game_indices == (20, 21, 22)
    assert overwritten.dataset_metadata["generation_metadata"] == {
        "mcts_iterations": 5
    }

    resumed = buffer.ReplayBuffer.from_config_or_load(
        buffer.Config(
            max_size=100,
            spatial_input_shape=(2, sj.ROW_COUNT, sj.COLUMN_COUNT, sj.FINGER_SIZE),
            non_spatial_input_shape=skynet.get_non_spatial_input_shape(2),
            action_mask_shape=(sj.MASK_SIZE,),
            path=path,
        )
    )
    assert resumed.max_size == 100
    assert len(resumed) == 6
    resumed.add_game_data(make_game_data(1, 0.5), game_index=23)
    assert resumed.game_indices == (20, 21, 22, 23)


def test_dataset_subset_and_split_are_deterministic_and_game_disjoint(tmp_path):
    replay_buffer = make_replay_buffer(max_size=32)
    for game_index in range(6):
        replay_buffer.add_game_data(
            make_game_data(2, game_index / 10),
            game_index=game_index,
            play_seed=game_index * 10,
            target_seed=game_index * 10 + 1,
        )
    path = replay_buffer.save(tmp_path / "dataset")

    subset_a = buffer.ReplayBuffer.load(path, max_games=3, subset_seed=7)
    subset_b = buffer.ReplayBuffer.load(path, max_games=3, subset_seed=7)
    assert subset_a.game_indices == subset_b.game_indices

    training, validation = replay_buffer.split_by_game(0.34, seed=9)
    assert set(training.game_indices).isdisjoint(validation.game_indices)
    assert set(training.game_indices) | set(validation.game_indices) == set(
        replay_buffer.game_indices
    )


def test_dataset_rejects_empty_and_unsupported_version(tmp_path):
    replay_buffer = make_replay_buffer()
    with pytest.raises(ValueError, match="empty"):
        replay_buffer.save(tmp_path / "empty")

    replay_buffer.add_game_data(make_game_data(1), game_index=0)
    path = replay_buffer.save(tmp_path / "dataset")
    manifest_path = path / buffer.MANIFEST_FILE
    manifest = __import__("json").loads(manifest_path.read_text())
    manifest["version"] = 999
    manifest_path.write_text(__import__("json").dumps(manifest))
    with pytest.raises(buffer.DatasetFormatError, match="unsupported"):
        buffer.ReplayBuffer.load(path)
