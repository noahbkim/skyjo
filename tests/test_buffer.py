import json

import numpy as np
import pytest

from skyjo.learning import batches, buffer, replay_io


def make_batch(length, marker=0.0):
    return batches.TrainingBatch(
        np.full((length, 2, 1), marker, dtype=np.float32),
        np.zeros((length, 3), dtype=np.float32),
        np.ones((length, 2), dtype=np.float32),
        {
            "value": np.tile([marker, 1 - marker], (length, 1)).astype(np.float32),
            "policy": np.full((length, 2), 0.5, dtype=np.float32),
        },
    )


def make_replay_buffer(max_size=32):
    return buffer.ReplayBuffer(max_size, (2, 1), (3,), (2,))


def test_eviction_preserves_complete_games_and_chronological_rows():
    replay = make_replay_buffer(6)
    for index, length in ((10, 3), (11, 4), (12, 2)):
        replay.append(make_batch(length, index / 100), buffer.GameProvenance(index))
    assert len(replay) == 6
    assert replay.game_indices == (11, 12)
    np.testing.assert_allclose(
        replay.ordered_batch().targets["value"][:, 0], [0.11] * 4 + [0.12] * 2
    )
    with pytest.raises(ValueError, match="exceeding replay capacity"):
        replay.append(make_batch(7), buffer.GameProvenance(13))
    assert replay.game_indices == (11, 12)


@pytest.mark.parametrize("invalid", ["missing", "shape", "dtype", "duplicate"])
def test_rejected_append_leaves_retained_rows_unchanged(invalid):
    replay = make_replay_buffer(2)
    replay.append(make_batch(2, 0.25), buffer.GameProvenance(3))
    before = replay.ordered_batch()
    incoming = make_batch(1)
    if invalid == "missing":
        incoming.targets.pop("policy")
    elif invalid == "shape":
        incoming.targets["value"] = np.array([1.0])
    elif invalid == "dtype":
        incoming.targets["value"] = np.array([["bad", "data"]])
    with pytest.raises(ValueError):
        replay.append(
            incoming, buffer.GameProvenance(3 if invalid == "duplicate" else 9)
        )
    assert replay.game_indices == (3,)
    assert replay.count == 2
    after = replay.ordered_batch()
    np.testing.assert_array_equal(after.spatial_inputs, before.spatial_inputs)
    for name in before.targets:
        np.testing.assert_array_equal(after.targets[name], before.targets[name])


def test_dataset_round_trip_subset_and_mutable_reimport(tmp_path):
    replay = make_replay_buffer(100)
    for index in range(6):
        replay.append(
            make_batch(2, index / 10),
            buffer.GameProvenance(index, index * 2, index * 2 + 1),
        )
    path = replay_io.save(replay, tmp_path / "dataset", provenance={"run_id": "test"})
    loaded = replay_io.load(path)
    assert loaded.dataset_id == replay.dataset_id
    assert loaded.dataset_metadata["provenance"] == {"run_id": "test"}
    assert len(loaded) == loaded.max_size == 12
    assert loaded.game_indices == replay.game_indices
    np.testing.assert_array_equal(
        loaded.ordered_batch().targets["value"], replay.ordered_batch().targets["value"]
    )
    assert np.load(path / "spatial_inputs.npy").shape[0] == 12
    # A loaded dataset with exactly-full capacity must still accept new games.
    loaded.append(make_batch(2, 0.9), buffer.GameProvenance(6))
    assert loaded.game_indices == (1, 2, 3, 4, 5, 6)
    replay_io.save(loaded, path)
    assert replay_io.load(path).game_indices == loaded.game_indices
    subset = replay_io.load(path, max_games=3, subset_seed=7)
    assert (
        subset.game_indices
        == replay_io.load(path, max_games=3, subset_seed=7).game_indices
    )
    train, valid = loaded.split_by_game(0.34, seed=9)
    assert set(train.game_indices).isdisjoint(valid.game_indices)
    assert set(train.game_indices) | set(valid.game_indices) == set(loaded.game_indices)
    expanded = replay_io.load_for_config(buffer.Config(100, (2, 1), (3,), (2,)), path)
    expanded.append(make_batch(2), buffer.GameProvenance(7))
    assert expanded.max_size == 100 and len(expanded) == 14


def test_dataset_rejects_empty_and_unsupported_version(tmp_path):
    replay = make_replay_buffer()
    with pytest.raises(replay_io.DatasetFormatError, match="could not read"):
        replay_io.load_for_config(
            buffer.Config(100, (2, 1), (3,), (2,)), tmp_path / "missing"
        )
    with pytest.raises(ValueError, match="empty"):
        replay_io.save(replay, tmp_path / "empty")
    replay.append(make_batch(1), buffer.GameProvenance(0))
    path = replay_io.save(replay, tmp_path / "dataset")
    manifest_path = path / replay_io.MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text())
    manifest["version"] = 999
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(replay_io.DatasetFormatError, match="unsupported"):
        replay_io.load(path)


def test_sampling_uses_only_supplied_generator():
    replay = make_replay_buffer()
    for index in range(6):
        replay.append(make_batch(1, index / 10), buffer.GameProvenance(index))
    first = replay.sample_batch(20, rng=np.random.default_rng(7))
    np.random.seed(999)
    second = replay.sample_batch(20, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(first.targets["value"], second.targets["value"])


def test_dataset_rejects_a_concurrent_replacement_while_reading(tmp_path, monkeypatch):
    replay = make_replay_buffer()
    replay.append(make_batch(1), buffer.GameProvenance(0))
    path = replay_io.save(replay, tmp_path / "dataset")
    original = np.load
    replaced = False

    def replacing_load(*args, **kwargs):
        nonlocal replaced
        result = original(*args, **kwargs)
        if not replaced:
            replaced = True
            manifest = path / replay_io.MANIFEST_FILE
            manifest.write_bytes(manifest.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(replay_io.np, "load", replacing_load)
    with pytest.raises(replay_io.DatasetFormatError, match="changed while loading"):
        replay_io.load(path)
