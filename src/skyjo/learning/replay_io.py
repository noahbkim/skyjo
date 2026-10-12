"""Atomic replay datasets; storage policy and provenance are supplied by callers."""

from __future__ import annotations

import json
import pathlib
import shutil
import typing
import uuid

import numpy as np

from . import buffer, checkpoint

DATASET_FORMAT = "skyjo.replay-dataset"
DATASET_VERSION = 1
MANIFEST_FILE = "manifest.json"


class DatasetFormatError(ValueError):
    """A replay dataset is incomplete or has an unsupported format."""


def _remove_artifact_path(path: pathlib.Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def load_for_config(config: buffer.Config, path: pathlib.Path) -> buffer.ReplayBuffer:
    """Import a persisted dataset into storage sized for the requested run."""
    replay_buffer = load(path, capacity=config.max_size)
    expected_specs = buffer.resolve_target_specs(
        config.target_specs,
        spatial_input_shape=config.spatial_input_shape,
        action_mask_shape=config.action_mask_shape,
    )
    actual_shapes = (
        replay_buffer.spatial_input_buffer.shape[1:],
        replay_buffer.non_spatial_input_buffer.shape[1:],
        replay_buffer.action_masks.shape[1:],
    )
    expected_shapes = (
        config.spatial_input_shape,
        config.non_spatial_input_shape,
        config.action_mask_shape,
    )
    if actual_shapes != expected_shapes or replay_buffer.target_specs != expected_specs:
        raise ValueError(
            "saved replay dataset does not match the configured shapes or targets"
        )
    return replay_buffer


def load(
    path: pathlib.Path,
    *,
    max_games: int | None = None,
    subset_seed: int = 0,
    capacity: int | None = None,
) -> buffer.ReplayBuffer:
    """Load a complete dataset or a deterministic subset of whole games."""
    manifest_path = path / MANIFEST_FILE
    try:
        source_manifest = manifest_path.read_bytes()
        manifest = json.loads(source_manifest)
    except (OSError, json.JSONDecodeError) as error:
        raise DatasetFormatError(f"could not read {manifest_path}: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("format") != DATASET_FORMAT:
        raise DatasetFormatError(f"{path} is not a {DATASET_FORMAT} dataset")
    if manifest.get("version") != DATASET_VERSION:
        raise DatasetFormatError(
            f"unsupported dataset version {manifest.get('version')!r}; "
            f"expected {DATASET_VERSION}"
        )

    try:
        target_specs = tuple(
            buffer.TargetShapeSpec(item["name"], tuple(item["shape"]))
            for item in manifest["target_specs"]
        )
        spatial_inputs = np.load(
            path / "spatial_inputs.npy",
            allow_pickle=False,
            mmap_mode="r",
        )
        non_spatial_inputs = np.load(
            path / "non_spatial_inputs.npy",
            allow_pickle=False,
            mmap_mode="r",
        )
        action_masks = np.load(
            path / "action_masks.npy",
            allow_pickle=False,
            mmap_mode="r",
        )
        target_arrays = {
            spec.name: np.load(
                path / "targets" / f"{spec.name}.npy",
                allow_pickle=False,
                mmap_mode="r",
            )
            for spec in target_specs
        }
        game_offsets = np.load(path / "game_offsets.npy", allow_pickle=False)
        game_indices = np.load(path / "game_indices.npy", allow_pickle=False)
        play_seeds = np.load(path / "play_seeds.npy", allow_pickle=False)
        target_seeds = np.load(path / "target_seeds.npy", allow_pickle=False)
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise DatasetFormatError(f"malformed dataset at {path}: {error}") from error

    _validate_loaded_arrays(
        manifest=manifest,
        spatial_inputs=spatial_inputs,
        non_spatial_inputs=non_spatial_inputs,
        action_masks=action_masks,
        target_specs=target_specs,
        target_arrays=target_arrays,
        game_offsets=game_offsets,
        game_indices=game_indices,
        play_seeds=play_seeds,
        target_seeds=target_seeds,
    )
    game_count = len(game_indices)
    if max_games is not None:
        if max_games < 1:
            raise ValueError("max_games must be at least one")
        if max_games > game_count:
            raise ValueError(
                f"max_games {max_games} exceeds dataset game count {game_count}"
            )
        generator = np.random.default_rng(subset_seed)
        selected_games = np.sort(
            generator.choice(game_count, size=max_games, replace=False)
        )
    else:
        selected_games = np.arange(game_count)

    row_indices = buffer.ReplayBuffer._row_indices_for_games(
        game_offsets, selected_games
    )
    lengths = np.diff(game_offsets)[selected_games].astype(np.int64)
    selected_offsets = np.concatenate(
        (np.array([0], dtype=np.int64), np.cumsum(lengths, dtype=np.int64))
    )
    loaded = buffer.ReplayBuffer._from_arrays(
        spatial_inputs=np.asarray(spatial_inputs[row_indices]),
        non_spatial_inputs=np.asarray(non_spatial_inputs[row_indices]),
        action_masks=np.asarray(action_masks[row_indices]),
        target_specs=target_specs,
        target_arrays={
            name: np.asarray(values[row_indices])
            for name, values in target_arrays.items()
        },
        game_offsets=selected_offsets,
        game_indices=np.asarray(game_indices[selected_games]).copy(),
        play_seeds=np.asarray(play_seeds[selected_games]).copy(),
        target_seeds=np.asarray(target_seeds[selected_games]).copy(),
        capacity=capacity,
    )
    if manifest_path.read_bytes() != source_manifest:
        raise DatasetFormatError("dataset changed while loading")
    loaded.dataset_id = str(manifest["dataset_id"])
    loaded.dataset_metadata = dict(manifest)
    return loaded


def _validate_loaded_arrays(
    *,
    manifest: dict[str, typing.Any],
    spatial_inputs: np.ndarray,
    non_spatial_inputs: np.ndarray,
    action_masks: np.ndarray,
    target_specs: buffer.TargetSpecs,
    target_arrays: dict[str, np.ndarray],
    game_offsets: np.ndarray,
    game_indices: np.ndarray,
    play_seeds: np.ndarray,
    target_seeds: np.ndarray,
) -> None:
    position_count = manifest.get("position_count")
    game_count = manifest.get("game_count")
    if not isinstance(position_count, int) or position_count < 1:
        raise DatasetFormatError("dataset position_count must be positive")
    if not isinstance(game_count, int) or game_count < 1:
        raise DatasetFormatError("dataset game_count must be positive")
    if not isinstance(manifest.get("dataset_id"), str):
        raise DatasetFormatError("dataset manifest is missing dataset_id")
    arrays = (
        spatial_inputs,
        non_spatial_inputs,
        action_masks,
        *target_arrays.values(),
    )
    if any(len(array) != position_count for array in arrays):
        raise DatasetFormatError("dataset arrays do not match position_count")
    expected_shapes = {
        "spatial_inputs": (
            position_count,
            *tuple(manifest.get("spatial_input_shape", ())),
        ),
        "non_spatial_inputs": (
            position_count,
            *tuple(manifest.get("non_spatial_input_shape", ())),
        ),
        "action_masks": (
            position_count,
            *tuple(manifest.get("action_mask_shape", ())),
        ),
    }
    for name, array in (
        ("spatial_inputs", spatial_inputs),
        ("non_spatial_inputs", non_spatial_inputs),
        ("action_masks", action_masks),
    ):
        if array.shape != expected_shapes[name]:
            raise DatasetFormatError(
                f"{name} has shape {array.shape}, expected {expected_shapes[name]}"
            )
    dtypes = manifest.get("dtypes")
    if not isinstance(dtypes, dict):
        raise DatasetFormatError("dataset manifest is missing dtypes")
    arrays_by_name = {
        "spatial_inputs": spatial_inputs,
        "non_spatial_inputs": non_spatial_inputs,
        "action_masks": action_masks,
        **{f"target:{name}": array for name, array in target_arrays.items()},
    }
    for name, array in arrays_by_name.items():
        if dtypes.get(name) != str(array.dtype):
            raise DatasetFormatError(
                f"{name} has dtype {array.dtype}, expected {dtypes.get(name)!r}"
            )
    for name, array in (
        ("game_offsets", game_offsets),
        ("game_indices", game_indices),
        ("play_seeds", play_seeds),
        ("target_seeds", target_seeds),
    ):
        if dtypes.get(name) != str(array.dtype):
            raise DatasetFormatError(
                f"{name} has dtype {array.dtype}, expected {dtypes.get(name)!r}"
            )
    if game_offsets.shape != (game_count + 1,):
        raise DatasetFormatError("game_offsets has the wrong shape")
    if (
        game_offsets[0] != 0
        or game_offsets[-1] != position_count
        or np.any(np.diff(game_offsets) <= 0)
    ):
        raise DatasetFormatError("game_offsets must describe non-empty games")
    for name, array in (
        ("game_indices", game_indices),
        ("play_seeds", play_seeds),
        ("target_seeds", target_seeds),
    ):
        if array.shape != (game_count,):
            raise DatasetFormatError(f"{name} has the wrong shape")
    if len(np.unique(game_indices)) != game_count:
        raise DatasetFormatError("game_indices must be unique")
    for spec in target_specs:
        expected_shape = (position_count, *spec.shape)
        if target_arrays[spec.name].shape != expected_shape:
            raise DatasetFormatError(
                f"target {spec.name!r} has shape "
                f"{target_arrays[spec.name].shape}, expected {expected_shape}"
            )


def _save_ordered_array(
    replay: buffer.ReplayBuffer,
    path: pathlib.Path,
    values: np.ndarray,
    *,
    chunk_size: int = 65_536,
) -> None:
    destination = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=values.dtype,
        shape=(len(replay), *values.shape[1:]),
    )
    try:
        for start in range(0, len(replay), chunk_size):
            stop = min(start + chunk_size, len(replay))
            logical_indices = np.arange(start, stop, dtype=np.int64)
            physical_indices = replay._logical_to_physical(logical_indices)
            destination[start:stop] = values[physical_indices]
    finally:
        destination.flush()
        del destination


def save(
    replay: buffer.ReplayBuffer,
    path: pathlib.Path,
    *,
    generation_metadata: typing.Any = None,
    source_checkpoint: pathlib.Path | None = None,
    provenance: dict[str, typing.Any] | None = None,
) -> pathlib.Path:
    """Atomically save populated rows as a versioned NumPy dataset."""
    if not replay:
        raise ValueError("cannot save an empty ReplayBuffer")

    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    backup_path = path.parent / f".{path.name}.backup-{uuid.uuid4().hex}"
    temporary_path.mkdir()
    try:
        _save_ordered_array(
            replay,
            temporary_path / "spatial_inputs.npy",
            replay.spatial_input_buffer,
        )
        _save_ordered_array(
            replay,
            temporary_path / "non_spatial_inputs.npy",
            replay.non_spatial_input_buffer,
        )
        _save_ordered_array(
            replay,
            temporary_path / "action_masks.npy",
            replay.action_masks,
        )
        targets_path = temporary_path / "targets"
        targets_path.mkdir()
        for name, values in replay.target_buffers.items():
            _save_ordered_array(
                replay,
                targets_path / f"{name}.npy",
                values,
            )

        records = list(replay._games)
        lengths = np.array([record.length for record in records], dtype=np.int64)
        game_offsets = np.concatenate(
            (np.array([0], dtype=np.int64), np.cumsum(lengths, dtype=np.int64))
        )
        game_indices = np.array(
            [record.game_index for record in records], dtype=np.int64
        )
        play_seeds = np.array([record.play_seed for record in records], dtype=np.int64)
        target_seeds = np.array(
            [record.target_seed for record in records], dtype=np.int64
        )
        np.save(temporary_path / "game_offsets.npy", game_offsets)
        np.save(temporary_path / "game_indices.npy", game_indices)
        np.save(temporary_path / "play_seeds.npy", play_seeds)
        np.save(temporary_path / "target_seeds.npy", target_seeds)

        dataset_id = uuid.uuid4().hex
        manifest = {
            "format": DATASET_FORMAT,
            "version": DATASET_VERSION,
            "dataset_id": dataset_id,
            "position_count": len(replay),
            "game_count": replay.game_count,
            "replay_capacity": replay.max_size,
            "spatial_input_shape": list(replay.spatial_input_buffer.shape[1:]),
            "non_spatial_input_shape": list(replay.non_spatial_input_buffer.shape[1:]),
            "action_mask_shape": list(replay.action_masks.shape[1:]),
            "target_specs": [
                {"name": spec.name, "shape": list(spec.shape)}
                for spec in replay.target_specs
            ],
            "dtypes": {
                "spatial_inputs": str(replay.spatial_input_buffer.dtype),
                "non_spatial_inputs": str(replay.non_spatial_input_buffer.dtype),
                "action_masks": str(replay.action_masks.dtype),
                **{
                    f"target:{name}": str(values.dtype)
                    for name, values in replay.target_buffers.items()
                },
                "game_offsets": str(game_offsets.dtype),
                "game_indices": str(game_indices.dtype),
                "play_seeds": str(play_seeds.dtype),
                "target_seeds": str(target_seeds.dtype),
            },
            "generation_metadata": checkpoint.normalize_configuration(
                generation_metadata
            ),
            "source_checkpoint": (
                None if source_checkpoint is None else str(source_checkpoint)
            ),
            "provenance": provenance,
        }
        with (temporary_path / MANIFEST_FILE).open("w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2, sort_keys=True)
            file.write("\n")

        if path.exists():
            path.replace(backup_path)
        temporary_path.replace(path)
        if backup_path.exists():
            _remove_artifact_path(backup_path)
        replay.dataset_id = dataset_id
        replay.dataset_metadata = manifest
    except BaseException:
        if temporary_path.exists():
            _remove_artifact_path(temporary_path)
        if backup_path.exists() and not path.exists():
            backup_path.replace(path)
        raise
    return path
