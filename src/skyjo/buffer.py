from __future__ import annotations

import dataclasses
import json
import pathlib
import shutil
import subprocess
import typing
import uuid
from collections import deque

import numpy as np

from . import checkpoint
from . import config
from . import game as sj
from . import play
from . import skynet
from . import train_utils

DATASET_FORMAT = "skyjo.replay-dataset"
DATASET_VERSION = 1
MANIFEST_FILE = "manifest.json"


class DatasetFormatError(ValueError):
    """Raised when a replay dataset is missing or has an unsupported format."""


@dataclasses.dataclass(frozen=True, slots=True)
class TargetShapeSpec:
    name: str
    shape: tuple[int, ...]


TargetSpecs: typing.TypeAlias = tuple[TargetShapeSpec, ...]
TargetSpecInput: typing.TypeAlias = (
    TargetSpecs | list[TargetShapeSpec | dict[str, typing.Any]] | None
)


def core_target_specs(
    players: int,
    action_mask_shape: tuple[int, ...],
) -> TargetSpecs:
    return (
        TargetShapeSpec(
            name=train_utils.VALUE_TARGET_NAME,
            shape=(players,),
        ),
        TargetShapeSpec(
            name=train_utils.POLICY_TARGET_NAME,
            shape=action_mask_shape,
        ),
    )


def default_target_specs(
    spatial_input_shape: tuple[int, ...],
    action_mask_shape: tuple[int, ...],
) -> TargetSpecs:
    assert len(spatial_input_shape) > 0, (
        "spatial_input_shape must include the player dimension"
    )
    return core_target_specs(
        players=spatial_input_shape[0],
        action_mask_shape=action_mask_shape,
    )


def resolve_target_specs(
    target_specs: TargetSpecInput,
    *,
    spatial_input_shape: tuple[int, ...],
    action_mask_shape: tuple[int, ...],
) -> TargetSpecs:
    if target_specs is None:
        return default_target_specs(spatial_input_shape, action_mask_shape)
    return tuple(
        spec if isinstance(spec, TargetShapeSpec) else TargetShapeSpec(**spec)
        for spec in target_specs
    )


@dataclasses.dataclass(slots=True)
class Config(config.Config):
    max_size: int
    spatial_input_shape: tuple[int, ...]
    non_spatial_input_shape: tuple[int, ...]
    action_mask_shape: tuple[int, ...]
    target_specs: TargetSpecs | None = None
    path: pathlib.Path | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class GameRecord:
    game_index: int
    play_seed: int
    target_seed: int
    start: int
    length: int


def _git_provenance() -> dict[str, typing.Any]:
    repository = pathlib.Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}
    return {"revision": revision, "dirty": dirty}


def _remove_artifact_path(path: pathlib.Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


class ReplayBuffer:
    """A fixed-capacity, complete-game replay buffer.

    Games are retained and evicted as units. Arrays use circular storage in
    memory, while persisted datasets are written in chronological game order.
    """

    def __init__(
        self,
        max_size: int,
        spatial_input_shape: tuple[int, ...],
        non_spatial_input_shape: tuple[int, ...],
        action_mask_shape: tuple[int, ...],
        target_specs: TargetSpecInput = None,
        path: pathlib.Path | None = None,
    ):
        if max_size < 1:
            raise ValueError("max_size must be at least one")
        self.max_size = max_size
        self.target_specs = resolve_target_specs(
            target_specs,
            spatial_input_shape=spatial_input_shape,
            action_mask_shape=action_mask_shape,
        )
        self.target_names = tuple(spec.name for spec in self.target_specs)
        if len(set(self.target_names)) != len(self.target_names):
            raise ValueError("target names must be unique")
        self.spatial_input_buffer = np.empty(
            (max_size, *spatial_input_shape), dtype=np.float32
        )
        self.non_spatial_input_buffer = np.empty(
            (max_size, *non_spatial_input_shape), dtype=np.float32
        )
        self.action_masks = np.empty((max_size, *action_mask_shape), dtype=np.float32)
        self.target_buffers = {
            spec.name: np.empty((max_size, *spec.shape), dtype=np.float32)
            for spec in self.target_specs
        }
        self.count = 0
        self._size = 0
        self._write_index = 0
        self._games: deque[GameRecord] = deque()
        self._next_game_index = 0
        self.path = path
        self.dataset_id: str | None = None
        self.dataset_metadata: dict[str, typing.Any] = {}

    @classmethod
    def from_config(cls, config: Config) -> typing.Self:
        return cls(
            config.max_size,
            config.spatial_input_shape,
            config.non_spatial_input_shape,
            config.action_mask_shape,
            config.target_specs,
            config.path,
        )

    @classmethod
    def from_config_or_load(cls, config: Config) -> typing.Self:
        """Resume the configured dataset when present, otherwise start empty."""
        if config.path is None or not (config.path / MANIFEST_FILE).is_file():
            return cls.from_config(config)
        replay_buffer = cls.load(config.path, capacity=config.max_size)
        expected_specs = resolve_target_specs(
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

    @classmethod
    def load(
        cls,
        path: pathlib.Path,
        *,
        max_games: int | None = None,
        subset_seed: int = 0,
        capacity: int | None = None,
    ) -> typing.Self:
        """Load a complete dataset or a deterministic subset of whole games."""
        manifest_path = path / MANIFEST_FILE
        try:
            with manifest_path.open(encoding="utf-8") as file:
                manifest = json.load(file)
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
                TargetShapeSpec(item["name"], tuple(item["shape"]))
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

        cls._validate_loaded_arrays(
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

        row_indices = cls._row_indices_for_games(game_offsets, selected_games)
        lengths = np.diff(game_offsets)[selected_games].astype(np.int64)
        selected_offsets = np.concatenate(
            (np.array([0], dtype=np.int64), np.cumsum(lengths, dtype=np.int64))
        )
        loaded = cls._from_arrays(
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
            path=path,
            capacity=capacity,
        )
        loaded.dataset_id = str(manifest["dataset_id"])
        loaded.dataset_metadata = dict(manifest)
        return loaded

    @staticmethod
    def _validate_loaded_arrays(
        *,
        manifest: dict[str, typing.Any],
        spatial_inputs: np.ndarray,
        non_spatial_inputs: np.ndarray,
        action_masks: np.ndarray,
        target_specs: TargetSpecs,
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
        arrays = (spatial_inputs, non_spatial_inputs, action_masks, *target_arrays.values())
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

    @staticmethod
    def _row_indices_for_games(
        game_offsets: np.ndarray,
        selected_games: np.ndarray,
    ) -> np.ndarray:
        if len(selected_games) == 0:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(
            [
                np.arange(game_offsets[index], game_offsets[index + 1])
                for index in selected_games
            ]
        )

    @classmethod
    def _from_arrays(
        cls,
        *,
        spatial_inputs: np.ndarray,
        non_spatial_inputs: np.ndarray,
        action_masks: np.ndarray,
        target_specs: TargetSpecs,
        target_arrays: dict[str, np.ndarray],
        game_offsets: np.ndarray,
        game_indices: np.ndarray,
        play_seeds: np.ndarray,
        target_seeds: np.ndarray,
        path: pathlib.Path | None,
        capacity: int | None = None,
    ) -> typing.Self:
        position_count = len(spatial_inputs)
        if position_count < 1:
            raise ValueError("cannot construct an empty ReplayBuffer")
        if capacity is None:
            capacity = position_count
        if capacity < position_count:
            raise ValueError(
                f"capacity {capacity} is smaller than the loaded position count "
                f"{position_count}"
            )
        replay_buffer = cls(
            max_size=1,
            spatial_input_shape=spatial_inputs.shape[1:],
            non_spatial_input_shape=non_spatial_inputs.shape[1:],
            action_mask_shape=action_masks.shape[1:],
            target_specs=target_specs,
            path=path,
        )
        replay_buffer.max_size = capacity
        if capacity == position_count:
            replay_buffer.spatial_input_buffer = np.asarray(spatial_inputs)
            replay_buffer.non_spatial_input_buffer = np.asarray(non_spatial_inputs)
            replay_buffer.action_masks = np.asarray(action_masks)
            replay_buffer.target_buffers = {
                name: np.asarray(values) for name, values in target_arrays.items()
            }
        else:
            replay_buffer.spatial_input_buffer = np.empty(
                (capacity, *spatial_inputs.shape[1:]), dtype=spatial_inputs.dtype
            )
            replay_buffer.non_spatial_input_buffer = np.empty(
                (capacity, *non_spatial_inputs.shape[1:]),
                dtype=non_spatial_inputs.dtype,
            )
            replay_buffer.action_masks = np.empty(
                (capacity, *action_masks.shape[1:]), dtype=action_masks.dtype
            )
            replay_buffer.target_buffers = {
                name: np.empty(
                    (capacity, *values.shape[1:]), dtype=values.dtype
                )
                for name, values in target_arrays.items()
            }
            replay_buffer.spatial_input_buffer[:position_count] = spatial_inputs
            replay_buffer.non_spatial_input_buffer[:position_count] = (
                non_spatial_inputs
            )
            replay_buffer.action_masks[:position_count] = action_masks
            for name, values in target_arrays.items():
                replay_buffer.target_buffers[name][:position_count] = values
        replay_buffer.count = position_count
        replay_buffer._size = position_count
        replay_buffer._write_index = position_count % capacity
        replay_buffer._games = deque(
            GameRecord(
                game_index=int(game_indices[index]),
                play_seed=int(play_seeds[index]),
                target_seed=int(target_seeds[index]),
                start=int(game_offsets[index]),
                length=int(game_offsets[index + 1] - game_offsets[index]),
            )
            for index in range(len(game_indices))
        )
        replay_buffer._next_game_index = int(np.max(game_indices)) + 1
        return replay_buffer

    def __len__(self) -> int:
        return self._size

    @property
    def game_count(self) -> int:
        return len(self._games)

    @property
    def game_indices(self) -> tuple[int, ...]:
        return tuple(record.game_index for record in self._games)

    def _oldest_start(self) -> int:
        if not self._games:
            raise ValueError("ReplayBuffer is empty")
        return self._games[0].start

    def _logical_to_physical(self, logical_indices: np.ndarray) -> np.ndarray:
        return (self._oldest_start() + logical_indices) % self.max_size

    def _ordered_physical_indices(self) -> np.ndarray:
        if not self:
            return np.empty(0, dtype=np.int64)
        return self._logical_to_physical(np.arange(len(self), dtype=np.int64))

    def add(
        self,
        game_state: sj.Skyjo,
        targets: typing.Any,
        *,
        game_index: int | None = None,
        play_seed: int = -1,
        target_seed: int = -1,
    ) -> None:
        """Add one position as a one-position game.

        Normal generation should use :meth:`add_game_data` so boundaries are
        retained correctly.
        """
        self.add_game_data(
            [play.GameDataPoint(game_state, None, targets)],
            game_index=game_index,
            play_seed=play_seed,
            target_seed=target_seed,
        )

    def add_game_data(
        self,
        game_data: play.GameData,
        *,
        game_index: int | None = None,
        play_seed: int = -1,
        target_seed: int = -1,
    ) -> None:
        """Add a complete game's training rows, evicting complete old games."""
        game_length = len(game_data)
        if game_length < 1:
            raise ValueError("cannot add an empty game")
        if game_length > self.max_size:
            raise ValueError(
                f"game has {game_length} positions, exceeding replay capacity "
                f"{self.max_size}"
            )
        if game_index is None:
            game_index = self._next_game_index
        if any(record.game_index == game_index for record in self._games):
            raise ValueError(f"game_index {game_index} is already in the buffer")
        self._next_game_index = max(self._next_game_index, game_index + 1)

        while self._games and self._size + game_length > self.max_size:
            evicted = self._games.popleft()
            self._size -= evicted.length

        start = self._write_index
        for offset, data_point in enumerate(game_data):
            index = (start + offset) % self.max_size
            normalized_targets = train_utils.normalize_numpy_targets(
                data_point.targets, self.target_names
            )
            self.spatial_input_buffer[index] = skynet.get_spatial_state_numpy(
                data_point.state
            )
            self.non_spatial_input_buffer[index] = (
                skynet.get_non_spatial_state_numpy(data_point.state)
            )
            self.action_masks[index] = sj.actions(data_point.state).astype(np.float32)
            for name, target_buffer in self.target_buffers.items():
                target_buffer[index] = normalized_targets[name]

        self._games.append(
            GameRecord(
                game_index=game_index,
                play_seed=play_seed,
                target_seed=target_seed,
                start=start,
                length=game_length,
            )
        )
        self._write_index = (start + game_length) % self.max_size
        self._size += game_length
        self.count += game_length

    def sample_element(self) -> train_utils.TrainingDataPoint:
        if not self:
            raise ValueError("ReplayBuffer is empty")
        logical_index = np.random.randint(len(self))
        index = int(
            self._logical_to_physical(np.array([logical_index], dtype=np.int64))[0]
        )
        return train_utils.TrainingDataPoint(
            self.spatial_input_buffer[index],
            self.non_spatial_input_buffer[index],
            self.action_masks[index],
            {
                name: target_buffer[index]
                for name, target_buffer in self.target_buffers.items()
            },
        )

    def sample_batch(
        self, batch_size: int, *, rng: np.random.Generator | None = None
    ) -> train_utils.TrainingBatch:
        if not self:
            raise ValueError("ReplayBuffer is empty")
        if batch_size < 1:
            raise ValueError("batch_size must be at least one")
        logical_indices = (np.random if rng is None else rng).choice(
            len(self), size=batch_size, replace=True
        )
        indices = self._logical_to_physical(logical_indices)
        return train_utils.TrainingBatch(
            self.spatial_input_buffer[indices],
            self.non_spatial_input_buffer[indices],
            self.action_masks[indices],
            {
                name: target_buffer[indices]
                for name, target_buffer in self.target_buffers.items()
            },
        )

    def generate_training_batches(
        self, batch_size: int, batch_count: int
    ) -> typing.Generator[train_utils.TrainingBatch, None, None]:
        for _ in range(batch_count):
            yield self.sample_batch(batch_size)

    def ordered_batch(self) -> train_utils.TrainingBatch:
        """Return all retained positions in chronological game order."""
        return self.batch_range(0, len(self))

    def batch_range(self, start: int, stop: int) -> train_utils.TrainingBatch:
        """Return a chronological half-open range of retained positions."""
        if not 0 <= start <= stop <= len(self):
            raise IndexError(
                f"invalid replay range [{start}, {stop}) for size {len(self)}"
            )
        return self.batch_indices(np.arange(start, stop, dtype=np.int64))

    def batch_indices(self, logical_indices: np.ndarray) -> train_utils.TrainingBatch:
        logical_indices = np.asarray(logical_indices, dtype=np.int64)
        if (
            logical_indices.ndim != 1
            or np.any(logical_indices < 0)
            or np.any(logical_indices >= len(self))
        ):
            raise IndexError("Replay indices out of range")
        indices = self._logical_to_physical(logical_indices)
        return train_utils.TrainingBatch(
            self.spatial_input_buffer[indices],
            self.non_spatial_input_buffer[indices],
            self.action_masks[indices],
            {
                name: target_buffer[indices]
                for name, target_buffer in self.target_buffers.items()
            },
        )

    def _save_ordered_array(
        self,
        path: pathlib.Path,
        values: np.ndarray,
        *,
        chunk_size: int = 65_536,
    ) -> None:
        destination = np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=values.dtype,
            shape=(len(self), *values.shape[1:]),
        )
        try:
            for start in range(0, len(self), chunk_size):
                stop = min(start + chunk_size, len(self))
                logical_indices = np.arange(start, stop, dtype=np.int64)
                physical_indices = self._logical_to_physical(logical_indices)
                destination[start:stop] = values[physical_indices]
        finally:
            destination.flush()
            del destination

    def select_games(self, game_positions: typing.Sequence[int]) -> typing.Self:
        """Create an in-memory view copy containing selected game positions."""
        selected = np.array(sorted(set(game_positions)), dtype=np.int64)
        if len(selected) < 1:
            raise ValueError("at least one game must be selected")
        if selected[0] < 0 or selected[-1] >= self.game_count:
            raise IndexError("selected game position is out of range")
        lengths = np.array([record.length for record in self._games], dtype=np.int64)
        offsets = np.concatenate(
            (np.array([0], dtype=np.int64), np.cumsum(lengths, dtype=np.int64))
        )
        logical_rows = self._row_indices_for_games(offsets, selected)
        physical_rows = self._logical_to_physical(logical_rows)
        selected_lengths = lengths[selected]
        selected_offsets = np.concatenate(
            (
                np.array([0], dtype=np.int64),
                np.cumsum(selected_lengths, dtype=np.int64),
            )
        )
        records = list(self._games)
        result = self._from_arrays(
            spatial_inputs=self.spatial_input_buffer[physical_rows],
            non_spatial_inputs=self.non_spatial_input_buffer[physical_rows],
            action_masks=self.action_masks[physical_rows],
            target_specs=self.target_specs,
            target_arrays={
                name: values[physical_rows]
                for name, values in self.target_buffers.items()
            },
            game_offsets=selected_offsets,
            game_indices=np.array(
                [records[index].game_index for index in selected], dtype=np.int64
            ),
            play_seeds=np.array(
                [records[index].play_seed for index in selected], dtype=np.int64
            ),
            target_seeds=np.array(
                [records[index].target_seed for index in selected], dtype=np.int64
            ),
            path=self.path,
        )
        result.dataset_id = self.dataset_id
        result.dataset_metadata = dict(self.dataset_metadata)
        return result

    def split_by_game(
        self,
        validation_fraction: float,
        *,
        seed: int,
    ) -> tuple[typing.Self, typing.Self]:
        """Return deterministic, disjoint train and validation buffers."""
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between zero and one")
        if self.game_count < 2:
            raise ValueError("at least two games are required for a validation split")
        validation_games = round(self.game_count * validation_fraction)
        validation_games = min(max(validation_games, 1), self.game_count - 1)
        generator = np.random.default_rng(seed)
        permutation = generator.permutation(self.game_count)
        validation_positions = np.sort(permutation[:validation_games])
        training_positions = np.sort(permutation[validation_games:])
        return (
            self.select_games(training_positions),
            self.select_games(validation_positions),
        )

    def save(
        self,
        path: pathlib.Path | None = None,
        *,
        generation_metadata: typing.Any = None,
        source_checkpoint: pathlib.Path | None = None,
    ) -> pathlib.Path:
        """Atomically save populated rows as a versioned NumPy dataset."""
        if path is None:
            path = self.path
        if path is None:
            raise ValueError(
                "path must be supplied to save() or ReplayBuffer initialization"
            )
        if not self:
            raise ValueError("cannot save an empty ReplayBuffer")

        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
        backup_path = path.parent / f".{path.name}.backup-{uuid.uuid4().hex}"
        temporary_path.mkdir()
        try:
            self._save_ordered_array(
                temporary_path / "spatial_inputs.npy",
                self.spatial_input_buffer,
            )
            self._save_ordered_array(
                temporary_path / "non_spatial_inputs.npy",
                self.non_spatial_input_buffer,
            )
            self._save_ordered_array(
                temporary_path / "action_masks.npy",
                self.action_masks,
            )
            targets_path = temporary_path / "targets"
            targets_path.mkdir()
            for name, values in self.target_buffers.items():
                self._save_ordered_array(
                    targets_path / f"{name}.npy",
                    values,
                )

            records = list(self._games)
            lengths = np.array([record.length for record in records], dtype=np.int64)
            game_offsets = np.concatenate(
                (np.array([0], dtype=np.int64), np.cumsum(lengths, dtype=np.int64))
            )
            game_indices = np.array(
                [record.game_index for record in records], dtype=np.int64
            )
            play_seeds = np.array(
                [record.play_seed for record in records], dtype=np.int64
            )
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
                "position_count": len(self),
                "game_count": self.game_count,
                "replay_capacity": self.max_size,
                "spatial_input_shape": list(
                    self.spatial_input_buffer.shape[1:]
                ),
                "non_spatial_input_shape": list(
                    self.non_spatial_input_buffer.shape[1:]
                ),
                "action_mask_shape": list(self.action_masks.shape[1:]),
                "target_specs": [
                    {"name": spec.name, "shape": list(spec.shape)}
                    for spec in self.target_specs
                ],
                "dtypes": {
                    "spatial_inputs": str(self.spatial_input_buffer.dtype),
                    "non_spatial_inputs": str(
                        self.non_spatial_input_buffer.dtype
                    ),
                    "action_masks": str(self.action_masks.dtype),
                    **{
                        f"target:{name}": str(values.dtype)
                        for name, values in self.target_buffers.items()
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
                "git": _git_provenance(),
            }
            with (temporary_path / MANIFEST_FILE).open(
                "w", encoding="utf-8"
            ) as file:
                json.dump(manifest, file, indent=2, sort_keys=True)
                file.write("\n")

            if path.exists():
                path.replace(backup_path)
            temporary_path.replace(path)
            if backup_path.exists():
                _remove_artifact_path(backup_path)
            self.dataset_id = dataset_id
            self.dataset_metadata = manifest
            self.path = path
        except BaseException:
            if temporary_path.exists():
                _remove_artifact_path(temporary_path)
            if backup_path.exists() and not path.exists():
                backup_path.replace(path)
            raise
        return path
