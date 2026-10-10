from __future__ import annotations

import dataclasses
import typing
from collections import deque

import numpy as np

from skyjo.learning import batches


@dataclasses.dataclass(frozen=True, slots=True)
class TargetShapeSpec:
    name: str
    shape: tuple[int, ...]


TargetSpecs: typing.TypeAlias = tuple[TargetShapeSpec, ...]
TargetSpecInput: typing.TypeAlias = typing.Sequence[TargetShapeSpec] | None


def core_target_specs(
    players: int,
    action_mask_shape: tuple[int, ...],
) -> TargetSpecs:
    return (
        TargetShapeSpec(
            name=batches.VALUE_TARGET_NAME,
            shape=(players,),
        ),
        TargetShapeSpec(
            name=batches.POLICY_TARGET_NAME,
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
    return tuple(target_specs)


@dataclasses.dataclass(frozen=True, slots=True)
class Config:
    max_size: int
    spatial_input_shape: tuple[int, ...]
    non_spatial_input_shape: tuple[int, ...]
    action_mask_shape: tuple[int, ...]
    target_specs: TargetSpecs | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class GameProvenance:
    game_index: int
    play_seed: int = -1
    target_seed: int = -1


@dataclasses.dataclass(frozen=True, slots=True)
class GameRecord:
    game_index: int
    play_seed: int
    target_seed: int
    start: int
    length: int


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
        )
        replay_buffer.max_size = capacity
        if capacity == position_count:
            replay_buffer.spatial_input_buffer = np.array(spatial_inputs, copy=True)
            replay_buffer.non_spatial_input_buffer = np.array(
                non_spatial_inputs, copy=True
            )
            replay_buffer.action_masks = np.array(action_masks, copy=True)
            replay_buffer.target_buffers = {
                name: np.array(values, copy=True)
                for name, values in target_arrays.items()
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
                name: np.empty((capacity, *values.shape[1:]), dtype=values.dtype)
                for name, values in target_arrays.items()
            }
            replay_buffer.spatial_input_buffer[:position_count] = spatial_inputs
            replay_buffer.non_spatial_input_buffer[:position_count] = non_spatial_inputs
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

    def append(self, batch: batches.TrainingBatch, provenance: GameProvenance) -> None:
        """Append one complete game, evicting complete older games as needed."""
        game_length = len(batch)
        game_index = provenance.game_index
        play_seed = provenance.play_seed
        target_seed = provenance.target_seed
        if game_length < 1:
            raise ValueError("cannot add an empty game")
        if game_length > self.max_size:
            raise ValueError(
                f"game has {game_length} positions, exceeding replay capacity "
                f"{self.max_size}"
            )
        if any(record.game_index == game_index for record in self._games):
            raise ValueError(f"game_index {game_index} is already in the buffer")
        # Prepare the whole game before eviction: a rejected append is a no-op.
        if set(batch.targets) != set(self.target_names):
            raise ValueError("Training batch targets do not match replay targets")
        arrays = [
            batch.spatial_inputs,
            batch.non_spatial_inputs,
            batch.action_masks,
        ]
        destinations = [
            self.spatial_input_buffer,
            self.non_spatial_input_buffer,
            self.action_masks,
        ]
        arrays.extend(batch.targets[name] for name in self.target_names)
        destinations.extend(self.target_buffers[name] for name in self.target_names)
        for array, destination in zip(arrays, destinations, strict=True):
            expected = (game_length, *destination.shape[1:])
            if array.shape != expected:
                raise ValueError(f"Expected replay shape {expected}, got {array.shape}")

        # Convert every input before eviction so a bad dtype cannot partially append.
        arrays = [
            np.asarray(array, dtype=destination.dtype)
            for array, destination in zip(arrays, destinations, strict=True)
        ]

        while self._games and self._size + game_length > self.max_size:
            evicted = self._games.popleft()
            self._size -= evicted.length
        start = self._write_index
        indices = (start + np.arange(game_length)) % self.max_size
        for array, destination in zip(arrays, destinations, strict=True):
            destination[indices] = array

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

    def sample_batch(
        self, batch_size: int, *, rng: np.random.Generator
    ) -> batches.TrainingBatch:
        if not self:
            raise ValueError("ReplayBuffer is empty")
        if batch_size < 1:
            raise ValueError("batch_size must be at least one")
        logical_indices = rng.choice(len(self), size=batch_size, replace=True)
        indices = self._logical_to_physical(logical_indices)
        return batches.TrainingBatch(
            self.spatial_input_buffer[indices],
            self.non_spatial_input_buffer[indices],
            self.action_masks[indices],
            {
                name: target_buffer[indices]
                for name, target_buffer in self.target_buffers.items()
            },
        )

    def ordered_batch(self) -> batches.TrainingBatch:
        """Return all retained positions in chronological game order."""
        return self.batch_range(0, len(self))

    def batch_range(self, start: int, stop: int) -> batches.TrainingBatch:
        """Return a chronological half-open range of retained positions."""
        if not 0 <= start <= stop <= len(self):
            raise IndexError(
                f"invalid replay range [{start}, {stop}) for size {len(self)}"
            )
        return self.batch_indices(np.arange(start, stop, dtype=np.int64))

    def batch_indices(self, logical_indices: np.ndarray) -> batches.TrainingBatch:
        logical_indices = np.asarray(logical_indices, dtype=np.int64)
        if (
            logical_indices.ndim != 1
            or np.any(logical_indices < 0)
            or np.any(logical_indices >= len(self))
        ):
            raise IndexError("Replay indices out of range")
        indices = self._logical_to_physical(logical_indices)
        return batches.TrainingBatch(
            self.spatial_input_buffer[indices],
            self.non_spatial_input_buffer[indices],
            self.action_masks[indices],
            {
                name: target_buffer[indices]
                for name, target_buffer in self.target_buffers.items()
            },
        )

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
