"""Concrete observation batches and named training targets."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from . import game, observations

if TYPE_CHECKING:
    from . import play

VALUE_TARGET_NAME = "value"
POLICY_TARGET_NAME = "policy"
CORE_TARGET_NAMES = (VALUE_TARGET_NAME, POLICY_TARGET_NAME)
TargetArrays = dict[str, np.ndarray]
TensorTargets = dict[str, torch.Tensor]


@dataclass(frozen=True, slots=True)
class TrainingBatch:
    spatial_inputs: np.ndarray
    non_spatial_inputs: np.ndarray
    action_masks: np.ndarray
    targets: TargetArrays = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TensorBatch:
    spatial_inputs: torch.Tensor
    non_spatial_inputs: torch.Tensor
    action_masks: torch.Tensor
    targets: TensorTargets


def to_tensors(batch: TrainingBatch, *, device: torch.device) -> TensorBatch:
    """Copy a batch to float32 tensors on the model's device."""

    def tensor(array):
        return torch.tensor(array, dtype=torch.float32, device=device)

    return TensorBatch(
        tensor(batch.spatial_inputs),
        tensor(batch.non_spatial_inputs),
        tensor(batch.action_masks),
        {name: tensor(value) for name, value in batch.targets.items()},
    )


def states_to_batch(states: Sequence[game.Skyjo]) -> TrainingBatch:
    """Encode a nonempty batch of states using the persisted observation layout."""
    return TrainingBatch(
        np.stack([observations.get_spatial_state_numpy(state) for state in states]),
        np.stack([observations.get_non_spatial_state_numpy(state) for state in states]),
        np.stack([observations.action_mask(state) for state in states]),
    )


def game_data_to_training_batch(
    game_data: play.GameData,
    target_names: Sequence[str] = CORE_TARGET_NAMES,
) -> TrainingBatch:
    inputs = states_to_batch([point.state for point in game_data])
    return TrainingBatch(
        inputs.spatial_inputs,
        inputs.non_spatial_inputs,
        inputs.action_masks,
        {
            name: np.stack(
                [
                    np.asarray(point.targets[name], dtype=np.float32)
                    for point in game_data
                ]
            )
            for name in target_names
        },
    )
