"""Concrete observation batches and named training targets."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

from skyjo.engine import game
from skyjo.learning import observations

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

    def __len__(self) -> int:
        return len(self.spatial_inputs)

    def __getitem__(self, indices) -> TrainingBatch:
        """Select positions with a slice or an array of indices."""
        return TrainingBatch(
            self.spatial_inputs[indices],
            self.non_spatial_inputs[indices],
            self.action_masks[indices],
            {name: values[indices] for name, values in self.targets.items()},
        )


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
