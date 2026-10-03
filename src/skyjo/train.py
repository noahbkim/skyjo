"""
Module to Train Skyjo models
"""

import dataclasses
import math
import time
import typing

import numpy as np
import torch

from . import buffer, config, gradient_diagnostic, skynet, train_utils

# MARK: Training


@dataclasses.dataclass(slots=True)
class ReplayRatioTrainConfig(config.Config):
    """Training budget expressed as sampled positions per new position."""

    batch_size: int
    replay_ratio: float
    loss_function: train_utils.LossFunction
    learn_rate: float
    gradient_diagnostic: bool = False
    diagnostic_done: bool = dataclasses.field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least one")
        if self.replay_ratio <= 0:
            raise ValueError("replay_ratio must be positive")


@dataclasses.dataclass(frozen=True)
class TrainingResult:
    losses: list[dict]
    diagnostics: dict[str, float | int]
    steps: int
    sampled_positions: int
    seconds: float
    gradient_scales: dict | None = None


def train_iteration(
    model: skynet.SkyNet,
    replay: buffer.ReplayBuffer,
    optimizer: torch.optim.Optimizer,
    config: ReplayRatioTrainConfig,
    new_positions: int,
) -> TrainingResult:
    """Allocate a replay-ratio budget and collect detached diagnostics."""
    steps = math.ceil(new_positions * config.replay_ratio / config.batch_size)
    diagnostics = train_utils.TrainingDiagnostics()
    started = time.perf_counter()
    scales = {} if config.gradient_diagnostic and not config.diagnostic_done else None
    losses = train_steps(
        model,
        replay,
        training_batch_size=config.batch_size,
        optimizer_steps=steps,
        optimizer=optimizer,
        loss_function=config.loss_function,
        diagnostics=diagnostics,
        gradient_scales=scales,
    )
    if scales:
        config.diagnostic_done = True
    return TrainingResult(
        losses,
        diagnostics.summary(),
        steps,
        steps * config.batch_size,
        time.perf_counter() - started,
        scales,
    )


def train_step(
    model: skynet.SkyNet,
    batch: train_utils.TrainingBatch,
    loss_function: train_utils.LossFunction,
    optimizer: torch.optim.Optimizer,
    diagnostics: train_utils.TrainingDiagnostics | None = None,
    gradient_scales: dict | None = None,
) -> tuple[float, train_utils.LossDetails]:
    """Performs a single training step on the model."""
    model.train()
    spatial_inputs_tensor = torch.tensor(
        batch.spatial_inputs, dtype=torch.float32, device=model.device
    )
    non_spatial_inputs_tensor = torch.tensor(
        batch.non_spatial_inputs, dtype=torch.float32, device=model.device
    )
    masks_tensor = torch.tensor(
        batch.action_masks, dtype=torch.float32, device=model.device
    )
    tensor_targets = train_utils.numpy_targets_to_tensors(
        batch.targets,
        device=model.device,
    )
    model_output = model(spatial_inputs_tensor, non_spatial_inputs_tensor, masks_tensor)
    loss, loss_detail = loss_function(model_output, tensor_targets)
    if diagnostics is not None:
        diagnostics.update(
            model_output.policy_logits, tensor_targets.policy, masks_tensor
        )
    if gradient_scales is not None:
        gradient_scales.update(
            gradient_diagnostic.measure(
                model,
                model_output,
                tensor_targets,
                **getattr(loss_function, "keywords", {}),
            )
        )
    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), {"total_loss": loss.item(), **loss_detail}


def train_steps(
    model: skynet.SkyNet,
    training_data_buffer: buffer.ReplayBuffer,
    training_batch_size: int,
    optimizer_steps: int,
    optimizer: torch.optim.Optimizer,
    loss_function: train_utils.LossFunction,
    diagnostics: train_utils.TrainingDiagnostics | None = None,
    gradient_scales: dict | None = None,
    sampling_rng: np.random.Generator | None = None,
) -> list[train_utils.LossDetails]:
    """Run exactly ``optimizer_steps`` updates sampled from the replay buffer."""
    if optimizer_steps < 0:
        raise ValueError("optimizer_steps cannot be negative")
    if training_batch_size < 1:
        raise ValueError("training_batch_size must be at least one")
    loss_details = []
    for _ in range(optimizer_steps):
        batch = training_data_buffer.sample_batch(
            batch_size=training_batch_size,
            **({"rng": sampling_rng} if sampling_rng is not None else {}),
        )
        _, step_loss_details = train_step(
            model,
            batch,
            loss_function,
            optimizer,
            diagnostics,
            gradient_scales,
        )
        gradient_scales = None
        loss_details.append(step_loss_details)
    return loss_details


@torch.inference_mode()
def evaluate_loss(
    model: skynet.SkyNet,
    evaluation_data_buffer: buffer.ReplayBuffer,
    evaluation_batch_size: int,
    loss_function: train_utils.LossFunction,
) -> train_utils.LossDetails:
    """Evaluate position-weighted total and component losses."""
    if not evaluation_data_buffer:
        raise ValueError("evaluation_data_buffer cannot be empty")
    if evaluation_batch_size < 1:
        raise ValueError("evaluation_batch_size must be at least one")
    model.eval()
    weighted_totals: dict[str, float] = {}
    evaluated_positions = 0
    for start in range(0, len(evaluation_data_buffer), evaluation_batch_size):
        stop = min(start + evaluation_batch_size, len(evaluation_data_buffer))
        batch = evaluation_data_buffer.batch_range(start, stop)
        spatial_inputs_tensor = torch.tensor(
            batch.spatial_inputs, dtype=torch.float32, device=model.device
        )
        non_spatial_inputs_tensor = torch.tensor(
            batch.non_spatial_inputs, dtype=torch.float32, device=model.device
        )
        masks_tensor = torch.tensor(
            batch.action_masks, dtype=torch.float32, device=model.device
        )
        tensor_targets = train_utils.numpy_targets_to_tensors(
            batch.targets,
            device=model.device,
        )
        model_output = model(
            spatial_inputs_tensor,
            non_spatial_inputs_tensor,
            masks_tensor,
        )
        loss, details = loss_function(model_output, tensor_targets)
        batch_size = stop - start
        for name, value in {"total_loss": loss.item(), **details}.items():
            weighted_totals[name] = (
                weighted_totals.get(name, 0.0) + float(value) * batch_size
            )
        evaluated_positions += batch_size
    return {
        name: total / evaluated_positions for name, total in weighted_totals.items()
    }


def make_optimizer(
    model: skynet.SkyNet,
    learn_rate: float,
) -> torch.optim.Optimizer:
    """Creates the optimizer for a model's training run."""
    return torch.optim.Adam(model.parameters(), lr=learn_rate, weight_decay=1e-4)


@dataclasses.dataclass(slots=True)
class LearnConfig(config.Config):
    torch_device: torch.device
    learn_steps: int
    games_generated_per_iteration: int
    checkpoint_interval: int
    loss_stats_function: (
        typing.Callable[[list[train_utils.LossDetails]], object] | None
    ) = None
