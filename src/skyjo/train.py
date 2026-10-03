"""
Module to Train Skyjo models
"""

import dataclasses
import math
import time

import numpy as np
import torch

from . import batches, buffer, checkpoint, config, gradient_diagnostic, losses, skynet
from . import game as sj


@dataclasses.dataclass
class TrainingDiagnostics:
    """Position-weighted policy statistics, independent of optimization and I/O."""

    totals: dict[str, torch.Tensor] = dataclasses.field(default_factory=dict)

    @torch.no_grad()
    def update(
        self, logits: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor
    ) -> None:
        logits, targets = logits.detach(), targets.detach()
        log_probs = logits.masked_fill(~masks.bool(), -torch.inf).log_softmax(dim=-1)
        probabilities = log_probs.exp()
        # Zero-probability entries (including masked actions) contribute zero.
        safe_logs = log_probs.masked_fill(~masks.bool(), 0)
        target_entropy = -torch.special.xlogy(targets, targets).sum(dim=-1)
        predicted_entropy = -(probabilities * safe_logs).sum(dim=-1)
        cross_entropy = -(targets * safe_logs).sum(dim=-1)
        values = torch.stack(
            (target_entropy, predicted_entropy, cross_entropy - target_entropy), dim=-1
        )
        groups = {
            "all": torch.ones(len(logits), dtype=torch.bool, device=logits.device),
            "initial_reveal": masks[:, : sj.MASK_DRAW].bool().any(dim=-1),
            "draw_take": masks[:, sj.MASK_DRAW].bool(),
            "flip_replace": masks[:, sj.MASK_FLIP : sj.MASK_REPLACE].bool().any(dim=-1),
        }
        groups["replace_only"] = ~(
            groups["initial_reveal"] | groups["draw_take"] | groups["flip_replace"]
        )
        for name, selected in groups.items():
            total = torch.cat((selected.sum().reshape(1), values[selected].sum(dim=0)))
            if name not in self.totals:
                self.totals[name] = total
            else:
                self.totals[name] += total

    def summary(self) -> dict[str, float | int]:
        metrics = {}
        names = ("target_entropy", "predicted_entropy", "target_kl")
        for group, total in self.totals.items():
            count, *sums = total.cpu().tolist()
            metrics[f"policy/{group}/positions"] = int(count)
            if count:
                metrics.update(
                    {
                        f"policy/{group}/{key}": value / count
                        for key, value in zip(names, sums)
                    }
                )
        return metrics


# MARK: Training


@dataclasses.dataclass(slots=True)
class ReplayRatioTrainConfig(config.Config):
    """Training budget expressed as sampled positions per new position."""

    batch_size: int
    replay_ratio: float
    loss_function: losses.LossFunction
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
    diagnostics = TrainingDiagnostics()
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
    batch: batches.TrainingBatch,
    loss_function: losses.LossFunction,
    optimizer: torch.optim.Optimizer,
    diagnostics: TrainingDiagnostics | None = None,
    gradient_scales: dict | None = None,
) -> tuple[float, losses.LossDetails]:
    """Performs a single training step on the model."""
    model.train()
    tensors = batches.to_tensors(batch, device=model.device)
    model_output = model(
        tensors.spatial_inputs, tensors.non_spatial_inputs, tensors.action_masks
    )
    tensor_targets = tensors.targets
    loss, loss_detail = loss_function(model_output, tensor_targets)
    if diagnostics is not None:
        diagnostics.update(
            model_output.policy_logits, tensor_targets["policy"], tensors.action_masks
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
    loss_function: losses.LossFunction,
    diagnostics: TrainingDiagnostics | None = None,
    gradient_scales: dict | None = None,
    sampling_rng: np.random.Generator | None = None,
) -> list[losses.LossDetails]:
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
    loss_function: losses.LossFunction,
    *,
    indices: np.ndarray | None = None,
    diagnostics: bool = False,
) -> dict[str, float | int | None]:
    """Evaluate position-weighted losses without changing model mode or RNG streams."""
    if indices is None:
        indices = np.arange(len(evaluation_data_buffer))
    if not len(indices) or evaluation_batch_size < 1:
        raise ValueError("Evaluation requires positions and a positive batch size")
    rng = checkpoint.capture_rng_state()
    modes = [(module, module.training) for module in model.modules()]
    totals = {}
    known_count, known_error = 0, 0.0
    has_raw = False
    weights = getattr(loss_function, "keywords", {})
    try:
        model.eval()
        for start in range(0, len(indices), evaluation_batch_size):
            batch = batches.to_tensors(
                evaluation_data_buffer.batch_indices(
                    indices[start : start + evaluation_batch_size]
                ),
                device=model.device,
            )
            output = model(
                batch.spatial_inputs, batch.non_spatial_inputs, batch.action_masks
            )
            loss, details = loss_function(output, batch.targets)
            details = {"total_loss": loss.item(), **details}
            if diagnostics:
                details["outcome_value_weighted_loss"] = (
                    weights.get("value_scale", 1.0) * details["outcome_value_loss"]
                )
                details["policy_weighted_loss"] = (
                    weights.get("policy_scale", 1.0) * details["policy_loss"]
                )
                policy = batch.targets["policy"]
                entropy = (
                    -(policy * policy.clamp_min(1e-30).log()).sum(-1).mean().item()
                )
                details["policy_target_entropy"] = entropy
                details["policy_kl"] = details["policy_loss"] - entropy
                auxiliary = getattr(output, "auxiliary_outputs", {})
                if "round_raw_score" in auxiliary:
                    has_raw = True
                    known = batch.spatial_inputs[..., sj.FINGER_HIDDEN].sum((2, 3)) == 0
                    known_count += int(known.sum().item())
                    known_error += (
                        auxiliary["round_raw_score"] - batch.targets["round_raw_score"]
                    )[known].abs().sum().item() * 144
            for name, value in details.items():
                totals[name] = totals.get(name, 0.0) + float(value) * len(
                    batch.spatial_inputs
                )
    finally:
        for module, training in modes:
            module.training = training
        checkpoint.restore_rng_state(rng)
    result = {name: total / len(indices) for name, total in totals.items()}
    if diagnostics:
        result["evaluated_positions"] = len(indices)
        if has_raw:
            result["round_raw_score_known_count"] = known_count
            result["round_raw_score_known_mae_points"] = (
                known_error / known_count if known_count else None
            )
    return result


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
