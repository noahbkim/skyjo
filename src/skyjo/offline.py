"""Offline execution shared by the single-run CLI and fixed-replay comparisons."""

import functools
import random
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch

from . import checkpoint, game, models, objectives, train, train_utils


@contextmanager
def preserve_rng():
    state = checkpoint.capture_rng_state()
    try:
        yield
    finally:
        checkpoint.restore_rng_state(state)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class OfflineTrainer:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    loss_function: object
    sampling_rng: np.random.Generator | None = None

    @classmethod
    def from_configuration(cls, configuration, seed):
        set_seed(seed)
        model = models.build(
            configuration["model"],
            players=configuration["players"],
            device=configuration["execution"]["device"],
            auxiliary_objectives=configuration["auxiliary_objectives"],
        )
        settings = configuration["training"]
        loss = functools.partial(
            objectives.configured_loss,
            auxiliary_objectives=configuration["auxiliary_objectives"],
            value_scale=settings["value_scale"],
            policy_scale=settings["policy_scale"],
        )
        # Model construction and stochastic training never advance this stream.
        sampling_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x53414D50]))
        set_seed(int(np.random.SeedSequence([seed, 0x54524149]).generate_state(1)[0]))
        return cls(
            model,
            train.make_optimizer(model, settings["learn_rate"]),
            loss,
            sampling_rng,
        )

    def fit(self, replay, *, steps, batch_size, gradient_scales=None):
        return train.train_steps(
            self.model,
            replay,
            training_batch_size=batch_size,
            optimizer_steps=steps,
            optimizer=self.optimizer,
            loss_function=self.loss_function,
            gradient_scales=gradient_scales,
            sampling_rng=self.sampling_rng,
        )

    def evaluate(self, replay, *, batch_size, indices=None, diagnostics=False):
        training = self.model.training
        try:
            with preserve_rng():
                if not diagnostics and indices is None:
                    return train.evaluate_loss(
                        self.model, replay, batch_size, self.loss_function
                    )
                return evaluate_metrics(
                    self.model, replay, self.loss_function, batch_size, indices
                )
        finally:
            self.model.train(training)


@torch.inference_mode()
def evaluate_metrics(model, replay, loss_function, batch_size, indices=None):
    """Position-weighted losses; known-score errors use player-example counts."""
    model.eval()
    if indices is None:
        indices = np.arange(len(replay))
    if not len(indices) or batch_size < 1:
        raise ValueError("Evaluation requires positions and a positive batch size")
    totals = {}
    known_count, known_error = 0, 0.0
    has_raw = False
    weights = getattr(loss_function, "keywords", {})
    for start in range(0, len(indices), batch_size):
        batch = replay.batch_indices(indices[start : start + batch_size])
        spatial = torch.tensor(
            batch.spatial_inputs, dtype=torch.float32, device=model.device
        )
        output = model(
            spatial,
            torch.tensor(
                batch.non_spatial_inputs, dtype=torch.float32, device=model.device
            ),
            torch.tensor(batch.action_masks, dtype=torch.float32, device=model.device),
        )
        targets = train_utils.numpy_targets_to_tensors(
            batch.targets, device=model.device
        )
        loss, details = loss_function(output, targets)
        details = {"total_loss": loss.item(), **details}
        details["outcome_value_weighted_loss"] = (
            weights.get("value_scale", 1.0) * details["outcome_value_loss"]
        )
        details["policy_weighted_loss"] = (
            weights.get("policy_scale", 1.0) * details["policy_loss"]
        )
        # KL removes the target-entropy constant from the policy cross entropy.
        entropy = (
            -(targets.policy * targets.policy.clamp_min(1e-30).log())
            .sum(-1)
            .mean()
            .item()
        )
        details["policy_target_entropy"] = entropy
        details["policy_kl"] = details["policy_loss"] - entropy
        for key, value in details.items():
            totals[key] = totals.get(key, 0.0) + float(value) * len(
                batch.spatial_inputs
            )
        auxiliary = getattr(output, "auxiliary_outputs", {})
        if "round_raw_score" in auxiliary:
            has_raw = True
            known = spatial[..., game.FINGER_HIDDEN].sum((2, 3)) == 0
            known_count += int(known.sum().item())
            known_error += (
                float(
                    (auxiliary["round_raw_score"] - targets["round_raw_score"])[known]
                    .abs()
                    .sum()
                    .item()
                )
                * 144
            )
    result = {key: value / len(indices) for key, value in totals.items()}
    result["evaluated_positions"] = len(indices)
    if has_raw:
        result["round_raw_score_known_count"] = known_count
        result["round_raw_score_known_mae_points"] = (
            known_error / known_count if known_count else None
        )
    return result
