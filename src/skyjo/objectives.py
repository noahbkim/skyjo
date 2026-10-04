"""Named auxiliary objectives shared by models, target construction, and training."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import game as sj

if TYPE_CHECKING:
    from .targets import TerminalSummary


@dataclasses.dataclass(frozen=True)
class Objective:
    target_shape: Callable[[int], tuple[int, ...]]
    dependency: str
    target: Callable[[Any, sj.Skyjo], np.ndarray]
    head: Callable[[int, int], nn.Module]
    loss: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, dict[str, float]]]


def player_shape(players: int) -> tuple[int, ...]:
    return (players,)


def raw_score_target(summary: TerminalSummary, state: sj.Skyjo) -> np.ndarray:
    return np.roll(summary.raw_scores / 144.0, -sj.get_player(state)).astype(np.float32)


def doubled_target(summary: TerminalSummary, state: sj.Skyjo) -> np.ndarray:
    return np.roll(summary.doubled, -sj.get_player(state)).astype(np.float32)


def raw_score_loss(
    prediction: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, dict[str, float]]:
    return F.mse_loss(prediction, target), {
        "mae_points": 144.0 * F.l1_loss(prediction, target).detach().item(),
    }


def doubled_loss(
    prediction: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, dict[str, float]]:
    return F.binary_cross_entropy_with_logits(prediction, target), {}


def charged_target(summary, state):
    from .skynet import normalize_round_scores

    return normalize_round_scores(np.roll(summary.scores, -sj.get_player(state)))


def charged_head(width, players):
    from .skynet import NormalizedRoundScoreTail

    return NormalizedRoundScoreTail(width, players)


def charged_loss(prediction, target):
    return F.mse_loss(prediction, target), {
        "mae_points": 336.0 * F.l1_loss(prediction, target).detach().item(),
    }


REGISTRY: dict[str, Objective] = {
    "round_score": Objective(
        player_shape, "terminal", charged_target, charged_head, charged_loss
    ),
    "round_raw_score": Objective(
        player_shape, "terminal", raw_score_target, nn.Linear, raw_score_loss
    ),
    "round_doubled": Objective(
        player_shape, "terminal", doubled_target, nn.Linear, doubled_loss
    ),
}


@dataclasses.dataclass(frozen=True)
class ResolvedObjectives:
    entries: tuple[tuple[str, float, Objective], ...] = ()

    @property
    def weights(self) -> dict[str, float]:
        return {name: weight for name, weight, _ in self.entries}

    def shapes(self, players: int) -> dict[str, tuple[int, ...]]:
        return {
            name: objective.target_shape(players) for name, _, objective in self.entries
        }

    def make_heads(self, width: int, players: int) -> nn.ModuleDict:
        heads = nn.ModuleDict()
        # Each head starts from the post-core RNG state, independently of which
        # other heads are enabled. Neither CPU nor gameplay RNG advances.
        for name, _, objective in self.entries:
            with torch.random.fork_rng(devices=[]):
                heads[name] = objective.head(width, players)
        return heads

    def add_losses(
        self,
        total: torch.Tensor,
        details: dict[str, float],
        output: Any,
        targets: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        predictions = getattr(output, "auxiliary_outputs", {})
        for name, weight, objective in self.entries:
            if name not in targets:
                raise ValueError(
                    f"Missing target for enabled auxiliary objective: {name}"
                )
            if name not in predictions:
                raise ValueError(
                    f"Missing model output for auxiliary objective: {name}"
                )
            if predictions[name].shape != targets[name].shape:
                raise ValueError(f"Prediction/target shape mismatch for {name}")
            loss, metrics = objective.loss(predictions[name], targets[name])
            total = total + weight * loss
            details[f"{name}_loss"] = loss.detach().item()
            details[f"{name}_weighted_loss"] = weight * loss.detach().item()
            details.update({f"{name}_{key}": value for key, value in metrics.items()})
        return total, details


ObjectiveConfig = Mapping[str, float] | ResolvedObjectives | None


def resolve(config: ObjectiveConfig = None) -> ResolvedObjectives:
    if isinstance(config, ResolvedObjectives):
        return config
    entries = []
    for name, weight in sorted((config or {}).items()):
        if name not in REGISTRY:
            raise ValueError(f"Unknown auxiliary objective: {name}")
        if type(weight) not in (float, int) or not math.isfinite(weight) or weight < 0:
            raise ValueError(f"Auxiliary weight must be finite and nonnegative: {name}")
        if weight > 0:
            entries.append((name, float(weight), REGISTRY[name]))
    return ResolvedObjectives(tuple(entries))


def configured_loss(
    output, targets, *, auxiliary_objectives=None, value_scale=1.0, policy_scale=1.0
):
    from .losses import base_loss

    total, details = base_loss(
        output, targets, value_scale=value_scale, policy_scale=policy_scale
    )
    return resolve(auxiliary_objectives).add_losses(total, details, output, targets)
