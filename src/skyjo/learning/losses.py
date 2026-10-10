"""Core full-game value and policy losses."""

from __future__ import annotations

import typing

import torch

from skyjo.learning import batches, skynet

from . import objectives

type LossDetails = dict[str, float]


class LossFunction(typing.Protocol):
    def __call__(
        self,
        model_output: skynet.ModelOutput,
        targets: batches.TensorTargets,
    ) -> tuple[torch.Tensor, LossDetails]: ...


def cross_entropy_policy_loss(
    predicted: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    return torch.nn.CrossEntropyLoss(reduction="mean")(predicted, target)


def mse_value_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.MSELoss(reduction="mean")(predicted, target)


def policy_value_losses(
    model_output: skynet.ModelOutput,
    targets: batches.TensorTargets,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert model_output.policy_logits.shape == targets["policy"].shape, (
        f"expected policy_logits of shape {targets['policy'].shape}, got {model_output.policy_logits.shape}"
    )
    assert model_output.value.shape == targets["value"].shape, (
        f"expected value of shape {targets['value'].shape}, got {model_output.value.shape}"
    )
    policy_loss = cross_entropy_policy_loss(
        model_output.policy_logits,
        targets["policy"],
    )
    value_loss = mse_value_loss(
        model_output.value,
        targets["value"],
    )
    return value_loss, policy_loss


def base_loss(
    model_output: skynet.ModelOutput,
    targets: batches.TensorTargets,
    value_scale: float = 1.0,
    policy_scale: float = 1.0,
) -> tuple[torch.Tensor, LossDetails]:
    value_loss, policy_loss = policy_value_losses(model_output, targets)
    return (
        value_scale * value_loss + policy_scale * policy_loss,
        {
            "outcome_value_loss": value_loss.item(),
            "policy_loss": policy_loss.item(),
        },
    )


def raw_score_loss(prediction: torch.Tensor, target: torch.Tensor):
    return torch.nn.functional.mse_loss(prediction, target), {
        "mae_points": 144.0
        * torch.nn.functional.l1_loss(prediction, target).detach().item(),
    }


def charged_loss(prediction: torch.Tensor, target: torch.Tensor):
    return torch.nn.functional.mse_loss(prediction, target), {
        "mae_points": 336.0
        * torch.nn.functional.l1_loss(prediction, target).detach().item(),
    }


def doubled_loss(prediction: torch.Tensor, target: torch.Tensor):
    return torch.nn.functional.binary_cross_entropy_with_logits(prediction, target), {}


def auxiliary_loss(name: str, prediction: torch.Tensor, target: torch.Tensor):
    if name == "round_score":
        return charged_loss(prediction, target)
    if name == "round_raw_score":
        return raw_score_loss(prediction, target)
    if name == "round_doubled":
        return doubled_loss(prediction, target)
    raise ValueError(f"Unknown auxiliary objective: {name}")


def configured_loss(
    output: skynet.ModelOutput,
    targets: batches.TensorTargets,
    *,
    auxiliary_objectives: objectives.ObjectiveConfig = None,
    value_scale: float = 1.0,
    policy_scale: float = 1.0,
):
    total, details = base_loss(output, targets, value_scale, policy_scale)
    for name, weight in objectives.resolve(auxiliary_objectives).entries:
        if name not in targets:
            raise ValueError(f"Missing target for enabled auxiliary objective: {name}")
        if name not in output.auxiliary_outputs:
            raise ValueError(f"Missing model output for auxiliary objective: {name}")
        prediction = output.auxiliary_outputs[name]
        if prediction.shape != targets[name].shape:
            raise ValueError(f"Prediction/target shape mismatch for {name}")
        loss, metrics = auxiliary_loss(name, prediction, targets[name])
        total = total + weight * loss
        details[f"{name}_loss"] = loss.detach().item()
        details[f"{name}_weighted_loss"] = weight * loss.detach().item()
        details.update({f"{name}_{key}": value for key, value in metrics.items()})
    return total, details
