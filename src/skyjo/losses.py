"""Core full-game value and policy losses."""

import typing

import torch

from . import batches, skynet

LossDetails: typing.TypeAlias = dict[
    str, float
]  # loss component name: loss component value


class LossFunction(typing.Protocol):
    def __call__(
        self,
        model_output: skynet.SupportsCoreSkyNetOutput,
        targets: batches.TensorTargets,
    ) -> tuple[torch.Tensor, LossDetails]: ...


def cross_entropy_policy_loss(
    predicted: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    return torch.nn.CrossEntropyLoss(reduction="mean")(predicted, target)


def mse_value_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.MSELoss(reduction="mean")(predicted, target)


def policy_value_losses(
    model_output: skynet.SupportsCoreSkyNetOutput,
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
    model_output: skynet.SupportsCoreSkyNetOutput,
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
