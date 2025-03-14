import dataclasses
import logging
import typing

import numpy as np
import torch
import torch.nn as nn

import skyjo.skyjo_immutable as sj


## LOSS FUNCTIONS
def compute_policy_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    loss = -(target * torch.log(predicted)).sum(dim=1)
    return loss.mean()


def compute_value_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sum((target - predicted) ** 2) / target.size()[0]


# OUTPUT DATACLASSES
@dataclasses.dataclass(slots=True)
class PolicyOutput:
    probabilities: np.ndarray
    is_renormalized: bool = False

    def __post_init__(self):
        assert len(self.probabilities.shape) == 1 + len(sj.ACTION_SHAPE), (
            f"expected 3D probabilities array, got {self.probabilities.shape}"
        )
        assert self.probabilities.shape[1:] == sj.ACTION_SHAPE, (
            f"expected probabilities array of shape (n, {sj.ACTION_SHAPE}), got {self.probabilities.shape}"
        )
        assert abs(self.probabilities.sum() - 1) < 1e-6, (
            f"expected probabilities to sum to 1, got {self.probabilities.sum()}"
        )

    @classmethod
    def from_tensor_output(cls, output: torch.Tensor) -> typing.Self:
        return cls(output.detach().reshape(-1, *sj.ACTION_SHAPE).numpy())

    def mask_invalid_and_renormalize(
        self, valid_actions_mask: np.ndarray
    ) -> typing.Self:
        assert valid_actions_mask.shape == self.probabilities.shape, (
            f"expected valid_actions_mask of shape {self.probabilities.shape}, got {valid_actions_mask.shape}"
        )
        valid_action_probabilities = self.probabilities * valid_actions_mask
        total_valid_action_probabilities = np.sum(
            valid_action_probabilities,
            axis=tuple(range(1, len(self.probabilities.shape))),
        ).reshape(-1, 1, 1)
        if total_valid_action_probabilities > 0:
            renormalized_valid_action_probabilities = (
                valid_action_probabilities / total_valid_action_probabilities
            )
        # If no probabilities were valid, assign equal probability to all valid actions
        else:
            logging.warning("No valid actions, renormalizing to uniform")
            renormalized_valid_action_probabilities = valid_actions_mask / np.sum(
                valid_actions_mask, axis=tuple(range(1, len(self.probabilities.shape)))
            ).reshape(-1, 1, 1)
        return PolicyOutput(
            renormalized_valid_action_probabilities, is_renormalized=True
        )

    def get_action_probability(
        self, action: sj.SkyjoAction, sample_point_index: int = 0
    ) -> float:
        if action.action_type == sj.SkyjoActionType.DRAW:
            return self.probabilities[sample_point_index, 0].sum().item()
        return self.probabilities[
            sample_point_index,
            action.action_type.value,
            sj.Hand.flat_index(action.row_idx, action.col_idx),
        ].item()


@dataclasses.dataclass(slots=True)
class ValueOutput:
    values: np.ndarray

    def __post_init__(self):
        assert len(self.values.shape) == 2, (
            f"expected 2D (num_samples, num_players) values array, got {self.values.shape}"
        )

    @classmethod
    def from_tensor_output(cls, output: torch.Tensor) -> typing.Self:
        return cls(output.detach().numpy())


@dataclasses.dataclass(slots=True)
class SkyNetOutput:
    policy_output: PolicyOutput
    value_output: ValueOutput


# NETWORKS
class ResBlock(nn.Module):
    """Basic 1-D Residual Block with two 1-D convolutions, BatchNorm, and ReLU activation"""

    def __init__(
        self,
        in_channels: int,
        out_channels: tuple[int, int],
        kernel_sizes: tuple[int, int] = (3, 3),
        strides: tuple[int, int] = (1, 1),
        paddings: tuple[int, int] = (1, 1),
    ):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels[0],
            stride=strides[0],
            kernel_size=kernel_sizes[0],
            padding=paddings[0],
        )
        # output_length = ( input_length - kernel_size + 2 * padding) / stride + 1
        self.bn1 = nn.BatchNorm1d(num_features=out_channels[0])
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            in_channels=out_channels[0],
            out_channels=out_channels[1],
            stride=strides[1],
            kernel_size=kernel_sizes[1],
            padding=paddings[1],
        )
        self.bn2 = nn.BatchNorm1d(num_features=out_channels[1])
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu2(out)
        return out


class PolicyHead(nn.Module):
    def __init__(self, in_channels: int, input_length: int):
        super(PolicyHead, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=in_channels, out_channels=32, stride=1, kernel_size=1, padding=0
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.flatten1 = nn.Flatten()
        linear_out_length = np.cumprod([dim for dim in sj.ACTION_SHAPE])[-1]
        self.linear1 = nn.Linear(
            in_features=input_length * 32,  # input_length * out_channels
            out_features=linear_out_length,  # number of possible moves
        )
        self.softmax1 = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.flatten1(x)
        x = self.linear1(x)
        x = self.softmax1(x)
        return x.reshape(-1, *sj.ACTION_SHAPE)


class ValueHead(nn.Module):
    def __init__(self, in_channels: int, input_length: int, num_players: int):
        super(ValueHead, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=in_channels, out_channels=32, stride=1, kernel_size=1, padding=0
        )
        self.bn1 = nn.BatchNorm1d(num_features=32)
        self.relu1 = nn.ReLU(inplace=True)
        self.flatten1 = nn.Flatten()
        self.linear1 = nn.Linear(
            in_features=input_length * 32, out_features=num_players
        )
        self.softmax1 = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.flatten1(x)
        x = self.linear1(x)
        x = self.softmax1(x)
        return x


# TODO: Parameterize this more fully
class SkyNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_players: int = 2,
        device: torch.device = torch.device("cpu"),
    ):
        super(SkyNet, self).__init__()

        self.device = device
        # This will output sequences of length ( input_length - kernel_size + 2*padding) / stride + 1
        self.conv1 = nn.Conv1d(
            in_channels=in_channels, out_channels=32, stride=1, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm1d(num_features=32)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            in_channels=32, out_channels=64, stride=1, kernel_size=3, padding=1
        )  # output_length: 17
        self.bn2 = nn.BatchNorm1d(num_features=64)
        self.relu2 = nn.ReLU(inplace=True)
        self.resid1 = ResBlock(
            in_channels=64, out_channels=(64, 64)
        )  # output_length: 17
        self.policy_head = PolicyHead(in_channels=64, input_length=17)
        self.value_head = ValueHead(
            in_channels=64, input_length=17, num_players=num_players
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.resid1(out)
        policy_probs = self.policy_head(out)
        values = self.value_head(out)

        return policy_probs, values

    def predict(self, state_repr: np.ndarray) -> SkyNetOutput:
        """Convenience function that wraps forward() and returns a SkyNetOutput dataclass and takes an np.ndarray as input"""
        policy_tensor, value_tensor = self.forward(
            torch.tensor(state_repr, dtype=torch.float32).unsqueeze(0)
        )
        return SkyNetOutput(
            policy_output=PolicyOutput.from_tensor_output(policy_tensor),
            value_output=ValueOutput.from_tensor_output(value_tensor),
        )
