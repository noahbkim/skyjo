from __future__ import annotations

import dataclasses
import datetime
import pathlib
import typing

import einops
import numpy as np
import torch
import torch.nn as nn

import skyjo as sj

"""
einops and general dimension notation:

b,N = batch size
p,P = number of players
c,C = number of channels
w,W = width
h,H = height
n_a,A = action space
t,T = action_type
f,F = feature space
"""

# TYPE ALIASES
StateValue: typing.TypeAlias = np.ndarray[tuple[int], np.float32]
"""A vector representing the value of a Skyjo game for each player.

IMPORTANT: This is from a fixed perspective and not relative to the current player
    i.e. the first element is always the value of player 0, the second is for player 1, etc.

This can also be used to represent the outcome of the game where all entries are 0 except for the winner.
"""


# State value convenience functions
def skyjo_to_state_value(skyjo: sj.Skyjo) -> StateValue:
    """Get the outcome of the game from the fixed perspective."""
    players = skyjo[3]
    outcome = np.zeros((players,), dtype=np.float32)
    outcome[sj.get_fixed_perspective_winner(skyjo)] = 1.0
    return outcome


def state_value_for_player(state_value: StateValue, player: int) -> float:
    """Get the value of the game for a given player."""
    state_value = state_value.squeeze()
    assert len(state_value.shape) == 1, "Expected a 1D state value"
    return state_value[player].item()


def to_state_value(
    value_output: np.ndarray[tuple[int], np.float32], curr_player: int
) -> StateValue:
    return np.roll(value_output, shift=-curr_player)


## Training Data
TrainingDataPoint: typing.TypeAlias = tuple[
    sj.Skyjo,  # game state
    np.ndarray[tuple[int], np.float32],  # policy target
    np.ndarray[tuple[int], np.float32],  # value target
    np.ndarray[tuple[int], np.float32],  # points target
]
TrainingBatch: typing.TypeAlias = list[TrainingDataPoint]


def get_policy_target(
    batch: TrainingBatch,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    return torch.tensor(
        np.array([data_point[0] for data_point in batch]), device=device
    )


def get_value_target(
    batch: TrainingBatch,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    return torch.tensor(
        np.array([data_point[1] for data_point in batch]), device=device
    )


def get_points_target(
    batch: TrainingBatch,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    return torch.tensor(
        np.array([data_point[2] for data_point in batch]), device=device
    )


## LOSS FUNCTIONS
def compute_policy_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    loss = -(target * torch.log(predicted + 1e-9)).sum(
        dim=1
    )  # add small epsilon to avoid log(0)
    return loss.mean()


def compute_value_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sum((target - predicted) ** 2) / target.size()[0]


def base_total_loss(
    model_output: SkyNetOutput, batch: TrainingBatch, value_scale: float = 3.0
) -> torch.Tensor:
    value_output, points_output, policy_output = model_output
    _, policy_targets, value_targets, points_targets = zip(*batch)
    policy_loss = compute_policy_loss(
        policy_output,
        torch.tensor(
            np.array(policy_targets), device=policy_output.device, dtype=torch.float32
        ),
    )
    value_loss = compute_value_loss(
        value_output,
        torch.tensor(
            np.array(value_targets), device=value_output.device, dtype=torch.float32
        ),
    )
    return policy_loss + value_scale * value_loss


def mask_and_renormalize_policy_probabilities(
    policy_probabilities: np.ndarray[tuple[int], np.float32],
    valid_actions_mask: np.ndarray[tuple[int], np.int8],
) -> np.ndarray[tuple[int], np.float32]:
    assert valid_actions_mask.shape == policy_probabilities.shape, (
        f"expected valid_actions_mask of shape {policy_probabilities.shape}, got {valid_actions_mask.shape}"
    )
    valid_action_probabilities = policy_probabilities * valid_actions_mask
    total_valid_action_probabilities = valid_action_probabilities.sum()
    num_valid_actions = valid_actions_mask.sum()
    assert not np.any(num_valid_actions == 0), (
        "expected no samples with no valid actions"
    )
    if total_valid_action_probabilities == 0:
        return np.ones_like(policy_probabilities) / num_valid_actions
    renormalized_valid_action_probabilities = (
        valid_action_probabilities / total_valid_action_probabilities
    )
    return renormalized_valid_action_probabilities


def get_single_model_output(
    model_output: SkyNetOutput | SkyNetNumpyOutput, idx: int
) -> SkyNetOutput | SkyNetNumpyOutput:
    return (
        model_output[0][idx],
        model_output[1][idx],
        model_output[2][idx],
    )


def output_to_numpy(output: SkyNetOutput) -> SkyNetNumpyOutput:
    """Converts a SkyNetOutput to a SkyNetNumpyOutput.

    Handles the detaching and converting to numpy
    (including copying to cpu when device is not cpu).
    """
    value_output = output[0].detach()
    points_output = output[1].detach()
    policy_output = output[2].detach()
    if value_output.device != torch.device("cpu"):
        value_output = value_output.cpu()
    if points_output.device != torch.device("cpu"):
        points_output = points_output.cpu()
    if policy_output.device != torch.device("cpu"):
        policy_output = policy_output.cpu()
    return value_output.numpy(), points_output.numpy(), policy_output.numpy()


# Prediction dataclasses
@dataclasses.dataclass(slots=True)
class SkyNetPrediction:
    value_output: np.ndarray[tuple[int], np.float32]
    points_output: np.ndarray[tuple[int], np.float32]
    policy_output: np.ndarray[tuple[int], np.float32]

    @classmethod
    def from_skynet_output(
        cls,
        output: SkyNetOutput,
    ) -> SkyNetPrediction:
        numpy_output = output_to_numpy(output)
        value_numpy, points_numpy, policy_numpy = get_single_model_output(
            numpy_output, 0
        )
        assert (
            len(value_numpy.shape)
            == len(points_numpy.shape)
            == len(policy_numpy.shape)
            == 1
        ), (
            "expected value_output, points_output, and policy_output to be a single result and not batched results."
            f"value_output.shape: {value_numpy.shape}, points_output.shape: {points_numpy.shape}, policy_output.shape: {policy_numpy.shape}"
        )
        return SkyNetPrediction(
            value_output=value_numpy,
            points_output=points_numpy,
            policy_output=policy_numpy,
        )

    def __str__(self) -> str:
        return f"{self.policy_output}\n{self.value_output}\n{self.points_output}"

    def mask_and_renormalize(self, valid_actions_mask: np.ndarray[tuple[int], np.int8]):
        self.policy_output = mask_and_renormalize_policy_probabilities(
            self.policy_output, valid_actions_mask
        )


# NETWORKS
class ResBlock(nn.Module):
    """Residual Block"""

    def __init__(
        self,
        in_channels: int,
        out_channels: list[int],
        kernel_sizes: list[tuple[int, int]] | tuple[int, int] = [(1, 1), (1, 1)],
        strides: list[tuple[int, int]] | tuple[int, int] = [(1, 1), (1, 1)],
        paddings: list[tuple[int, int]] | tuple[int, int] = [(0, 0), (0, 0)],
        activations: list[nn.Module] = [nn.ReLU(inplace=True), nn.ReLU(inplace=True)],
        batch_norm_kwargs: list[dict[str, typing.Any]] = [{}, {}],
    ):
        """
        Initializes a Residual Block.
        """
        super(ResBlock, self).__init__()
        # Input: (N, in_channels, H_in, W_in)
        # Output: (N, out_channels[0], (H_in + 2*padding - kernel) / padding + 1, (H_in + 2*padding - kernel) / padding + 1)
        #   for kernel = (1, 1), stride = (1, 1), and padding = (0, 0) -> (N, out_channels[0], H_in, W_in)
        #   for kernel = (3, 1), stride = (1, 1), and padding = (1, 0) -> (N, out_channels[0], H_in, W_in)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels[0],
            stride=strides[0],
            kernel_size=kernel_sizes[0],
            padding=paddings[0],
        )
        # Input: (N, out_channels[0], H_in, W_in)
        # Output: (N, out_channels[0], H_in, W_in)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels[0], **batch_norm_kwargs[0])
        self.activation1 = activations[0]
        self.conv2 = nn.Conv2d(
            in_channels=out_channels[1],
            out_channels=out_channels[1],
            stride=strides[1],
            kernel_size=kernel_sizes[1],
            padding=paddings[1],
        )
        self.bn2 = nn.BatchNorm2d(num_features=out_channels[1], **batch_norm_kwargs[1])
        self.activation2 = activations[1]

    def forward(self, x):
        assert len(x.shape) == 4, f"expected 4D input (N, C, H, W), got {x.shape}"
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.activation2(out)
        return out


# MODEL HEADS (INPUT)
class Spatia1DInputHead(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, ...],
        hand_embedding_features: int,
        out_features: int,
        dropout_rate: float = 0.5,
    ):
        super(Spatia1DInputHead, self).__init__()
        assert len(input_shape) == 4, (
            f"expected 4D input (P, H, W, C), got {input_shape}"
        )

        self.input_shape = input_shape
        self.num_players = input_shape[0]
        self.num_rows = input_shape[1]
        self.num_columns = input_shape[2]
        self.num_channels = input_shape[3]

        self.hand_embedding_features = hand_embedding_features
        self.hand_encoder = nn.Sequential(
            nn.Linear(
                in_features=self.num_channels * self.num_rows * self.num_columns,
                out_features=(
                    self.num_channels * self.num_rows * self.num_columns
                    + hand_embedding_features
                ),
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                in_features=(
                    self.num_channels * self.num_rows * self.num_columns
                    + hand_embedding_features
                ),
                out_features=hand_embedding_features,
            ),
            nn.ReLU(inplace=True),
        )

        self.out_features = out_features
        self.transform_encoded_hands = nn.Sequential(
            nn.Linear(
                in_features=self.num_players * hand_embedding_features,
                out_features=self.out_features,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.out_features, out_features=self.out_features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        assert x.shape[1:] == (
            self.num_players,
            self.num_rows,
            self.num_columns,
            self.num_channels,
        ), f"expected input shape (P, H, W, C), got {x.shape}"
        x = einops.rearrange(x, "b p h w c -> (b p) (h w c)")
        encoded_hands = self.hand_encoder(x)
        x = einops.rearrange(encoded_hands, "(b p) f -> b (p f)", p=self.num_players)
        x = self.transform_encoded_hands(x)
        return x


class Spatial2DInputHead(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, ...],
        hand_embedding_channels: int,
        out_features: int,
        dropout_rate: float = 0.5,
    ):
        super(Spatial2DInputHead, self).__init__()
        assert len(input_shape) == 4, (
            f"expected 4D input (P, H, W, C), got {input_shape}"
        )
        self.input_shape = input_shape
        self.players = input_shape[0]
        self.out_features = out_features
        self.hand_embedding_channels = hand_embedding_channels
        self.conv1 = nn.Conv2d(
            in_channels=self.input_shape[3],
            out_channels=self.hand_embedding_channels,
            kernel_size=(3, 1),
            stride=(1, 1),
            padding=(1, 0),
        )
        self.resblock1 = ResBlock(
            in_channels=self.hand_embedding_channels,
            out_channels=[self.hand_embedding_channels, self.hand_embedding_channels],
            kernel_sizes=[(3, 3), (3, 3)],
            strides=[(1, 1), (1, 1)],
            paddings=[(1, 1), (1, 1)],
        )
        self.max_pool = nn.MaxPool2d(kernel_size=(sj.ROW_COUNT, sj.COLUMN_COUNT))
        self.combiner = nn.Sequential(
            nn.Linear(
                in_features=self.hand_embedding_channels * self.players,
                out_features=self.out_features,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        assert x.shape[1:] == self.input_shape, (
            f"expected input shape {self.input_shape}, got {x.shape}"
        )
        x = einops.rearrange(x, "b p h w c -> (b p) c h w")
        x = self.conv1(x)
        x = self.resblock1(x)
        x = self.max_pool(x)  # (b, c, 1, 1)
        x = einops.rearrange(x, "(b p) c h w -> b (p c h w)", p=self.players)
        x = self.combiner(x)  # (b, out_features)
        return x


class NonSpatialInputHead(nn.Module):
    """Handles the non-spatial features of the current skyjo state (i.e. the top card, discard pile, etc.)"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activations: list[nn.Module] = [nn.ReLU(inplace=True), nn.ReLU(inplace=True)],
        dropout_rate: float = 0.5,
    ):
        assert len(activations) == 2, "expected 2 activations, got {}".format(
            len(activations)
        )
        super(NonSpatialInputHead, self).__init__()
        self.out_features = out_features
        self.in_features = in_features
        self.linear1 = nn.Linear(in_features=in_features, out_features=out_features)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.activation1 = activations[0]
        self.linear2 = nn.Linear(in_features=out_features, out_features=out_features)
        self.activation2 = activations[1]

    def forward(self, x):
        """
        Input: (N, F)
        Output: (N, out_features)
        """
        assert len(x.shape) == 2, f"expected 2D input (N, F), got {x.shape}"
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        x = self.dropout1(x)
        return x


# MODEL TAILS (OUTPUT HEADS)
class PolicyTail(nn.Module):
    def __init__(self, num_features: int, out_features: tuple[int, int, int]):
        super(PolicyTail, self).__init__()
        self.num_features = num_features
        self.out_features = out_features
        self.linear1 = nn.Linear(
            in_features=num_features,
            out_features=out_features,  # number of possible moves
        )
        self.activation1 = nn.ReLU(inplace=True)
        self.softmax1 = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Input: (N, num_features)
        Output: (N, output_shape_size)
        """
        assert len(x.shape) == 2, f"expected 2D input (N, F), got {x.shape}"
        x = self.linear1(x)
        x = self.softmax1(x)
        return x


class ValueTail(nn.Module):
    def __init__(self, num_features: int, num_players: int):
        super(ValueTail, self).__init__()
        # INPUT: (N, F)
        # OUTPUT: (N, P)
        self.linear1 = nn.Linear(in_features=num_features, out_features=num_players)
        self.activation1 = nn.Sigmoid()

    def forward(self, x):
        """
        Input: (N, num_features)
        Output: (N, P)
        """
        x = self.linear1(x)
        x = self.activation1(x)
        return x


class PointsTail(nn.Module):
    def __init__(self, num_features: int, num_players: int):
        super(PointsTail, self).__init__()
        self.linear1 = nn.Linear(in_features=num_features, out_features=num_players)

    def forward(self, x):
        """
        Input: (N, num_features)
        Output: (N, P)
        """
        return self.linear1(x)


class SkyNet1D(nn.Module):
    """SkyNet that flattens spatial features and runs simpleler tranformations rather than ResBlock and Convoluational Layers"""

    def __init__(
        self,
        spatial_input_shape: tuple[int, ...],  # (players, )
        non_spatial_input_shape: tuple[int],
        value_output_shape: tuple[int],  # (players,)
        policy_output_shape: tuple[int],  # (mask_size,)
        device: torch.device = torch.device("cpu"),
        dropout_rate: float = 0.5,
    ):
        super(SkyNet1D, self).__init__()
        self.spatial_input_shape = spatial_input_shape
        self.non_spatial_input_shape = non_spatial_input_shape
        self.value_output_shape = value_output_shape
        self.policy_output_shape = policy_output_shape
        self.dropout_rate = dropout_rate
        self.device = device
        self.to(device)
        self.final_embedding_dim = 128
        self.spatial_input_head = Spatia1DInputHead(
            input_shape=spatial_input_shape,
            hand_embedding_features=64,
            out_features=64,
        )
        self.non_spatial_input_head = NonSpatialInputHead(
            in_features=non_spatial_input_shape[0],
            out_features=48,
        )
        self.mlp = nn.Sequential(
            nn.Linear(
                in_features=self.spatial_input_head.out_features
                + self.non_spatial_input_head.out_features,
                out_features=(
                    self.spatial_input_head.out_features
                    + self.non_spatial_input_head.out_features
                    + self.final_embedding_dim
                )
                // 2,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                in_features=(
                    self.spatial_input_head.out_features
                    + self.non_spatial_input_head.out_features
                    + self.final_embedding_dim
                )
                // 2,
                out_features=self.final_embedding_dim,
            ),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(self.dropout_rate)
        self.value_tail = ValueTail(
            num_features=self.final_embedding_dim,
            num_players=self.value_output_shape[0],
        )
        self.policy_tail = PolicyTail(
            num_features=self.final_embedding_dim,
            out_features=self.policy_output_shape[0],
        )
        self.points_tail = PointsTail(
            num_features=self.final_embedding_dim,
            num_players=self.value_output_shape[0],
        )

    def set_device(self, device: torch.device):
        self.device = device
        self.to(device)

    def forward(
        self,
        spatial_tensor: torch.Tensor,
        non_spatial_tensor: torch.Tensor,
    ) -> SkyNetOutput:
        spatial_out = self.spatial_input_head(spatial_tensor)
        non_spatial_out = self.non_spatial_input_head(non_spatial_tensor)
        combined_out = torch.cat((spatial_out, non_spatial_out), dim=1)
        mlp_out = self.mlp(combined_out)
        dropped_out = self.dropout(mlp_out)
        value_out = self.value_tail(dropped_out)
        policy_out = self.policy_tail(dropped_out)
        points_out = self.points_tail(dropped_out)
        return value_out, points_out, policy_out

    def predict(self, skyjo: sj.Skyjo) -> SkyNetPrediction:
        """Takes a single skyjo representation and returns a prediction object.

        Prediction object returned contains numpy arrays instead of tensors.
        This function is mostly a convenience function for testing and
        debugging. For actual optimized usage the `forward` method offers less
        overhead and flexibility.
        """
        spatial_tensor = einops.rearrange(
            torch.tensor(
                sj.get_spatial_input(skyjo), dtype=torch.float32, device=self.device
            ),
            "p h w c -> 1 p h w c",
        )
        non_spatial_tensor = einops.rearrange(
            torch.tensor(
                sj.get_non_spatial_input(skyjo), dtype=torch.float32, device=self.device
            ),
            "f -> 1 f",
        )
        output = self.forward(spatial_tensor, non_spatial_tensor)
        return SkyNetPrediction.from_skynet_output(output)

    def save(self, dir: pathlib.Path) -> pathlib.Path:
        curr_utc_dt = datetime.datetime.now(tz=datetime.timezone.utc)
        model_path = dir / f"model_{curr_utc_dt.strftime('%Y%m%d_%H%M%S')}.pth"
        torch.save(
            self.state_dict(),
            model_path,
        )
        return model_path


class SkyNet2D(nn.Module):
    def __init__(
        self,
        spatial_input_shape: tuple[int, ...],  # (players, )
        non_spatial_input_shape: tuple[int],
        value_output_shape: tuple[int],  # (players,)
        policy_output_shape: tuple[int],  # (mask_size,)
        device: torch.device = torch.device("cpu"),
        dropout_rate: float = 0.5,
    ):
        super(SkyNet2D, self).__init__()
        self.spatial_input_shape = spatial_input_shape
        self.non_spatial_input_shape = non_spatial_input_shape
        self.value_output_shape = value_output_shape
        self.policy_output_shape = policy_output_shape
        self.dropout_rate = dropout_rate
        self.device = device
        self.final_embedding_dim = 32
        self.spatial_input_head = Spatial2DInputHead(
            input_shape=spatial_input_shape,
            hand_embedding_channels=32,
            out_features=64,
            dropout_rate=dropout_rate,
        )
        self.non_spatial_input_head = NonSpatialInputHead(
            in_features=non_spatial_input_shape[0],
            out_features=32,
            dropout_rate=dropout_rate,
        )
        self.mlp = nn.Sequential(
            nn.Linear(
                in_features=self.spatial_input_head.out_features
                + self.non_spatial_input_head.out_features,
                out_features=(
                    self.spatial_input_head.out_features
                    + self.non_spatial_input_head.out_features
                    + self.final_embedding_dim
                )
                // 2,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                in_features=(
                    self.spatial_input_head.out_features
                    + self.non_spatial_input_head.out_features
                    + self.final_embedding_dim
                )
                // 2,
                out_features=self.final_embedding_dim,
            ),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(self.dropout_rate)
        self.value_tail = ValueTail(
            num_features=self.final_embedding_dim,
            num_players=self.value_output_shape[0],
        )
        self.policy_tail = PolicyTail(
            num_features=self.final_embedding_dim,
            out_features=self.policy_output_shape[0],
        )
        self.points_tail = PointsTail(
            num_features=self.final_embedding_dim,
            num_players=self.value_output_shape[0],
        )

    def set_device(self, device: torch.device):
        self.device = device
        self.to(device)

    def forward(
        self,
        spatial_tensor: torch.Tensor,
        non_spatial_tensor: torch.Tensor,
    ) -> SkyNetOutput:
        spatial_out = self.spatial_input_head(spatial_tensor)
        non_spatial_out = self.non_spatial_input_head(non_spatial_tensor)
        combined_out = torch.cat((spatial_out, non_spatial_out), dim=1)
        mlp_out = self.mlp(combined_out)
        dropped_out = self.dropout(mlp_out)
        value_out = self.value_tail(dropped_out)
        policy_out = self.policy_tail(dropped_out)
        points_out = self.points_tail(dropped_out)
        return value_out, points_out, policy_out

    def predict(self, skyjo: sj.Skyjo) -> SkyNetPrediction:
        """Takes a single skyjo representation and returns a prediction object.

        Prediction object returned contains numpy arrays instead of tensors.
        This function is mostly a convenience function for testing and
        debugging. For actual optimized usage the `forward` method offers less
        overhead and flexibility.
        """
        spatial_tensor = einops.rearrange(
            torch.tensor(
                sj.get_spatial_input(skyjo), dtype=torch.float32, device=self.device
            ),
            "p h w c -> 1 p h w c",
        )
        non_spatial_tensor = einops.rearrange(
            torch.tensor(
                sj.get_non_spatial_input(skyjo), dtype=torch.float32, device=self.device
            ),
            "f -> 1 f",
        )
        output = self.forward(spatial_tensor, non_spatial_tensor)

        return SkyNetPrediction.from_skynet_output(output)

    def save(self, dir: pathlib.Path) -> pathlib.Path:
        curr_utc_dt = datetime.datetime.now(tz=datetime.timezone.utc)
        model_path = dir / f"model_{curr_utc_dt.strftime('%Y%m%d_%H%M%S')}.pth"
        torch.save(
            self.state_dict(),
            model_path,
        )
        return model_path


SkyNetOutput: typing.TypeAlias = tuple[
    torch.Tensor,  # value
    torch.Tensor,  # points
    torch.Tensor,  # policy
]
SkyNetNumpyOutput: typing.TypeAlias = tuple[
    np.ndarray[tuple[int, ...], np.float32],  # value
    np.ndarray[tuple[int, ...], np.float32],  # points
    np.ndarray[tuple[int, ...], np.float32],  # policy
]
SkyNet: typing.TypeAlias = SkyNet1D | SkyNet2D

if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    players = 2
    game_state = sj.new(players=players)
    players = game_state[3]
    model = SkyNet1D(
        spatial_input_shape=(players, sj.ROW_COUNT, sj.COLUMN_COUNT, sj.FINGER_SIZE),
        non_spatial_input_shape=(sj.GAME_SIZE,),
        value_output_shape=(players,),
        policy_output_shape=(sj.MASK_SIZE,),
    )
    model.set_device(torch.device("mps"))
    game_state = sj.start_round(game_state)
    model.eval()
    prediction = model.predict(game_state)
    print(prediction)

    model_path = model.save(pathlib.Path("models/test/"))
    loaded_model = SkyNet1D(
        spatial_input_shape=model.spatial_input_shape,
        non_spatial_input_shape=model.non_spatial_input_shape,
        value_output_shape=model.value_output_shape,
        policy_output_shape=model.policy_output_shape,
        dropout_rate=model.dropout_rate,
    )
    loaded_model.load_state_dict(torch.load(model_path, weights_only=True))
    loaded_model.set_device(model.device)
    loaded_model.eval()
    loaded_prediction = loaded_model.predict(game_state)
    print(loaded_prediction)
    assert np.allclose(
        prediction.policy_output.probabilities,
        loaded_prediction.policy_output.probabilities,
    )
    assert np.allclose(
        prediction.value_output.state_values,
        loaded_prediction.value_output.state_values,
    )
    assert np.allclose(
        prediction.points_output.points,
        loaded_prediction.points_output.points,
    )
