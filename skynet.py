from __future__ import annotations

import dataclasses
import typing

import einops
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

import abstract
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
    """Get the outcome of the game from the perspective of the current player."""
    players = skyjo[3]
    outcome = np.zeros((players,), dtype=np.float32)
    outcome[sj.get_fixed_perspective_winner(skyjo)] = 1.0
    return outcome


def state_value_for_player(state_value: StateValue, player: int) -> float:
    """Get the value of the game for a given player."""
    state_value = state_value.squeeze()
    assert len(state_value.shape) == 1, "Expected a 1D state value"
    return state_value[player].item()


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

    @classmethod
    def from_tensor_output(cls, output: torch.Tensor) -> typing.Self:
        return cls(output.detach().numpy(), is_renormalized=False)

    def __post_init__(self):
        assert abs(self.probabilities.sum() - 1) < 1e-6, (
            f"expected probabilities to sum to 1, got {self.probabilities.sum()}"
        )

    def mask_invalid_and_renormalize(
        self, valid_actions_mask: npt.NDArray[np.int8]
    ) -> typing.Self:
        assert valid_actions_mask.shape == self.probabilities.shape, (
            f"expected valid_actions_mask of shape {self.probabilities.shape}, got {valid_actions_mask.shape}"
        )
        valid_action_probabilities = self.probabilities * valid_actions_mask
        total_valid_action_probabilities = einops.reduce(
            valid_action_probabilities, "b ... -> b", reduction="sum"
        )
        num_valid_actions = einops.reduce(
            valid_actions_mask, "b ... -> b", reduction="sum"
        )
        assert not np.any(num_valid_actions == 0), (
            "expected no samples with no valid actions"
        )
        # Change denominator to 1 if total probability is 0 to make division safe
        safe_denominator = np.where(
            total_valid_action_probabilities == 0, 1.0, total_valid_action_probabilities
        )
        renormalized_valid_action_probabilities = (
            valid_action_probabilities / safe_denominator
        )
        # Assign uniform probability where total probability is 0
        renormalized_valid_action_probabilities = np.where(
            total_valid_action_probabilities == 0,
            1 / num_valid_actions,
            renormalized_valid_action_probabilities,
        )

        return PolicyOutput(
            renormalized_valid_action_probabilities, is_renormalized=True
        )

    def get_action_probability(self, action: sj.SkyjoAction) -> float:
        assert self.probabilities.shape[0] == 1, (
            f"expected 1 sample, got {self.probabilities.shape[0]}"
        )
        return self.probabilities[:, action].item()


@dataclasses.dataclass(slots=True)
class ValueOutput:
    # _values is indexed starting with current player
    state_values: np.ndarray[tuple[int, int], np.float32]  # (batch_size, num_players)
    curr_player: int

    @classmethod
    def from_tensor_output(cls, output: torch.Tensor, curr_player: int) -> typing.Self:
        return cls(
            np.roll(output.detach().numpy(), -curr_player, axis=1),
            curr_player,
        )

    def state_value(self, sample_idx: int = 0) -> sj.StateValue:
        return self.state_values[sample_idx]


@dataclasses.dataclass(slots=True)
class PointDifferentialOutput:
    point_differentials: sj.StateValue
    curr_player: int

    @classmethod
    def from_tensor_output(cls, output: torch.Tensor, curr_player: int) -> typing.Self:
        return cls(
            np.roll(output.detach().numpy(), -curr_player, axis=1),
            curr_player,
        )


# Prediction dataclasses
@dataclasses.dataclass(slots=True)
class SkyNetPrediction(abstract.AbstractModelPrediction):
    policy_output: PolicyOutput
    point_differential_output: PointDifferentialOutput
    value_output: ValueOutput


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
                out_features=hand_embedding_features,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                in_features=hand_embedding_features,
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


class NonSpatialInputHead(nn.Module):
    """Handles the non-spatial features of the current skyjo state (i.e. the top card, discard pile, etc.)"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activations: list[nn.Module] = [nn.ReLU(inplace=True), nn.ReLU(inplace=True)],
    ):
        assert len(activations) == 2, "expected 2 activations, got {}".format(
            len(activations)
        )
        super(NonSpatialInputHead, self).__init__()
        self.out_features = out_features
        self.in_features = in_features
        self.linear1 = nn.Linear(in_features=in_features, out_features=out_features)
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
        return x


class ActionInputHead(nn.Module):
    """Handles the action features of the current skyjo state (i.e. the action to take)"""

    def __init__(
        self,
        action_shape: typing.Iterable[int],
        out_features: int,
        activations: list[nn.Module] = [nn.ReLU(inplace=True), nn.ReLU(inplace=True)],
    ):
        super(ActionInputHead, self).__init__()
        self.out_features = out_features
        self.action_shape = action_shape
        self.linear1 = nn.Linear(
            in_features=np.cumprod([dim for dim in action_shape])[-1],
            out_features=out_features,
        )
        self.activation1 = activations[0]
        self.linear2 = nn.Linear(
            in_features=out_features,
            out_features=out_features,
        )
        self.activation2 = activations[1]

    def forward(self, x):
        """
        Input: (N, T, W, H)
        Output: (N, out_features)
        """
        assert len(x.shape) == 4, f"expected 4D input (N, T, W, H), got {x.shape}"
        x = einops.rearrange(x, "b t w h -> b (t w h)")
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        return x


# MODEL TAILS (OUTPUT HEADS)
class PolicyTail(nn.Module):
    def __init__(self, num_features: int, out_features: tuple[int, int, int]):
        super(PolicyTail, self).__init__()
        self.num_features = num_features
        self.out_features = out_features
        # INPUT: (N, F)
        # OUTPUT: (N, A)
        self.linear1 = nn.Linear(
            in_features=num_features,
            out_features=out_features,  # number of possible moves
        )
        # INPUT: (N, A)
        # OUTPUT: (N, A)
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
        # INPUT: (N, P)
        # OUTPUT: (N, P)
        self.softmax1 = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Input: (N, num_features)
        Output: (N, P)
        """
        x = self.linear1(x)
        x = self.softmax1(x)
        return x


class PointDifferenceTail(nn.Module):
    def __init__(self, num_features: int, num_players: int):
        super(PointDifferenceTail, self).__init__()
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
    ):
        super(SkyNet1D, self).__init__()
        self.spatial_input_shape = spatial_input_shape
        self.non_spatial_input_shape = non_spatial_input_shape
        self.value_output_shape = value_output_shape
        self.policy_output_shape = policy_output_shape

        self.final_embedding_dim = 32
        self.spatial_input_head = Spatia1DInputHead(
            input_shape=spatial_input_shape,
            hand_embedding_features=32,
            out_features=32,
        )
        self.non_spatial_input_head = NonSpatialInputHead(
            in_features=non_spatial_input_shape[0],
            out_features=32,
        )
        self.mlp = nn.Sequential(
            nn.Linear(
                in_features=self.spatial_input_head.out_features
                + self.non_spatial_input_head.out_features,
                out_features=self.final_embedding_dim,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                in_features=self.final_embedding_dim,
                out_features=self.final_embedding_dim,
            ),
            nn.ReLU(inplace=True),
        )
        self.value_tail = ValueTail(
            num_features=self.final_embedding_dim,
            num_players=self.value_output_shape[0],
        )
        self.policy_tail = PolicyTail(
            num_features=self.final_embedding_dim,
            out_features=self.policy_output_shape[0],
        )
        self.point_difference_tail = PointDifferenceTail(
            num_features=self.final_embedding_dim,
            num_players=self.value_output_shape[0],
        )

    def forward(
        self,
        spatial_tensor: torch.Tensor,
        non_spatial_tensor: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        spatial_out = self.spatial_input_head(spatial_tensor)
        non_spatial_out = self.non_spatial_input_head(non_spatial_tensor)
        combined_out = torch.cat((spatial_out, non_spatial_out), dim=1)
        mlp_out = self.mlp(combined_out)
        value_out = self.value_tail(mlp_out)
        policy_out = self.policy_tail(mlp_out)
        point_difference_out = self.point_difference_tail(mlp_out)
        return value_out, policy_out, point_difference_out

    def predict(self, skyjo: sj.Skyjo) -> SkyNetPrediction:
        game, table, players = skyjo[0], skyjo[1], skyjo[3]
        spatial_tensor = torch.tensor(
            einops.rearrange(table[:players], "p h w c -> 1 p h w c"),
            dtype=torch.float32,
        )
        non_spatial_tensor = torch.tensor(
            einops.rearrange(game, "f -> 1 f"), dtype=torch.float32
        )
        value_out, policy_out, point_difference_out = self.forward(
            spatial_tensor, non_spatial_tensor
        )
        return SkyNetPrediction(
            policy_output=PolicyOutput.from_tensor_output(policy_out),
            value_output=ValueOutput.from_tensor_output(value_out, sj.get_turn(skyjo)),
            point_differential_output=PointDifferentialOutput.from_tensor_output(
                point_difference_out, sj.get_turn(skyjo)
            ),
        )


SkyNet: typing.TypeAlias = SkyNet1D

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
    game_state = sj.start_round(game_state)
    prediction = model.predict(game_state)
    print(prediction)
