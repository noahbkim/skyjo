import dataclasses
import typing

import einops
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

import abstract
import skyjo_immutable as sj

"""
einops and general dimension notation:

b,N = batch size
p,P = number of players
c,C = number of channels
w,W = width
h,H = height
n_a,A = action space
a = action_type
f,F = feature space
"""


## LOSS FUNCTIONS
def compute_policy_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    loss = -(target * torch.log(predicted)).sum(dim=1)
    return loss.mean()


def compute_value_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sum((target - predicted) ** 2) / target.size()[0]


# OUTPUT DATACLASSES
@dataclasses.dataclass(slots=True)
class PolicyOutput(abstract.AbstractModelPolicyOutput):
    probabilities: np.ndarray
    is_renormalized: bool = False

    @classmethod
    def from_tensor_output(cls, output: torch.Tensor) -> typing.Self:
        return cls(output.detach().numpy(), is_renormalized=False)

    def __post_init__(self):
        assert len(self.probabilities.shape) == 1 + len(sj.ACTION_SHAPE), (
            f"expected probabilities array of shape (N, {sj.ACTION_SHAPE}), got {self.probabilities.shape}"
        )
        assert self.probabilities.shape[1:] == sj.ACTION_SHAPE, (
            f"expected probabilities array of shape (N, {sj.ACTION_SHAPE}), got {self.probabilities.shape}"
        )
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
        # sum over all valid actions in each sample
        total_valid_action_probabilities = einops.reduce(
            valid_action_probabilities, "b a h w -> b 1 1 1", reduction="sum"
        )

        # Identify samples where the sum of valid probabilities is zero
        zero_valid_total_probabily_mask = (
            total_valid_action_probabilities == 0
        )  # Shape: (b, 1, 1, 1)

        # Case 1: Renormalize where total probability is > 0
        # Avoid division by zero using np.where; the result for zero_total_mask cases will be overridden later.
        safe_denominator = np.where(
            zero_valid_total_probabily_mask, 1.0, total_valid_action_probabilities
        )
        normalized_probs = valid_action_probabilities / safe_denominator

        # Case 2: Assign uniform probability where total probability is 0
        num_valid_actions = einops.reduce(
            valid_actions_mask, "b a h w -> b 1 1 1", reduction="sum"
        )
        assert len(num_valid_actions == 0) == 0, (
            "expected no samples with no valid actions"
        )
        uniform_probs = valid_actions_mask / num_valid_actions  # Shape: (b, a, h, w)

        # Combine the two cases
        renormalized_valid_action_probabilities = np.where(
            zero_valid_total_probabily_mask, uniform_probs, normalized_probs
        )

        return PolicyOutput(
            renormalized_valid_action_probabilities, is_renormalized=True
        )

    def get_action_probability(self, action: sj.SkyjoAction) -> float:
        assert self.probabilities.shape[0] == 1, (
            f"expected 1 sample, got {self.probabilities.shape[0]}"
        )
        if action.row_idx is None and action.col_idx is None:
            return self.probabilities[0, action.action_type.value, 0, 0].item()
        return self.probabilities[
            0,
            action.action_type.value,
            action.row_idx,
            action.col_idx,
        ].item()


@dataclasses.dataclass(slots=True)
class SkyjoGameStateValue(abstract.AbstractGameStateValue):
    """Class for storing game state values and accessing value from perspective of a player"""

    # _values is NOT indexed starting with current player
    # instead, indexed according to player number
    _values: np.ndarray

    @property
    def num_players(self) -> int:
        return self._values.shape[1]

    @classmethod
    def from_winning_player(cls, winning_player: int, num_players: int) -> typing.Self:
        result = np.zeros((1, num_players), dtype=np.float32)
        result[0, winning_player] = 1.0
        return cls(result)

    def __post_init__(self):
        assert len(self._values.shape) == 2, (
            f"expected 2D (num_samples, num_players) values array, got {self._values.shape}"
        )

    def numpy(self) -> np.ndarray:
        return self._values

    def values_from_perspective_of(self, player: int) -> npt.NDArray[np.float32]:
        return self._values[:, player]

    def value_from_perspective_of(self, player: int) -> float:
        assert self._values.shape[0] == 1, (
            f"expected 1 sample, got {self._values.shape[0]}"
        )
        return self._values[:, player].item()


@dataclasses.dataclass(slots=True)
class ValueOutput(abstract.AbstractModelValueOutput):
    # _values is indexed starting with current player
    _values: np.ndarray
    curr_player: int
    num_players: int

    @property
    def game_state_value(self) -> SkyjoGameStateValue:
        return SkyjoGameStateValue(np.roll(self._values, -self.curr_player, axis=1))

    def __post_init__(self):
        assert len(self._values.shape) == 2, (
            f"expected 2D (num_samples, num_players) values array, got {self._values.shape}"
        )
        assert self.num_players == self._values.shape[1], (
            f"expected {self.num_players} players, got {self._values.shape[1]} players"
        )

    @classmethod
    def from_tensor_output(
        cls, output: torch.Tensor, num_players: int, curr_player: int
    ) -> typing.Self:
        return cls(output.detach().numpy(), curr_player, num_players)


@dataclasses.dataclass(slots=True)
class PointDifferentialOutput:
    point_differentials: npt.NDArray[np.float32]
    curr_player: int
    num_players: int

    @property
    def game_state_value(self) -> SkyjoGameStateValue:
        return SkyjoGameStateValue(
            np.roll(self.point_differentials, -self.curr_player, axis=1)
        )

    @classmethod
    def from_tensor_output(
        cls, output: torch.Tensor, num_players: int, curr_player: int
    ) -> typing.Self:
        return cls(output.detach().numpy(), curr_player, num_players)

    def __post_init__(self):
        assert len(self.point_differentials.shape) == 2, (
            f"expected 2D (num_samples, num_players) point_differentials array, got {self.point_differentials.shape}"
        )
        assert self.point_differentials.shape[1] == self.num_players, (
            f"expected point_differentials of shape (n, {self.num_players}), got {self.point_differentials.shape}"
        )


# Prediction dataclasses
@dataclasses.dataclass(slots=True)
class SkyNetPrediction(abstract.AbstractModelPrediction):
    policy_output: PolicyOutput
    point_differential_output: PointDifferentialOutput
    value_output: ValueOutput


@dataclasses.dataclass(slots=True)
class SkyNetAfterstatePrediction(abstract.AbstractModelPrediction):
    outcome_output: PolicyOutput
    point_differential_output: PointDifferentialOutput
    value_output: ValueOutput


@dataclasses.dataclass(slots=True)
class ActionProbabilityDistribution:
    action_probabilities: npt.NDArray[np.float32]

    def get_action_probability(self, action: sj.SkyjoAction) -> float:
        if action.action_type == sj.SkyjoActionType.DRAW:
            return self.action_probabilities[0, 1].item()
        return self.action_probabilities[
            action.action_type.value,
            sj.Hand.flat_index(action.row_idx, action.col_idx),
        ].item()

    # TODO: think a bit more about any class methods and how this applies to chance nodes


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


class SpatialInputHead(nn.Module):
    """Handles the spatial features of the current skyjo state (i.e. the player hands)"""

    # TODO: think more about second residual block and what parameters make sense
    def __init__(self, in_channels: int, num_players: int):
        super(SpatialInputHead, self).__init__()
        self.num_players = num_players
        # Input: (N * P, C, H, W)
        # Output: (N, 64, 1, W)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=(sj.NUM_ROWS, 1),
            stride=(1, 1),
            padding=(0, 0),
        )
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.activation1 = nn.ReLU(inplace=True)
        # Input: (N, 64, P, W)
        # Output: (N, 64, P, W)
        self.resblock2 = ResBlock(
            in_channels=64,
            out_channels=[64, 64],
            kernel_sizes=[(3, 3), (3, 3)],
            strides=[(1, 1), (1, 1)],
            paddings=[(1, 1), (1, 1)],
            activations=[nn.ReLU(inplace=True), nn.ReLU(inplace=True)],
            batch_norm_kwargs=[{}, {}],
        )

    def forward(self, x):
        assert len(x.shape) == 5, f"expected 5D input (N, P, C, H, W), got {x.shape}"
        # combining players into batch dimension so each hand is treated independently first
        # input has column first then row so switching to more standard row then column
        x = einops.rearrange(x, "b p c w h -> (b p) c h w", p=self.num_players)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        # Remove now collased row dimension and add player dimension back as "row"
        # Allows a "seeing" all embedded hands at once
        x = einops.rearrange(x, "(b p) c 1 w -> b c p w", p=self.num_players)
        x = self.resblock2(x)

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
        self.linear1 = nn.Linear(in_features=in_features, out_features=out_features)
        self.activation1 = activations[0]
        self.linear2 = nn.Linear(in_features=out_features, out_features=out_features)
        self.activation2 = activations[1]

    def forward(self, x):
        assert len(x.shape) == 2, f"expected 2D input (N, F), got {x.shape}"
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        return x


class PolicyHead(nn.Module):
    def __init__(self, num_features: int, action_shape: tuple[int, int, int]):
        super(PolicyHead, self).__init__()
        linear_out_length = np.cumprod([dim for dim in action_shape])[-1]
        # INPUT: (N, F)
        # OUTPUT: (N, A)
        self.linear1 = nn.Linear(
            in_features=num_features,
            out_features=linear_out_length,  # number of possible moves
        )
        # INPUT: (N, A)
        # OUTPUT: (N, A)
        self.softmax1 = nn.Softmax(dim=1)

    def forward(self, x):
        assert len(x.shape) == 2, f"expected 2D input (N, F), got {x.shape}"
        x = self.linear1(x)
        x = self.softmax1(x)
        return einops.rearrange(
            x,
            "b (t h w) -> b t h w",
            t=sj.ACTION_SHAPE[0],
            h=sj.ACTION_SHAPE[1],
            w=sj.ACTION_SHAPE[2],
        )


class ValueHead(nn.Module):
    def __init__(self, num_features: int, num_players: int):
        super(ValueHead, self).__init__()
        # INPUT: (N, F)
        # OUTPUT: (N, P)
        self.linear1 = nn.Linear(in_features=num_features, out_features=num_players)
        # INPUT: (N, P)
        # OUTPUT: (N, P)
        self.softmax1 = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.softmax1(x)
        return x


class PointDifferenceHead(nn.Module):
    def __init__(self, num_features: int, num_players: int):
        super(PointDifferenceHead, self).__init__()
        self.linear1 = nn.Linear(in_features=num_features, out_features=num_players)

    def forward(self, x):
        return self.linear1(x)


# TODO: Parameterize this more fully
class SkyNet(nn.Module):
    def __init__(
        self,
        spatial_input_channels: int,
        non_spatial_features: int,
        num_players: int = 2,
        device: torch.device = torch.device("cpu"),
    ):
        super(SkyNet, self).__init__()

        self.device = device
        self.num_players = num_players
        self.spatial_input_head = SpatialInputHead(
            in_channels=spatial_input_channels, num_players=self.num_players
        )
        self.non_spatial_input_head = NonSpatialInputHead(
            in_features=non_spatial_features, out_features=128
        )
        # Each non-spatial input is expanded and appended to each players input.
        # This may not be necessary, and it may just require a single global context.
        non_spatial_output_features = 128 * self.num_players
        flattened_spatial_output_features = 64 * self.num_players * sj.NUM_COLUMNS
        flattened_combined_input_length = (
            non_spatial_output_features + flattened_spatial_output_features
        )
        self.mlp = nn.Sequential(
            nn.Linear(in_features=flattened_combined_input_length, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(inplace=True),
        )
        self.policy_head = PolicyHead(num_features=128, action_shape=sj.ACTION_SHAPE)
        self.value_head = ValueHead(num_features=128, num_players=self.num_players)
        self.point_difference_head = PointDifferenceHead(
            num_features=128, num_players=self.num_players
        )

    def forward(
        self,
        spatial_features_input: torch.Tensor,
        non_spatial_features_input: torch.Tensor,
    ):
        assert len(spatial_features_input.shape) == 5, (
            f"expected 5D input (N, P, C, H, W), got {spatial_features_input.shape}"
        )
        assert len(non_spatial_features_input.shape) == 2, (
            f"expected 2D input (N, F), got {non_spatial_features_input.shape}"
        )
        spatial_out = self.spatial_input_head(spatial_features_input)
        non_spatial_out = self.non_spatial_input_head(non_spatial_features_input)
        # For now just flatten and combine features
        # TODO: try implementing attention mechanism here
        flattened_spatial_out = einops.rearrange(spatial_out, "b c p w -> b p (c w)")
        expanded_non_spatial_out = einops.repeat(
            non_spatial_out, "b f -> b p f", p=self.num_players
        )
        # Combines spatial and non-spatial features shape: (N, P * (C * W + F))
        combined_features = einops.rearrange(
            torch.cat((flattened_spatial_out, expanded_non_spatial_out), dim=2),
            "b p f -> b (p f)",
        )
        mlp_out = self.mlp(combined_features)
        policy_probs = self.policy_head(mlp_out)
        values = self.value_head(mlp_out)
        point_differentials = self.point_difference_head(mlp_out)
        return policy_probs, values, point_differentials

    def afterstate_forward(
        self,
        spatial_input: torch.Tensor,
        non_spatial_input: torch.Tensor,
        action_input: torch.Tensor,
    ) -> SkyNetAfterstatePrediction:
        # TEMPORARY: return forward of current state
        # Later, figure out if we want to do some sort of chance outcome prediction here
        return self.forward(spatial_input, non_spatial_input)

    def predict(self, state: sj.ImmutableState) -> SkyNetPrediction:
        """Takes a skyjo state and makes a prediction"""
        spatial_input = einops.rearrange(
            torch.tensor(state.spatial_numpy(), dtype=torch.float32),
            "p h w c -> 1 p c h w",
        )
        non_spatial_input = einops.rearrange(
            torch.tensor(state.non_spatial_numpy(), dtype=torch.float32),
            "f -> 1 f",
        )
        policy_tensor, value_tensor, point_differentials_tensor = self.forward(
            spatial_input, non_spatial_input
        )
        return SkyNetPrediction(
            policy_output=PolicyOutput.from_tensor_output(policy_tensor),
            value_output=ValueOutput.from_tensor_output(
                value_tensor, state.num_players, state.curr_player
            ),
            point_differential_output=PointDifferentialOutput.from_tensor_output(
                point_differentials_tensor, state.num_players, state.curr_player
            ),
        )

    def afterstate_predict(
        self, state: sj.ImmutableState, action: sj.SkyjoAction
    ) -> SkyNetAfterstatePrediction:
        spatial_input = einops.rearrange(
            torch.tensor(state.spatial_numpy(), dtype=torch.float32),
            "p h w c -> 1 p c h w",
        )
        non_spatial_input = einops.rearrange(
            torch.tensor(state.non_spatial_numpy(), dtype=torch.float32),
            "f -> 1 f",
        )
        action_input = einops.rearrange(
            torch.tensor(action.numpy(), dtype=torch.float32),
            "a w h -> 1 a w h",
        )
        outcome_tensor, value_tensor, point_differentials_tensor = (
            self.afterstate_forward(spatial_input, non_spatial_input, action_input)
        )
        return SkyNetAfterstatePrediction(
            outcome_output=PolicyOutput.from_tensor_output(outcome_tensor),
            value_output=ValueOutput.from_tensor_output(
                value_tensor, state.num_players, state.curr_player
            ),
            point_differential_output=PointDifferentialOutput.from_tensor_output(
                point_differentials_tensor, state.num_players, state.curr_player
            ),
        )


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    deck = sj.CardCounts.create_initial_deck_counts()
    state = sj.ImmutableState(
        num_players=2,
        player_scores=np.array([0, 0], dtype=np.int16),
        remaining_card_counts=deck,
        valid_actions=[sj.SkyjoAction(action_type=sj.SkyjoActionType.START_ROUND)],
    ).start_round(deck.generate_random_card())
    model = SkyNet(
        spatial_input_channels=sj.CARD_TYPES,
        non_spatial_features=state.non_spatial_numpy().shape[0],
        num_players=state.num_players,
    )
    prediction = model.predict(state)
    print(prediction)
