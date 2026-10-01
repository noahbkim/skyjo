from __future__ import annotations

import dataclasses
import datetime
import pathlib
import typing

import einops
import numpy as np
import torch
import torch.nn as nn

from . import checkpoint
from . import game as sj

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


# MARK: State Value


StateValue: typing.TypeAlias = np.ndarray[tuple[int], np.float32]
"""A vector representing the value of a Skyjo game for each player.

IMPORTANT: This is from a fixed perspective and not relative to the current player
    i.e. the first element is always the value of player 0, the second is for player 1, etc.

This is higher-is-better from the perspective of each player.
"""

SCORE_DIFFERENTIAL_CAP = 156.0
ROUND_SCORE_MIN = -48.0
ROUND_SCORE_MAX = 288.0
ROUND_SCORE_RANGE = ROUND_SCORE_MAX - ROUND_SCORE_MIN
ROUND_SCORE_TARGET_NAME = "round_score"
FUTURE_CLEAR_TARGET_NAME = "future_clear"
EQUIVARIANT_ARCHITECTURE_NAME = "hierarchical_equivariant_v2"
EQUIVARIANT_AUX_ARCHITECTURE_NAME = "hierarchical_equivariant_v3_aux"
EQUIVARIANT_SCORE_AUX_ARCHITECTURE_NAME = "hierarchical_equivariant_v3_score_aux"


def skyjo_to_state_value(skyjo: sj.Skyjo) -> StateValue:
    """Get the outcome of the game from the fixed perspective."""
    players = skyjo[3]
    outcome = np.zeros((players,), dtype=np.float32)
    outcome[sj.get_fixed_perspective_winner(skyjo)] = 1.0
    return outcome


def skyjo_to_game_state_value(skyjo: sj.Skyjo) -> StateValue:
    """Exact full-game outcome in fixed player order, sharing tied wins equally."""
    if not sj.get_game_over(skyjo):
        raise ValueError("Full-game outcomes require a completed game")
    scores = sj.get_fixed_perspective_game_scores(skyjo)
    winners = (scores == scores.min()).astype(np.float32)
    return winners / winners.sum()


def scores_to_score_differential_value(
    scores: np.ndarray[tuple[int], np.float32] | np.ndarray[tuple[int], np.int16],
) -> StateValue:
    """Convert final scores into higher-is-better score-differential values.

    Skyjo is lower-is-better, so the best score maps to 0 and every worse
    score maps to a negative margin from the best score.
    """
    scores = scores.astype(np.float32)
    return (scores.min() - scores).astype(np.float32)


def skyjo_to_score_differential_state_value(skyjo: sj.Skyjo) -> StateValue:
    """Get raw score-differential utility from the fixed perspective."""
    return scores_to_score_differential_value(
        sj.get_fixed_perspective_round_scores(skyjo)
    )


def normalize_round_scores(
    scores: np.ndarray[tuple[int], np.float32] | np.ndarray[tuple[int], np.int16],
) -> StateValue:
    """Normalize Skyjo round scores to [0, 1] using the configured score bounds."""
    return ((scores.astype(np.float32) - ROUND_SCORE_MIN) / ROUND_SCORE_RANGE).astype(
        np.float32
    )


def skyjo_to_normalized_round_score_state_value(skyjo: sj.Skyjo) -> StateValue:
    """Get normalized round scores from the fixed perspective."""
    return normalize_round_scores(sj.get_fixed_perspective_round_scores(skyjo))


def compose_search_value(
    outcome_probabilities: StateValue,
    normalized_round_scores: StateValue | None,
    score_utility_weight: float,
) -> StateValue:
    """Compose higher-is-better search utility from win and score predictions."""
    if score_utility_weight < 0:
        raise ValueError("score_utility_weight cannot be negative")
    outcomes = np.asarray(outcome_probabilities, dtype=np.float32)
    if score_utility_weight == 0 or normalized_round_scores is None:
        return outcomes.copy()
    scores = np.asarray(normalized_round_scores, dtype=np.float32)
    if scores.shape != outcomes.shape:
        raise ValueError(
            f"normalized_round_scores must have shape {outcomes.shape}, got {scores.shape}"
        )
    return (outcomes + score_utility_weight * (1.0 - scores)).astype(np.float32)


def skyjo_to_search_state_value(
    skyjo: sj.Skyjo,
    score_utility_weight: float,
) -> StateValue:
    """Get exact terminal search utility from the fixed player perspective."""
    return compose_search_value(
        skyjo_to_state_value(skyjo),
        skyjo_to_normalized_round_score_state_value(skyjo),
        score_utility_weight,
    )


def state_value_for_player(state_value: StateValue, player: int) -> float:
    """Get the value of the game for a given player."""
    state_value = state_value.squeeze()
    assert len(state_value.shape) == 1, "Expected a 1D state value"
    return state_value[player].item()


def to_state_value(
    value_output: np.ndarray[tuple[int], np.float32], curr_player: int
) -> StateValue:
    return np.roll(value_output, shift=curr_player)


def get_spatial_state_numpy(
    skyjo: sj.Skyjo,
) -> np.ndarray[tuple[int], np.float32]:
    return sj.get_table(skyjo).astype(np.float32)


def get_non_spatial_input_shape(players: int) -> tuple[int]:
    """Return the complete non-spatial observation shape for a player count."""
    if players < 1:
        raise ValueError("players must be positive")
    return (sj.GAME_SIZE + sj.CARD_SIZE + 2 + players,)


def get_non_spatial_state_numpy(
    skyjo: sj.Skyjo,
) -> np.ndarray[tuple[int], np.float32]:
    players = sj.get_player_count(skyjo)
    turn = sj.get_turn(skyjo)
    countdown = sj.get_countdown(skyjo)
    turns_since_reveal = turn - sj.get_last_revealed_turns(skyjo)
    observation = np.concatenate(
        (
            sj.get_game(skyjo),
            sj.get_deck(skyjo),
            np.array(
                [turn, -1 if countdown is None else countdown],
                dtype=np.int16,
            ),
            turns_since_reveal,
        )
    ).astype(np.float32)
    expected_shape = get_non_spatial_input_shape(players)
    if observation.shape != expected_shape:
        raise ValueError(
            f"non-spatial observation has shape {observation.shape}, "
            f"expected {expected_shape}"
        )
    return observation


# MARK: Policy Targets


def symmetrize_policy_target(
    state: sj.Skyjo,
    policy_target: np.ndarray[tuple[int], np.float32],
) -> np.ndarray[tuple[int], np.float32]:
    """Average positional policy mass over board-symmetry action orbits.

    Rows may be permuted independently within a column, and columns may be
    permuted as units. Two active-player slots therefore share an orbit when
    their finger states match and their columns contain the same multiset of
    finger states. Flip and replace actions are averaged independently.
    """
    if policy_target.shape != (sj.MASK_SIZE,):
        raise ValueError(
            f"policy_target must have shape {(sj.MASK_SIZE,)}, "
            f"got {policy_target.shape}"
        )

    symmetrized = np.array(policy_target, dtype=np.float32, copy=True)
    active_board = sj.get_table(state)[0]
    finger_states = np.argmax(active_board, axis=-1)
    column_signatures = [
        tuple(sorted(int(finger) for finger in finger_states[:, column]))
        for column in range(sj.COLUMN_COUNT)
    ]
    slot_orbits: dict[tuple[int, tuple[int, ...]], list[int]] = {}
    for row in range(sj.ROW_COUNT):
        for column in range(sj.COLUMN_COUNT):
            slot = row * sj.COLUMN_COUNT + column
            orbit_key = (
                int(finger_states[row, column]),
                column_signatures[column],
            )
            slot_orbits.setdefault(orbit_key, []).append(slot)

    for action_offset in (sj.MASK_FLIP, sj.MASK_REPLACE):
        for slots in slot_orbits.values():
            action_indices = np.asarray(slots, dtype=np.intp) + action_offset
            symmetrized[action_indices] = symmetrized[action_indices].mean()
    return symmetrized


# MARK: Model Output


class SupportsCoreSkyNetOutput(typing.Protocol):
    """Shared tensor fields required by the core SkyNet losses."""

    value: torch.Tensor
    policy_logits: torch.Tensor


class EquivariantOutput(typing.NamedTuple):
    """Core output returned by EquivariantSkyNet.

    The field order is intentionally tuple-compatible with the base
    value/policy model contract.
    """

    value: torch.Tensor
    policy_logits: torch.Tensor


class EquivariantAuxOutput(typing.NamedTuple):
    """Core output plus opt-in auxiliary predictions for training experiments."""

    value: torch.Tensor
    policy_logits: torch.Tensor
    auxiliary_outputs: dict[str, torch.Tensor]


SkyNetOutput: typing.TypeAlias = EquivariantOutput | EquivariantAuxOutput


class SkyNetNumpyOutput(typing.NamedTuple):
    value: np.ndarray[tuple[int, ...], np.float32]
    policy_logits: np.ndarray[tuple[int, ...], np.float32]
    auxiliary_outputs: dict[str, np.ndarray[tuple[int, ...], np.float32]]


def batch_mask_and_renormalize_policy_probabilities(
    batch_policies: np.ndarray[tuple[int, int], np.float32],
    batch_masks: np.ndarray[tuple[int, int], np.int8],
) -> np.ndarray[tuple[int, int], np.float32]:
    assert batch_policies.shape == batch_masks.shape, (
        f"expected valid_actions_mask of shape {batch_policies.shape}, got {batch_masks.shape}"
    )
    valid_action_probabilities = batch_policies * batch_masks
    total_valid_action_probabilities = einops.reduce(
        valid_action_probabilities, "b ... -> b", reduction="sum"
    )
    num_valid_actions = einops.reduce(batch_masks, "b ... -> b", reduction="sum")
    assert not np.any(num_valid_actions == 0), (
        "expected no samples with no valid actions"
    )
    # Change denominator to 1 if total probability is 0 to make division safe
    zero_probability_rows = total_valid_action_probabilities == 0
    safe_denominator = np.where(
        zero_probability_rows, 1.0, total_valid_action_probabilities
    )
    renormalized_valid_action_probabilities = (
        valid_action_probabilities / safe_denominator[:, np.newaxis]
    )
    # Assign uniform probability where total probability is 0
    renormalized_valid_action_probabilities = np.where(
        zero_probability_rows[:, np.newaxis],
        batch_masks / num_valid_actions[:, np.newaxis],
        renormalized_valid_action_probabilities,
    )
    return renormalized_valid_action_probabilities


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
    auxiliary_outputs = getattr(model_output, "auxiliary_outputs", None)
    if auxiliary_outputs is not None:
        return type(model_output)(
            model_output.value[idx],
            model_output.policy_logits[idx],
            {name: value[idx] for name, value in auxiliary_outputs.items()},
        )
    return type(model_output)(
        model_output.value[idx],
        model_output.policy_logits[idx],
    )


def output_to_numpy(output: SkyNetOutput) -> SkyNetNumpyOutput:
    """Converts a SkyNetOutput to a SkyNetNumpyOutput.

    Handles the detaching and converting to numpy
    (including copying to cpu when device is not cpu).
    """
    value_output = output.value.detach()
    policy_output = output.policy_logits.detach()
    auxiliary_outputs = {
        name: value.detach().cpu().numpy()
        for name, value in getattr(output, "auxiliary_outputs", {}).items()
    }
    if value_output.device != torch.device("cpu"):
        value_output = value_output.cpu()
    if policy_output.device != torch.device("cpu"):
        policy_output = policy_output.cpu()
    return SkyNetNumpyOutput(
        value_output.numpy(),
        policy_output.numpy(),
        auxiliary_outputs,
    )


def numpy_to_tensors(
    *numpy_arrays: np.ndarray[tuple[int, ...], np.float32],
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, ...]:
    return tuple(
        torch.tensor(array, dtype=dtype, device=device) for array in numpy_arrays
    )


@dataclasses.dataclass(slots=True)
class SkyNetPrediction:
    value_output: np.ndarray[tuple[int], np.float32]
    policy_output: np.ndarray[tuple[int], np.float32]
    policy_logits: np.ndarray[tuple[int], np.float32] | None = None
    auxiliary_outputs: dict[str, np.ndarray[tuple[int, ...], np.float32]] | None = None

    @classmethod
    def from_skynet_output(
        cls,
        output: SkyNetOutput,
    ) -> SkyNetPrediction:
        numpy_output = output_to_numpy(output)
        single_output = get_single_model_output(numpy_output, 0)
        value_numpy = single_output.value
        policy_logits_numpy = single_output.policy_logits

        # Convert masked policy logits to probabilities
        # The logits from PolicyTail.forward are already masked (large negative numbers for invalid actions)
        # A standard softmax will handle these correctly, assigning near-zero probability to masked actions.
        policy_exp_logits = np.exp(
            policy_logits_numpy - np.max(policy_logits_numpy, axis=-1, keepdims=True)
        )  # for numerical stability
        policy_probabilities_numpy = policy_exp_logits / np.sum(
            policy_exp_logits, axis=-1, keepdims=True
        )

        # Convert masked policy logits to probabilities
        # The logits from PolicyTail.forward are already masked (large negative numbers for invalid actions)
        # A standard softmax will handle these correctly, assigning near-zero probability to masked actions.

        assert (
            len(value_numpy.shape)
            == len(policy_probabilities_numpy.shape)
            == 1
        ), (
            "expected value_output and policy_output to be a single result and not batched results."
            f"value_output.shape: {value_numpy.shape}, policy_output.shape: {policy_probabilities_numpy.shape}"
        )
        return SkyNetPrediction(
            value_output=value_numpy,
            policy_output=policy_probabilities_numpy,
            policy_logits=policy_logits_numpy,
            auxiliary_outputs=single_output.auxiliary_outputs or None,
        )

    def __str__(self) -> str:
        return f"{self.value_output}\n{self.policy_output}"

    def mask_and_renormalize(self, valid_actions_mask: np.ndarray[tuple[int], np.int8]):
        self.policy_output = mask_and_renormalize_policy_probabilities(
            self.policy_output, valid_actions_mask
        )

    @property
    def round_score_output(self) -> np.ndarray[tuple[int], np.float32] | None:
        if self.auxiliary_outputs is None:
            return None
        return self.auxiliary_outputs.get(ROUND_SCORE_TARGET_NAME)

    def search_value(self, score_utility_weight: float = 0.0) -> StateValue:
        """Return relative-player search utility for this prediction."""
        return compose_search_value(
            self.value_output,
            self.round_score_output,
            score_utility_weight,
        )

    def to_output(self) -> SkyNetOutput:
        assert self.policy_logits is not None, "expected policy logits"
        value = torch.tensor(np.expand_dims(self.value_output, 0), dtype=torch.float32)
        policy_logits = torch.tensor(
            np.expand_dims(self.policy_logits, 0), dtype=torch.float32
        )
        if self.auxiliary_outputs is None:
            return EquivariantOutput(value, policy_logits)
        return EquivariantAuxOutput(
            value,
            policy_logits,
            {
                name: torch.tensor(np.expand_dims(output, 0), dtype=torch.float32)
                for name, output in self.auxiliary_outputs.items()
            },
        )


# MARK: Tail Modules


class SimplePolicyLogitTail(nn.Module):
    """Simple tail that outputs policy logits.

    Transforms input using a single linear layer and applies optional
    masking to logits if provided."""

    def __init__(self, input_dimensions: int, output_dimensisons: int):
        super(SimplePolicyLogitTail, self).__init__()
        self.input_dimensions = input_dimensions
        self.output_dimensisons = output_dimensisons
        self.mlp = nn.Sequential(
            nn.Linear(
                in_features=self.input_dimensions,
                out_features=self.output_dimensisons,
            ),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """
        Input: (N, F)
        Output: (N, A)
        """
        assert len(x.shape) == 2, f"expected 2D input (N, F), got {x.shape}"
        logits = self.mlp(x)
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), -1e10)
        return logits


class SimpleOutcomeProbabilityTail(nn.Module):
    """Reusable outcome tail to predict winner probabilities over players.
    """

    def __init__(self, input_dimensions: int, players: int):
        super(SimpleOutcomeProbabilityTail, self).__init__()
        self.players = players
        self.input_dimensions = input_dimensions
        self.mlp = nn.Sequential(
            nn.Linear(
                in_features=self.input_dimensions,
                out_features=self.players,
            ),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        """
        Input: (N, F)
        Output: (N, P)
        """
        return self.mlp(x)


class BoundedScoreDifferentialTail(nn.Module):
    """Predict raw score-differential utility in the range [-156, 0]."""

    def __init__(
        self,
        input_dimensions: int,
        players: int,
        score_differential_cap: float = SCORE_DIFFERENTIAL_CAP,
    ):
        super(BoundedScoreDifferentialTail, self).__init__()
        self.players = players
        self.input_dimensions = input_dimensions
        self.score_differential_cap = score_differential_cap
        self.linear = nn.Linear(
            in_features=self.input_dimensions,
            out_features=self.players,
        )

    def forward(self, x):
        """
        Input: (N, F)
        Output: (N, P), where 0 is best and negative values are worse.
        """
        return -self.score_differential_cap * torch.sigmoid(self.linear(x))


class NormalizedRoundScoreTail(nn.Module):
    """Predict normalized final round scores for each player."""

    def __init__(self, input_dimensions: int, players: int):
        super(NormalizedRoundScoreTail, self).__init__()
        self.players = players
        self.input_dimensions = input_dimensions
        self.linear = nn.Linear(
            in_features=self.input_dimensions,
            out_features=self.players,
        )

    def forward(self, x):
        """
        Input: (N, F)
        Output: (N, P), normalized to [0, 1].
        """
        return torch.sigmoid(self.linear(x))


SimpleValueTail = SimpleOutcomeProbabilityTail


class EquivariantPolicyLogitTail(nn.Module):
    """Policy tail that outputs policy logits for Skyjo.

    Designed to be equivariant so swapping cards within a column will exactly
    swap the output logits in the flip and replace actions. Similar swapping
    whole columns will swap the logits corresponding to those columns.

    For the non-board based actions (initial flip, draw, take) the logits are a
    function of the global state embedding.

    For the flip and replace logits each board slot is transformed using a module
    that maps a slot + global state embedding to exactly one logit, the logit
    representing the flip or replace action on that slot. Since, the
    same transformation is applied to each board slot individually
    it is equivariant."""

    def __init__(
        self,
        embedding_dimensions: int,
        global_state_embedding_dimensions: int,
        non_positional_actions: int = 4,
        rows: int = 3,
        columns: int = 4,
    ):
        super(EquivariantPolicyLogitTail, self).__init__()
        self.embedding_dimensions = embedding_dimensions
        self.global_state_embedding_dimensions = global_state_embedding_dimensions
        self.non_positional_actions = non_positional_actions
        self.rows = rows
        self.columns = columns

        self.positional_logits_mlp = nn.Sequential(
            nn.Linear(
                in_features=2 * self.embedding_dimensions
                + self.global_state_embedding_dimensions,
                out_features=self.embedding_dimensions,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                in_features=self.embedding_dimensions,
                out_features=2,
            ),
        )
        self.non_positional_logits_mlp = nn.Sequential(
            # nn.Linear(
            #     in_features=self.global_state_embedding_dimensions,
            #     out_features=self.global_state_embedding_dimensions,
            # ),
            # nn.ReLU(inplace=True),
            nn.Linear(
                in_features=self.global_state_embedding_dimensions,
                out_features=self.non_positional_actions,
            ),
        )

    def forward(
        self,
        flattened_card_embeddings: torch.Tensor,
        flattened_column_embeddings: torch.Tensor,
        global_state_tensor: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        expanded_global_state_tensor = einops.repeat(
            global_state_tensor,
            "b f -> b (h w) f",
            h=self.rows,
            w=self.columns,
        )
        positional_features = torch.cat(
            (
                flattened_card_embeddings,
                flattened_column_embeddings,
                expanded_global_state_tensor,
            ),
            dim=2,
        )

        positional_logits = self.positional_logits_mlp(positional_features)
        flip_logits = positional_logits[:, :, 0]
        replace_logits = positional_logits[:, :, 1]
        non_positional_logits = self.non_positional_logits_mlp(global_state_tensor)
        logits = torch.cat((non_positional_logits, flip_logits, replace_logits), dim=1)
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), -1e10)
        return logits


class SimpleClearedCardsTail(nn.Module):
    """Shared per-column tail that outputs future-clear logits."""

    def __init__(
        self,
        global_state_embedding_dimensions: int,
        embedding_dimensions: int,
        players: int,
        columns: int,
    ):
        super(SimpleClearedCardsTail, self).__init__()
        self.global_state_embedding_dimensions = global_state_embedding_dimensions
        self.embedding_dimensions = embedding_dimensions
        self.players = players
        self.columns = columns
        self.linear = nn.Linear(
            in_features=self.global_state_embedding_dimensions
            + self.embedding_dimensions,
            out_features=1,
        )

    def forward(
        self,
        column_summaries: torch.Tensor,
        global_state_embedding: torch.Tensor,
    ):
        repeated_global_state_embedding = einops.repeat(
            global_state_embedding,  # (b f)
            "b f -> b p c f",
            p=self.players,
            c=self.columns,
        )
        x = torch.cat((column_summaries, repeated_global_state_embedding), dim=-1)
        return self.linear(x).squeeze(-1)


# MARK: SkyNets


class SimpleSkyNet(nn.Module):
    """Simple SkyNet Model.

    Leverages a single MLP to transform the concatenated spatial and non-spatial
    skyjo state features. Takes MLP output and passes to policy and value heads
    for final outputs."""

    def __init__(
        self,
        hidden_layers: list[int],
        spatial_input_shape: tuple[int, ...],  # (players, ...)
        non_spatial_input_shape: tuple[int],
        value_output_shape: tuple[int],  # (players,)
        policy_output_shape: tuple[int],  # (mask_size,)
        device: torch.device = torch.device("cpu"),
        dropout_rate: float = 0.0,
    ):
        import math

        super(SimpleSkyNet, self).__init__()
        self.spatial_input_shape = spatial_input_shape
        self.non_spatial_input_shape = non_spatial_input_shape
        self.value_output_shape = value_output_shape
        self.policy_output_shape = policy_output_shape
        self.device = device
        self.dropout_rate = dropout_rate
        self.set_device(device)
        in_features = [
            math.prod(spatial_input_shape) + math.prod(non_spatial_input_shape)
        ] + hidden_layers[:-1]
        out_features = hidden_layers
        linear_layers = [
            nn.Linear(in_features=in_features, out_features=out_features)
            for in_features, out_features in zip(in_features, out_features)
        ]
        with_activations = []
        for layer in range(len(linear_layers)):
            with_activations.append(linear_layers[layer])
            with_activations.append(nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(*with_activations, nn.Dropout(self.dropout_rate))
        self.value_tail = BoundedScoreDifferentialTail(
            hidden_layers[-1], value_output_shape[0]
        )
        self.policy_tail = SimplePolicyLogitTail(
            hidden_layers[-1], policy_output_shape[0]
        )

    def forward(
        self,
        spatial_tensor: torch.Tensor,
        non_spatial_tensor: torch.Tensor,
        mask: torch.Tensor,
    ) -> SkyNetOutput:
        spatial_tensor = einops.rearrange(spatial_tensor, "b p h w c -> b (p h w c)")
        x = torch.cat((spatial_tensor, non_spatial_tensor), dim=1)
        x = self.mlp(x)
        value_out = self.value_tail(x)
        policy_out = self.policy_tail(x, mask)
        return EquivariantOutput(
            value_out,
            policy_out,
        )

    @torch.inference_mode()
    def predict(self, skyjo: sj.Skyjo) -> SkyNetPrediction:
        self.eval()
        spatial_tensor = einops.rearrange(
            torch.tensor(
                sj.get_spatial_input(skyjo), dtype=torch.float32, device=self.device
            ),
            "p h w c -> 1 p h w c",
        ).contiguous()
        non_spatial_tensor = einops.rearrange(
            torch.tensor(
                get_non_spatial_state_numpy(skyjo),
                dtype=torch.float32,
                device=self.device,
            ),
            "f -> 1 f",
        ).contiguous()
        mask_tensor = torch.tensor(
            sj.actions(skyjo), dtype=torch.float32, device=self.device
        ).contiguous()
        output = self.forward(spatial_tensor, non_spatial_tensor, mask_tensor)

        return SkyNetPrediction.from_skynet_output(output)

    def set_device(self, device: torch.device):
        self.device = device
        self.to(device)

    def save(
        self,
        dir: pathlib.Path,
        optimizer: torch.optim.Optimizer | None = None,
        configuration: typing.Any = None,
        progress: checkpoint.TrainingProgress | None = None,
    ) -> pathlib.Path:
        curr_utc_dt = datetime.datetime.now(tz=datetime.timezone.utc)
        model_path = dir / (
            f"checkpoint_{curr_utc_dt.strftime('%Y%m%d_%H%M%S_%f')}.pth"
        )
        return checkpoint.save_checkpoint(
            model_path,
            model=self,
            optimizer=optimizer,
            configuration=configuration,
            progress=progress,
        )


class TransformerBlock(nn.Module):
    """
    One encoder-style transformer block à la Vaswani et al. (2017).

    Args
    ----
    embed_dim : int
        Token/patch embedding size (E).
    num_heads : int
        Number of attention heads (H).  E must be divisible by H.
    mlp_ratio : float
        Hidden size multiplier for the feed-forward “MLP”: usually 4.0.
    dropout    : float
        Dropout on attention weights and MLP activations.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        # --- Self-attention -------------------------------------------------
        self.self_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,  # (B, L, E) instead of (L, B, E)
        )

        # --- Two LayerNorms (post-LN setup) --------------------------------
        self.norm1 = nn.LayerNorm(embed_dim)

        # --- Feed-forward network (MLP) ------------------------------------
        if self.mlp_ratio is not None:
            hidden_dim = int(self.embed_dim * self.mlp_ratio)
            self.norm2 = nn.LayerNorm(self.embed_dim)
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(inplace=True),  # or nn.ReLU()
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, embed_dim),
                nn.Dropout(dropout),
            )

    # ----------------------------------------------------------------------
    def forward(self, x):
        """
        x : (batch, seq_len, embed_dim)
        """
        # --- Self-attention sub-layer -------------------------------------
        # LayerNorm first (post-LN). Residual added afterwards.
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            need_weights=False,
        )
        x = x + attn_out  # residual connection

        # --- Feed-forward sub-layer ---------------------------------------
        if self.mlp_ratio is not None:
            x_norm = self.norm2(x)
            mlp_out = self.mlp(x_norm)
            x = x + mlp_out  # residual connection

        return x


class ResidualAttentionBlock(nn.Module):
    """
    One "tiny" encoder-style transformer block à la Vaswani et al. (2017).
    This omits layernorms and the feed-forward sub-layer.

    Args
    ----
    embed_dim : int
        Token/patch embedding size (E).
    num_heads : int
        Number of attention heads (H).  E must be divisible by H.
    mlp_ratio : float
        Hidden size multiplier for the feed-forward “MLP”: usually 4.0.
    dropout    : float
        Dropout on attention weights and MLP activations.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        # --- Self-attention -------------------------------------------------
        self.self_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,  # (B, L, E) instead of (L, B, E)
        )

    # ----------------------------------------------------------------------
    def forward(self, x):
        """
        x : (batch, seq_len, embed_dim)
        """
        # --- Self-attention sub-layer -------------------------------------
        # LayerNorm first (post-LN). Residual added afterwards.
        attn_out, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            need_weights=False,
        )
        x = x + attn_out  # residual connection

        return x


class EquivariantSkyNet(nn.Module):
    """Equivariant SkyNet Model.

    The key property is that it is "equivariant" in the sense that swapping cards
    within a column will exactly swap the output logits in the flip and replace
    actions. Similar swapping whole columns will swap the logits corresponding
    to those columns. This was done be design since in the Skyjo game the order
    of the cards within the column do not matter in the evaluation of the game
    and on the underlying policy.

    Cards are contextualized within columns and columns are contextualized within each
    board before pooling. Ordered player-board summaries are concatenated to preserve
    strategically meaningful player order without a large player-level transformer.
    """

    architecture_name = EQUIVARIANT_ARCHITECTURE_NAME

    def __init__(
        self,
        spatial_input_shape: tuple[int, ...],  # (players, )
        non_spatial_input_shape: tuple[int],
        value_output_shape: tuple[int],  # (players,)
        policy_output_shape: tuple[int],  # (mask_size,)
        device: torch.device,
        embedding_dimensions: int = 16,
        global_state_embedding_dimensions: int = 32,
        num_heads: int = 4,
    ):
        super(EquivariantSkyNet, self).__init__()
        self.spatial_input_shape = spatial_input_shape
        self.non_spatial_input_shape = non_spatial_input_shape
        self.value_output_shape = value_output_shape
        self.policy_output_shape = policy_output_shape
        self.embedding_dimensions = embedding_dimensions
        self.global_state_embedding_dimensions = global_state_embedding_dimensions
        self.num_heads = num_heads
        self.players = self.spatial_input_shape[0]
        self.rows = self.spatial_input_shape[1]
        self.columns = self.spatial_input_shape[2]
        self.card_types = self.spatial_input_shape[3]

        expected_policy_output_shape = (
            4 + 2 * self.rows * self.columns,
        )
        if self.policy_output_shape != expected_policy_output_shape:
            raise ValueError(
                "policy_output_shape must be "
                f"{expected_policy_output_shape}, got {self.policy_output_shape}"
            )
        if self.value_output_shape != (self.players,):
            raise ValueError(
                f"value_output_shape must be {(self.players,)}, "
                f"got {self.value_output_shape}"
            )
        if self.embedding_dimensions % self.num_heads:
            raise ValueError(
                "embedding_dimensions must be divisible by num_heads, got "
                f"{self.embedding_dimensions} and {self.num_heads}"
            )
        if self.global_state_embedding_dimensions % self.num_heads:
            raise ValueError(
                "global_state_embedding_dimensions must be divisible by num_heads, got "
                f"{self.global_state_embedding_dimensions} and {self.num_heads}"
            )

        # Card Embedding
        self.card_embedder = nn.Linear(
            in_features=self.card_types,
            out_features=self.embedding_dimensions,
            bias=False,
        )

        # Non-Spatial Embedding
        self.non_spatial_embedder = nn.Linear(
            in_features=self.non_spatial_input_shape[0],
            out_features=self.embedding_dimensions,
            bias=False,
        )

        self.column_summary_token = nn.Parameter(
            torch.randn(1, 1, self.embedding_dimensions)
        )

        self.card_within_column_attention = nn.ModuleList(
            TransformerBlock(
                embed_dim=self.embedding_dimensions,
                num_heads=self.num_heads,
                mlp_ratio=None,
                dropout=0.0,
            )
            for _ in range(3)
        )
        self.column_within_board_attention = TransformerBlock(
            embed_dim=self.embedding_dimensions,
            num_heads=self.num_heads,
            mlp_ratio=None,
            dropout=0.0,
        )
        self.global_state_embedder = nn.Sequential(
            nn.Linear(
                in_features=self.embedding_dimensions * (self.players + 1),
                out_features=self.global_state_embedding_dimensions,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                in_features=self.global_state_embedding_dimensions,
                out_features=self.global_state_embedding_dimensions,
            ),
        )

        # Tails
        self.value_tail = SimpleOutcomeProbabilityTail(
            input_dimensions=self.global_state_embedding_dimensions,
            players=self.players,
        )
        self.policy_tail = EquivariantPolicyLogitTail(
            embedding_dimensions=self.embedding_dimensions,
            global_state_embedding_dimensions=self.global_state_embedding_dimensions,
            rows=self.rows,
            columns=self.columns,
        )
        self.set_device(device)

    def set_device(self, device: torch.device):
        self.device = device
        self.to(device)

    def save(
        self,
        dir: pathlib.Path,
        optimizer: torch.optim.Optimizer | None = None,
        configuration: typing.Any = None,
        progress: checkpoint.TrainingProgress | None = None,
    ) -> pathlib.Path:
        curr_utc_dt = datetime.datetime.now(tz=datetime.timezone.utc)
        model_path = dir / (
            f"checkpoint_{curr_utc_dt.strftime('%Y%m%d_%H%M%S_%f')}.pth"
        )
        return checkpoint.save_checkpoint(
            model_path,
            model=self,
            optimizer=optimizer,
            configuration=configuration,
            progress=progress,
        )

    def forward(
        self,
        spatial_tensor: torch.Tensor,
        non_spatial_tensor: torch.Tensor,
        mask: torch.Tensor,
    ) -> SkyNetOutput:
        features = self._forward_features(spatial_tensor, non_spatial_tensor)
        value_out = self.value_tail(features.global_state_embedding)
        policy_out = self._forward_policy(features, mask)
        return EquivariantOutput(
            value_out,
            policy_out,
        )

    @dataclasses.dataclass(slots=True)
    class Features:
        global_state_embedding: torch.Tensor
        active_player_card_embeddings: torch.Tensor
        active_player_column_embeddings: torch.Tensor
        all_player_column_embeddings: torch.Tensor

    def _forward_features(
        self,
        spatial_tensor: torch.Tensor,
        non_spatial_tensor: torch.Tensor,
    ) -> Features:
        card_embeddings = self.card_embedder(spatial_tensor)
        non_spatial_embeddings = self.non_spatial_embedder(non_spatial_tensor)
        card_embeddings = einops.rearrange(
            card_embeddings, "b p h w f -> (b p w) h f"
        ).contiguous()
        repeated_column_summary_tokens = einops.repeat(
            self.column_summary_token,
            "1 1 f -> bpw 1 f",
            bpw=card_embeddings.shape[0],
        )
        repeated_non_spatial_embeddings = einops.repeat(
            non_spatial_embeddings,
            "b f -> (b p w) 1 f",
            p=self.players,
            w=self.columns,
        )
        with_cls_tokens = torch.cat(
            (
                repeated_column_summary_tokens,
                repeated_non_spatial_embeddings,
                card_embeddings,
            ),
            dim=1,
        )
        # Try adding column summary token
        attended_cards = with_cls_tokens
        for attention_block in self.card_within_column_attention:
            attended_cards = attention_block(attended_cards)

        column_summaries = einops.rearrange(
            attended_cards[:, 0, :],
            "(b p w) f -> b p w f",
            b=spatial_tensor.shape[0],
            p=self.players,
            w=self.columns,
        )
        attended_cards = attended_cards[:, 2:, :]
        contextualized_columns = self.column_within_board_attention(
            einops.rearrange(column_summaries, "b p w f -> (b p) w f")
        )
        contextualized_columns = einops.rearrange(
            contextualized_columns,
            "(b p) w f -> b p w f",
            b=spatial_tensor.shape[0],
            p=self.players,
        )
        board_summaries = einops.reduce(
            contextualized_columns,
            "b p w f -> b (p f)",
            reduction="sum",
        )
        global_state_embedding = self.global_state_embedder(
            torch.cat(
                (
                    board_summaries,
                    non_spatial_embeddings,
                ),
                dim=1,
            )
        )
        active_player_card_embeddings = einops.rearrange(
            attended_cards,
            "(b p w) h f -> b p h w f",
            b=spatial_tensor.shape[0],
            p=self.players,
            w=self.columns,
        )[:, 0, :].contiguous()
        active_player_card_embeddings = einops.rearrange(
            active_player_card_embeddings,
            "b h w f -> b (h w) f",
        )
        active_player_column_embeddings = einops.repeat(
            contextualized_columns[:, 0, :, :],
            "b w f -> b (h w) f",
            h=self.rows,
        )
        return EquivariantSkyNet.Features(
            global_state_embedding=global_state_embedding,
            active_player_card_embeddings=active_player_card_embeddings,
            active_player_column_embeddings=active_player_column_embeddings,
            all_player_column_embeddings=contextualized_columns,
        )

    def _forward_policy(
        self,
        features: Features,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.policy_tail(
            features.active_player_card_embeddings,
            features.active_player_column_embeddings,
            features.global_state_embedding,
            mask,
        )

    @torch.inference_mode()
    def predict(self, skyjo: sj.Skyjo) -> SkyNetPrediction:
        self.eval()
        spatial_tensor = einops.rearrange(
            torch.tensor(
                sj.get_spatial_input(skyjo), dtype=torch.float32, device=self.device
            ),
            "p h w c -> 1 p h w c",
        )
        non_spatial_tensor = einops.rearrange(
            torch.tensor(
                get_non_spatial_state_numpy(skyjo),
                dtype=torch.float32,
                device=self.device,
            ),
            "f -> 1 f",
        )
        mask_tensor = torch.tensor(
            sj.actions(skyjo), dtype=torch.float32, device=self.device
        )
        output = self.forward(spatial_tensor, non_spatial_tensor, mask_tensor)
        return SkyNetPrediction.from_skynet_output(output)


class EquivariantSkyNetWithAuxiliaryHeads(EquivariantSkyNet):
    """EquivariantSkyNet with round-score and future-clear auxiliary heads."""

    architecture_name = EQUIVARIANT_AUX_ARCHITECTURE_NAME

    def __init__(
        self,
        spatial_input_shape: tuple[int, ...],
        non_spatial_input_shape: tuple[int],
        value_output_shape: tuple[int],
        policy_output_shape: tuple[int],
        device: torch.device,
        embedding_dimensions: int = 16,
        global_state_embedding_dimensions: int = 32,
        num_heads: int = 4,
    ):
        super().__init__(
            spatial_input_shape=spatial_input_shape,
            non_spatial_input_shape=non_spatial_input_shape,
            value_output_shape=value_output_shape,
            policy_output_shape=policy_output_shape,
            device=device,
            embedding_dimensions=embedding_dimensions,
            global_state_embedding_dimensions=global_state_embedding_dimensions,
            num_heads=num_heads,
        )
        self.round_score_tail = NormalizedRoundScoreTail(
            input_dimensions=self.global_state_embedding_dimensions,
            players=self.players,
        )
        self.future_clear_tail = SimpleClearedCardsTail(
            global_state_embedding_dimensions=self.global_state_embedding_dimensions,
            embedding_dimensions=self.embedding_dimensions,
            players=self.players,
            columns=self.columns,
        )
        self.set_device(device)

    def forward(
        self,
        spatial_tensor: torch.Tensor,
        non_spatial_tensor: torch.Tensor,
        mask: torch.Tensor,
    ) -> SkyNetOutput:
        features = self._forward_features(spatial_tensor, non_spatial_tensor)
        value_out = self.value_tail(features.global_state_embedding)
        policy_out = self._forward_policy(features, mask)
        return EquivariantAuxOutput(
            value=value_out,
            policy_logits=policy_out,
            auxiliary_outputs={
                ROUND_SCORE_TARGET_NAME: self.round_score_tail(
                    features.global_state_embedding
                ),
                FUTURE_CLEAR_TARGET_NAME: self.future_clear_tail(
                    features.all_player_column_embeddings,
                    features.global_state_embedding,
                ),
            },
        )


class EquivariantSkyNetWithRoundScoreAux(EquivariantSkyNet):
    """EquivariantSkyNet with final-round-score supervision."""

    architecture_name = EQUIVARIANT_SCORE_AUX_ARCHITECTURE_NAME

    def __init__(
        self,
        spatial_input_shape: tuple[int, ...],
        non_spatial_input_shape: tuple[int],
        value_output_shape: tuple[int],
        policy_output_shape: tuple[int],
        device: torch.device,
        embedding_dimensions: int = 16,
        global_state_embedding_dimensions: int = 32,
        num_heads: int = 4,
    ):
        super().__init__(
            spatial_input_shape=spatial_input_shape,
            non_spatial_input_shape=non_spatial_input_shape,
            value_output_shape=value_output_shape,
            policy_output_shape=policy_output_shape,
            device=device,
            embedding_dimensions=embedding_dimensions,
            global_state_embedding_dimensions=global_state_embedding_dimensions,
            num_heads=num_heads,
        )
        self.round_score_tail = NormalizedRoundScoreTail(
            input_dimensions=self.global_state_embedding_dimensions,
            players=self.players,
        )
        self.set_device(device)

    def forward(
        self,
        spatial_tensor: torch.Tensor,
        non_spatial_tensor: torch.Tensor,
        mask: torch.Tensor,
    ) -> SkyNetOutput:
        features = self._forward_features(spatial_tensor, non_spatial_tensor)
        return EquivariantAuxOutput(
            value=self.value_tail(features.global_state_embedding),
            policy_logits=self._forward_policy(features, mask),
            auxiliary_outputs={
                ROUND_SCORE_TARGET_NAME: self.round_score_tail(
                    features.global_state_embedding
                )
            },
        )


SkyNet: typing.TypeAlias = (
    SimpleSkyNet
    | EquivariantSkyNet
    | EquivariantSkyNetWithAuxiliaryHeads
    | EquivariantSkyNetWithRoundScoreAux
)

if __name__ == "__main__":
    # np.random.seed(0)
    # torch.manual_seed(0)
    # players = 2
    # game_state = sj.new(players=players)
    # players = game_state[3]
    # model = SimpleSkyNet(
    #     spatial_input_shape=(players, sj.ROW_COUNT, sj.COLUMN_COUNT, sj.FINGER_SIZE),
    #     non_spatial_input_shape=get_non_spatial_input_shape(players),
    #     value_output_shape=(players,),
    #     policy_output_shape=(sj.MASK_SIZE,),
    #     hidden_layers=[32, 32],
    # )
    # model.set_device(torch.device("mps"))
    # game_state = sj.start_round(game_state)
    # model.eval()
    # prediction = model.predict(game_state)
    # print(prediction)

    # model_path = model.save(pathlib.Path("models/test/"))
    # loaded_model = SimpleSkyNet(
    #     spatial_input_shape=model.spatial_input_shape,
    #     non_spatial_input_shape=model.non_spatial_input_shape,
    #     value_output_shape=model.value_output_shape,
    #     policy_output_shape=model.policy_output_shape,
    #     hidden_layers=[32, 32],
    # )
    # loaded_model.load_state_dict(torch.load(model_path, weights_only=True))
    # loaded_model.set_device(model.device)
    # loaded_model.eval()
    # loaded_prediction = loaded_model.predict(game_state)
    # print(loaded_prediction)
    # assert np.allclose(
    #     prediction.policy_output,
    #     loaded_prediction.policy_output,
    # )
    # assert np.allclose(
    #     prediction.value_output,
    #     loaded_prediction.value_output,
    # )
    # assert np.allclose(
    #     prediction.points_output,
    #     loaded_prediction.points_output,
    # )

    device = torch.device("mps")
    model = EquivariantSkyNet(
        spatial_input_shape=(2, sj.ROW_COUNT, sj.COLUMN_COUNT, sj.FINGER_SIZE),
        non_spatial_input_shape=get_non_spatial_input_shape(2),
        value_output_shape=(2,),
        policy_output_shape=(sj.MASK_SIZE,),
        device=device,
        embedding_dimensions=8,
        global_state_embedding_dimensions=16,
    )
    batch_size = 512
    spatial_tensor = torch.rand(
        (batch_size, 2, 3, 4, 17), dtype=torch.float32, device=device
    )
    nonspatial_tensor = torch.rand(
        (
            batch_size,
            get_non_spatial_input_shape(2)[0],
        ),
        dtype=torch.float32,
        device=device,
    )
    mask_tensor = torch.rand(
        (
            batch_size,
            sj.MASK_SIZE,
        ),
        dtype=torch.float32,
        device=device,
    )
    while True:
        with torch.inference_mode():
            model.forward(
                spatial_tensor,
                nonspatial_tensor,
                mask_tensor,
            )
