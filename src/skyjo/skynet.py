from __future__ import annotations

import dataclasses
import typing

import einops
import numpy as np
import torch
import torch.nn as nn

from . import batches, symmetry
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

ROUND_SCORE_MIN = -48.0
ROUND_SCORE_MAX = 288.0
ROUND_SCORE_RANGE = ROUND_SCORE_MAX - ROUND_SCORE_MIN
ROUND_SCORE_TARGET_NAME = "round_score"
EQUIVARIANT_ARCHITECTURE_NAME = "hierarchical_equivariant_v2"


def skyjo_to_state_value(skyjo: sj.Skyjo) -> StateValue:
    """Get the outcome of the game from the fixed perspective."""
    players = skyjo.players
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


def normalize_round_scores(
    scores: np.ndarray[tuple[int], np.float32] | np.ndarray[tuple[int], np.int16],
) -> StateValue:
    """Normalize Skyjo round scores to [0, 1] using the configured score bounds."""
    return ((scores.astype(np.float32) - ROUND_SCORE_MIN) / ROUND_SCORE_RANGE).astype(
        np.float32
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
    slot_orbits = symmetry.slot_orbits(state)
    for action_offset in (sj.MASK_FLIP, sj.MASK_REPLACE):
        for slots in slot_orbits:
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
        return cls.from_numpy_output(get_single_model_output(numpy_output, 0))

    @classmethod
    def from_numpy_output(cls, output: SkyNetNumpyOutput) -> SkyNetPrediction:
        """Convert one unbatched output into values and masked action probabilities."""
        value_numpy = output.value
        policy_logits_numpy = output.policy_logits

        # Subtract the maximum for a stable softmax over already-masked logits.
        policy_exp_logits = np.exp(
            policy_logits_numpy - np.max(policy_logits_numpy, axis=-1, keepdims=True)
        )
        policy_probabilities_numpy = policy_exp_logits / np.sum(
            policy_exp_logits, axis=-1, keepdims=True
        )

        assert len(value_numpy.shape) == len(policy_probabilities_numpy.shape) == 1, (
            "expected value_output and policy_output to be a single result and not batched results."
            f"value_output.shape: {value_numpy.shape}, policy_output.shape: {policy_probabilities_numpy.shape}"
        )
        return cls(
            value_output=value_numpy,
            policy_output=policy_probabilities_numpy,
            policy_logits=policy_logits_numpy,
            auxiliary_outputs=output.auxiliary_outputs or None,
        )

    def __str__(self) -> str:
        return f"{self.value_output}\n{self.policy_output}"

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


class SimpleOutcomeProbabilityTail(nn.Module):
    """Reusable outcome tail to predict winner probabilities over players."""

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


# MARK: SkyNets


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

        # --- Two LayerNorms (pre-LN setup) --------------------------------
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
        # LayerNorm first (pre-LN). Residual added afterwards.
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
        auxiliary_objectives: dict[str, float] | None = None,
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

        expected_policy_output_shape = (4 + 2 * self.rows * self.columns,)
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
        from . import objectives

        resolved = objectives.resolve(auxiliary_objectives)
        self.auxiliary_objectives = resolved.weights
        self.auxiliary_heads = resolved.make_heads(
            self.global_state_embedding_dimensions, self.players
        )
        self.set_device(device)

    def set_device(self, device: torch.device):
        self.device = device
        self.to(device)

    def forward(
        self,
        spatial_tensor: torch.Tensor,
        non_spatial_tensor: torch.Tensor,
        mask: torch.Tensor,
    ) -> SkyNetOutput:
        features = self._forward_features(spatial_tensor, non_spatial_tensor)
        value_out = self.value_tail(features.global_state_embedding)
        policy_out = self._forward_policy(features, mask)
        if self.auxiliary_heads:
            return EquivariantAuxOutput(
                value_out,
                policy_out,
                {
                    name: head(features.global_state_embedding)
                    for name, head in self.auxiliary_heads.items()
                },
            )
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
        batch = batches.to_tensors(batches.states_to_batch([skyjo]), device=self.device)
        output = self.forward(
            batch.spatial_inputs, batch.non_spatial_inputs, batch.action_masks
        )
        return SkyNetPrediction.from_skynet_output(output)


SkyNet: typing.TypeAlias = EquivariantSkyNet
