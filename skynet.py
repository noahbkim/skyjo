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


# MARK: State Value


StateValue: typing.TypeAlias = np.ndarray[tuple[int], np.float32]
"""A vector representing the value of a Skyjo game for each player.

IMPORTANT: This is from a fixed perspective and not relative to the current player
    i.e. the first element is always the value of player 0, the second is for player 1, etc.

This can also be used to represent the outcome of the game where all entries are 0 except for the winner.
"""


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
    return np.roll(value_output, shift=curr_player)


def get_spatial_state_numpy(
    skyjo: sj.Skyjo,
) -> np.ndarray[tuple[int], np.float32]:
    return sj.get_table(skyjo).astype(np.float32)


def get_non_spatial_state_numpy(
    skyjo: sj.Skyjo,
) -> np.ndarray[tuple[int], np.float32]:
    return sj.get_game(skyjo).astype(np.float32)


# MARK: Model Output


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
    safe_denominator = torch.where(
        total_valid_action_probabilities == 0, 1.0, total_valid_action_probabilities
    )
    renormalized_valid_action_probabilities = (
        valid_action_probabilities / safe_denominator
    )
    # Assign uniform probability where total probability is 0
    renormalized_valid_action_probabilities = torch.where(
        total_valid_action_probabilities == 0,
        1 / num_valid_actions,
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
    points_output: np.ndarray[tuple[int], np.float32]
    policy_output: np.ndarray[tuple[int], np.float32]
    policy_logits: np.ndarray[tuple[int], np.float32] | None = None

    @classmethod
    def from_skynet_output(
        cls,
        output: SkyNetOutput,
    ) -> SkyNetPrediction:
        numpy_output = output_to_numpy(output)
        value_numpy, points_numpy, policy_logits_numpy = get_single_model_output(
            numpy_output, 0
        )

        # Convert masked policy logits to probabilities
        # The logits from PolicyTail.forward are already masked (large negative numbers for invalid actions)
        # A standard softmax will handle these correctly, assigning near-zero probability to masked actions.
        exp_logits = np.exp(
            policy_logits_numpy - np.max(policy_logits_numpy, axis=-1, keepdims=True)
        )  # for numerical stability
        policy_probabilities_numpy = exp_logits / np.sum(
            exp_logits, axis=-1, keepdims=True
        )

        assert (
            len(value_numpy.shape)
            == len(points_numpy.shape)
            == len(policy_probabilities_numpy.shape)
            == 1
        ), (
            "expected value_output, points_output, and policy_output to be a single result and not batched results."
            f"value_output.shape: {value_numpy.shape}, points_output.shape: {points_numpy.shape}, policy_output.shape: {policy_probabilities_numpy.shape}"
        )
        return SkyNetPrediction(
            value_output=value_numpy,
            points_output=points_numpy,
            policy_output=policy_probabilities_numpy,
            policy_logits=policy_logits_numpy,
        )

    def __str__(self) -> str:
        return f"{self.value_output}\n{self.points_output}\n{self.policy_output}"

    def mask_and_renormalize(self, valid_actions_mask: np.ndarray[tuple[int], np.int8]):
        self.policy_output = mask_and_renormalize_policy_probabilities(
            self.policy_output, valid_actions_mask
        )

    def to_output(self) -> SkyNetOutput:
        return (
            torch.tensor(np.expand_dims(self.value_output, 0), dtype=torch.float32),
            torch.tensor(np.expand_dims(self.points_output, 0), dtype=torch.float32),
            torch.tensor(np.expand_dims(self.policy_logits, 0), dtype=torch.float32),
        )


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


class SimpleValueTail(nn.Module):
    """Simple value tail to predict win probability of players.

    Tranforms input using an MLP with final sigmoid activation."""

    def __init__(self, input_dimensions: int, players: int):
        super(SimpleValueTail, self).__init__()
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

        # Currently the flip and replace logit transformation is down
        # with just a multi-layer perceptron (MLP). But, this could be swapped
        # for an self-attention block -> MLP or something else as long as the
        # transformation process preserves equivariance.
        self.positional_logits_mlp = nn.Sequential(
            # nn.Linear(
            #     in_features=self.embedding_dimensions
            #     # + self.column_embedding_dimensions
            #     + self.global_state_embedding_dimensions,
            #     out_features=self.embedding_dimensions
            #     # + self.column_embedding_dimensions
            #     + self.global_state_embedding_dimensions,
            # ),
            # nn.ReLU(inplace=True),
            nn.Linear(
                in_features=self.embedding_dimensions
                # + self.column_embedding_dimensions
                + self.global_state_embedding_dimensions,
                out_features=2,
            ),
            # nn.Linear(
            #     in_features=self.card_embedding_dimensions
            #     + self.column_embedding_dimensions
            #     + self.global_state_embedding_dimensions,
            #     out_features=self.card_embedding_dimensions,
            # ),
            # nn.ReLU(inplace=True),
            # nn.Linear(
            #     in_features=self.card_embedding_dimensions,
            #     out_features=2,
            # ),
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
        global_state_tensor: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        expanded_global_state_tensor = einops.repeat(
            global_state_tensor,
            "b f -> b (h w) f",
            h=self.rows,
            w=self.columns,
        )
        # expanded_column_summaries = einops.repeat(
        #     column_summaries,
        #     "b w f -> b (h w) f",
        #     h=self.rows,
        # )
        expanded_global_state_tensor = torch.cat(
            (
                flattened_card_embeddings,
                # expanded_column_summaries,
                expanded_global_state_tensor,
            ),
            dim=2,
        )

        positional_logits = self.positional_logits_mlp(expanded_global_state_tensor)
        flip_logits = positional_logits[:, :, 0]
        replace_logits = positional_logits[:, :, 1]
        non_positional_logits = self.non_positional_logits_mlp(global_state_tensor)
        logits = torch.cat((non_positional_logits, flip_logits, replace_logits), dim=1)
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), -1e10)
        return logits


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
        self.points_output_shape = value_output_shape
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
        self.value_tail = SimpleValueTail(hidden_layers[-1], value_output_shape[0])
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
        return value_out, torch.zeros_like(value_out), policy_out

    def predict(self, skyjo: sj.Skyjo) -> SkyNetPrediction:
        spatial_tensor = einops.rearrange(
            torch.tensor(
                sj.get_spatial_input(skyjo), dtype=torch.float32, device=self.device
            ),
            "p h w c -> 1 p h w c",
        ).contiguous()
        non_spatial_tensor = einops.rearrange(
            torch.tensor(
                sj.get_non_spatial_input(skyjo), dtype=torch.float32, device=self.device
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

    def save(self, dir: pathlib.Path) -> pathlib.Path:
        curr_utc_dt = datetime.datetime.now(tz=datetime.timezone.utc)
        model_path = dir / f"model_{curr_utc_dt.strftime('%Y%m%d_%H%M%S')}.pth"
        torch.save(
            self.state_dict(),
            model_path,
        )
        return model_path


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


class EquivariantSkyNet(nn.Module):
    """Equivariant SkyNet Model.

    The key property is that it is "equivariant" in the sense that swapping cards
    within a column will exactly swap the output logits in the flip and replace
    actions. Similar swapping whole columns will swap the logits corresponding
    to those columns. This was done be design since in the Skyjo game the order
    of the cards within the column do not matter in the evaluation of the game
    and on the underlying policy.

    This property was achieved by pooling the cards within a column and then pooling
    those columns into a board summary. The board summaries of the players,
    along with the non-spatial state, are then transformed into a global state
    representation. This representation is then passed to with each card slot and
    identically transformed to produce the logits for the flip and replace actions.
    This way the logits for the flip and replace actions are equivariant since
    the pooling and transformation process is agnostic of position. And the logit
    transformation is identical for each card slot.
    """

    def __init__(
        self,
        spatial_input_shape: tuple[int, ...],  # (players, )
        non_spatial_input_shape: tuple[int],  # (sj.GAME_SIZE,)
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
        self.points_output_shape = value_output_shape
        self.policy_output_shape = policy_output_shape
        self.embedding_dimensions = embedding_dimensions
        self.global_state_embedding_dimensions = global_state_embedding_dimensions
        self.num_heads = num_heads
        self.players = self.spatial_input_shape[0]
        self.rows = self.spatial_input_shape[1]
        self.columns = self.spatial_input_shape[2]
        self.card_types = self.spatial_input_shape[3]

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

        self.column_attention = TransformerBlock(
            embed_dim=self.embedding_dimensions,
            num_heads=self.num_heads,
            mlp_ratio=None,
            dropout=0.0,
        )
        # self.board_summary_token = nn.Parameter(
        #     torch.randn(1, 1, self.embedding_dimensions)
        # )
        # self.board_attention = TransformerBlock(
        #     embed_dim=self.embedding_dimensions,
        #     num_heads=2,
        #     mlp_ratio=0.0,
        #     dropout=0.0,
        # )

        # Spatial Layers
        # self.column_summarizer = nn.Sequential(
        #     nn.Linear(
        #         in_features=self.card_embedding_dimensions,
        #         out_features=self.column_embedding_dimensions,
        #     ),
        #     nn.ReLU(inplace=True),
        # )

        # self.board_summarizer = nn.Sequential(
        #     nn.Linear(
        #         in_features=self.column_embedding_dimensions,
        #         out_features=self.board_embedding_dimensions,
        #     ),
        #     nn.ReLU(inplace=True),
        # )

        self.global_state_embedder = nn.Sequential(
            nn.Linear(
                in_features=self.embedding_dimensions * (self.players + 1),
                # + self.board_embedding_dimensions * self.players,
                # + self.embedding_dimensions * self.players,
                out_features=self.global_state_embedding_dimensions,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                in_features=self.global_state_embedding_dimensions,
                out_features=self.global_state_embedding_dimensions,
            ),
        )

        # Tails
        self.value_tail = SimpleValueTail(
            input_dimensions=self.global_state_embedding_dimensions,
            players=self.players,
        )
        self.policy_tail = EquivariantPolicyLogitTail(
            embedding_dimensions=self.embedding_dimensions,
            global_state_embedding_dimensions=self.global_state_embedding_dimensions,
        )
        self.set_device(device)

    def set_device(self, device: torch.device):
        self.device = device
        self.to(device)

    def save(self, dir: pathlib.Path) -> pathlib.Path:
        curr_utc_dt = datetime.datetime.now(tz=datetime.timezone.utc)
        model_path = dir / f"model_{curr_utc_dt.strftime('%Y%m%d_%H%M%S')}.pth"
        torch.save(
            self.state_dict(),
            model_path,
        )
        return model_path

    def forward(
        self,
        spatial_tensor: torch.Tensor,
        non_spatial_tensor: torch.Tensor,
        mask: torch.Tensor,
    ):
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
        attended_cards = self.column_attention(
            with_cls_tokens,
        )

        column_summaries = einops.rearrange(
            attended_cards[:, 0, :],
            "(b p w) f -> (b p) w f",
            p=self.players,
            w=self.columns,
        )
        attended_cards = attended_cards[:, 2:, :]
        # repeated_board_summary_tokens = einops.repeat(
        #     self.board_summary_token,
        #     "1 1 f -> bp 1 f",
        #     bp=column_summaries.shape[0],
        # )
        # repeated_non_spatial_embeddings = einops.repeat(
        #     non_spatial_embeddings,
        #     "b f -> (b p) 1 f",
        #     p=self.players,
        # )
        # with_board_summary_tokens = torch.cat(
        #     (
        #         repeated_board_summary_tokens,
        #         repeated_non_spatial_embeddings,
        #         column_summaries,
        #     ),
        #     dim=1,
        # )
        # attended_columns = self.board_attention(
        #     with_board_summary_tokens,
        # )
        # board_summaries = attended_columns[:, 0, :]
        # attended_columns = attended_columns[:, 2:, :]

        board_summaries = einops.reduce(
            column_summaries,
            "(b p) w f -> b (p f)",
            reduction="sum",
            w=self.columns,
            p=self.players,
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
        value_out = self.value_tail(global_state_embedding)
        policy_out = self.policy_tail(
            einops.rearrange(
                attended_cards,
                "(b p w) h f -> b p (h w) f",
                p=self.players,
                w=self.columns,
            )[:, 0, :].contiguous(),
            global_state_embedding,
            mask,
        )
        return value_out, torch.zeros_like(value_out), policy_out

    def predict(self, skyjo: sj.Skyjo) -> SkyNetPrediction:
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
        mask_tensor = torch.tensor(
            sj.actions(skyjo), dtype=torch.float32, device=self.device
        )
        output = self.forward(spatial_tensor, non_spatial_tensor, mask_tensor)
        return SkyNetPrediction.from_skynet_output(output)


SkyNet: typing.TypeAlias = SimpleSkyNet | EquivariantSkyNet

if __name__ == "__main__":
    # np.random.seed(0)
    # torch.manual_seed(0)
    # players = 2
    # game_state = sj.new(players=players)
    # players = game_state[3]
    # model = SimpleSkyNet(
    #     spatial_input_shape=(players, sj.ROW_COUNT, sj.COLUMN_COUNT, sj.FINGER_SIZE),
    #     non_spatial_input_shape=(sj.GAME_SIZE,),
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
        non_spatial_input_shape=(sj.GAME_SIZE,),
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
            sj.GAME_SIZE,
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
        with torch.no_grad():
            model.forward(
                spatial_tensor,
                nonspatial_tensor,
                mask_tensor,
            )
