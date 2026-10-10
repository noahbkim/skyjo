from __future__ import annotations

import dataclasses
import typing

import einops
import torch
import torch.nn as nn


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


EQUIVARIANT_ARCHITECTURE_NAME = "hierarchical_equivariant_v2"


@dataclasses.dataclass(slots=True)
class ModelOutput:
    """Tensor predictions in current-player order; inference owns conversion."""

    value: torch.Tensor
    policy_logits: torch.Tensor
    auxiliary_outputs: dict[str, torch.Tensor] = dataclasses.field(default_factory=dict)


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
        from skyjo.learning import objectives
        from .auxiliary_heads import make_heads

        resolved = objectives.resolve(auxiliary_objectives)
        self.auxiliary_objectives = resolved.weights
        self.auxiliary_heads = make_heads(
            resolved, self.global_state_embedding_dimensions, self.players
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
    ) -> ModelOutput:
        features = self._forward_features(spatial_tensor, non_spatial_tensor)
        value_out = self.value_tail(features.global_state_embedding)
        policy_out = self._forward_policy(features, mask)
        return ModelOutput(
            value_out,
            policy_out,
            {
                name: head(features.global_state_embedding)
                for name, head in self.auxiliary_heads.items()
            },
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


SkyNet: typing.TypeAlias = EquivariantSkyNet
