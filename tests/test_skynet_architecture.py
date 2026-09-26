from __future__ import annotations

import numpy as np
import pytest
import torch

from skyjo import game as sj
from skyjo import skynet


def make_model(players: int = 2) -> skynet.EquivariantSkyNet:
    return skynet.EquivariantSkyNet(
        spatial_input_shape=(
            players,
            sj.ROW_COUNT,
            sj.COLUMN_COUNT,
            sj.FINGER_SIZE,
        ),
        non_spatial_input_shape=(sj.GAME_SIZE,),
        value_output_shape=(players,),
        policy_output_shape=(sj.MASK_SIZE,),
        device=torch.device("cpu"),
        embedding_dimensions=8,
        global_state_embedding_dimensions=16,
        num_heads=2,
    )


def test_board_permutations_preserve_value_and_permute_masked_policy() -> None:
    torch.manual_seed(7)
    model = make_model(players=3).eval()
    spatial = torch.randn(1, 3, sj.ROW_COUNT, sj.COLUMN_COUNT, sj.FINGER_SIZE)
    non_spatial = torch.randn(1, sj.GAME_SIZE)
    mask = torch.tensor(
        [[1, 0, 1, 1, *([1, 0, 1, 1] * 3), *([0, 1, 1, 1] * 3)]],
        dtype=torch.float32,
    )
    column_permutations = (
        torch.tensor([2, 0, 3, 1]),
        torch.tensor([1, 3, 0, 2]),
        torch.tensor([3, 2, 1, 0]),
    )
    row_permutations = (
        (
            torch.tensor([2, 0, 1]),
            torch.tensor([1, 2, 0]),
            torch.tensor([0, 2, 1]),
            torch.tensor([2, 1, 0]),
        ),
        (
            torch.tensor([1, 0, 2]),
            torch.tensor([2, 0, 1]),
            torch.tensor([1, 2, 0]),
            torch.tensor([0, 1, 2]),
        ),
        (
            torch.tensor([0, 2, 1]),
            torch.tensor([2, 1, 0]),
            torch.tensor([1, 0, 2]),
            torch.tensor([2, 0, 1]),
        ),
    )
    permuted_spatial = torch.empty_like(spatial)
    for player, column_permutation in enumerate(column_permutations):
        for new_column, old_column in enumerate(column_permutation.tolist()):
            permuted_spatial[0, player, :, new_column] = spatial[
                0,
                player,
                row_permutations[player][new_column],
                old_column,
            ]

    active_slot_indices = []
    for new_row in range(sj.ROW_COUNT):
        for new_column, old_column in enumerate(column_permutations[0].tolist()):
            old_row = row_permutations[0][new_column][new_row].item()
            active_slot_indices.append(old_row * sj.COLUMN_COUNT + old_column)
    permuted_mask = torch.cat(
        (
            mask[:, : sj.MASK_FLIP],
            mask[:, sj.MASK_FLIP : sj.MASK_REPLACE][:, active_slot_indices],
            mask[:, sj.MASK_REPLACE :][:, active_slot_indices],
        ),
        dim=1,
    )

    with torch.inference_mode():
        output = model(spatial, non_spatial, mask)
        permuted_output = model(permuted_spatial, non_spatial, permuted_mask)
    expected_policy = torch.cat(
        (
            output.policy_logits[:, : sj.MASK_FLIP],
            output.policy_logits[:, sj.MASK_FLIP : sj.MASK_REPLACE][
                :, active_slot_indices
            ],
            output.policy_logits[:, sj.MASK_REPLACE :][:, active_slot_indices],
        ),
        dim=1,
    )

    assert torch.allclose(permuted_output.value, output.value, atol=1e-6)
    assert torch.allclose(
        permuted_output.policy_logits,
        expected_policy,
        atol=1e-6,
    )


def test_positional_ranking_uses_other_columns_and_opponents() -> None:
    torch.manual_seed(7)
    model = make_model()
    spatial = torch.randn(
        1,
        2,
        sj.ROW_COUNT,
        sj.COLUMN_COUNT,
        sj.FINGER_SIZE,
        requires_grad=True,
    )
    output = model(
        spatial,
        torch.randn(1, sj.GAME_SIZE),
        torch.ones(1, sj.MASK_SIZE),
    )
    replace_logit_difference = (
        output.policy_logits[0, sj.MASK_REPLACE]
        - output.policy_logits[0, sj.MASK_REPLACE + 1]
    )
    (spatial_gradient,) = torch.autograd.grad(replace_logit_difference, spatial)

    assert spatial_gradient[0, 0, :, 3, :].abs().sum() > 0
    assert spatial_gradient[0, 1].abs().sum() > 0


def test_relative_player_order_affects_global_and_value_outputs() -> None:
    torch.manual_seed(11)
    model = make_model(players=3).eval()
    spatial = torch.randn(1, 3, sj.ROW_COUNT, sj.COLUMN_COUNT, sj.FINGER_SIZE)
    swapped_spatial = spatial[:, [0, 2, 1]].clone()
    non_spatial = torch.randn(1, sj.GAME_SIZE)
    mask = torch.ones(1, sj.MASK_SIZE)

    with torch.inference_mode():
        output = model(spatial, non_spatial, mask)
        swapped_output = model(swapped_spatial, non_spatial, mask)

    assert not torch.allclose(
        swapped_output.policy_logits[:, : sj.MASK_FLIP],
        output.policy_logits[:, : sj.MASK_FLIP],
    )
    assert not torch.allclose(
        swapped_output.value,
        output.value[:, [0, 2, 1]],
    )


def test_three_player_outputs_masking_and_backward_contract() -> None:
    torch.manual_seed(13)
    model = make_model(players=3)
    mask = torch.ones(2, sj.MASK_SIZE)
    mask[:, [5, 18]] = 0
    output = model(
        torch.randn(2, 3, sj.ROW_COUNT, sj.COLUMN_COUNT, sj.FINGER_SIZE),
        torch.randn(2, sj.GAME_SIZE),
        mask,
    )

    assert output.value.shape == (2, 3)
    assert output.policy_logits.shape == (2, sj.MASK_SIZE)
    assert torch.allclose(output.value.sum(dim=-1), torch.ones(2), atol=1e-6)
    assert torch.all(output.policy_logits[:, [5, 18]] == -1e10)

    loss = output.value.square().sum() + output.policy_logits[mask.bool()].sum()
    loss.backward()
    assert model.card_embedder.weight.grad is not None
    assert model.global_state_embedder[0].weight.grad is not None
    assert model.policy_tail.positional_logits_mlp[0].weight.grad is not None


def test_default_model_stays_within_small_parameter_budget() -> None:
    model = skynet.EquivariantSkyNet(
        spatial_input_shape=(2, 3, 4, 17),
        non_spatial_input_shape=(50,),
        value_output_shape=(2,),
        policy_output_shape=(28,),
        device=torch.device("cpu"),
    )

    assert sum(parameter.numel() for parameter in model.parameters()) < 10_000


def test_predict_returns_probabilities_only_for_valid_actions() -> None:
    model = make_model()
    game_state = sj.new(players=2, top=sj.CARD_0)

    prediction = model.predict(game_state)
    valid_actions = sj.actions(game_state).astype(bool)

    assert prediction.value_output.shape == (2,)
    assert prediction.policy_output.shape == (sj.MASK_SIZE,)
    assert prediction.policy_output.sum() == pytest.approx(1.0)
    assert np.all(prediction.policy_output[~valid_actions] == 0)


@pytest.mark.parametrize(
    ("policy_output_shape", "embedding_dimensions", "global_dimensions", "heads", "message"),
    [
        ((27,), 8, 16, 2, "policy_output_shape"),
        ((28,), 8, 16, 3, "embedding_dimensions"),
        ((28,), 8, 18, 4, "global_state_embedding_dimensions"),
    ],
)
def test_model_rejects_incompatible_architecture_shapes(
    policy_output_shape: tuple[int],
    embedding_dimensions: int,
    global_dimensions: int,
    heads: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        skynet.EquivariantSkyNet(
            spatial_input_shape=(2, 3, 4, 17),
            non_spatial_input_shape=(50,),
            value_output_shape=(2,),
            policy_output_shape=policy_output_shape,
            device=torch.device("cpu"),
            embedding_dimensions=embedding_dimensions,
            global_state_embedding_dimensions=global_dimensions,
            num_heads=heads,
        )
