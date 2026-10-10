import numpy as np
import pytest
import torch

from skyjo.engine import game as sj
from skyjo.learning import losses
from skyjo.learning import observations
from skyjo.learning import skynet, targets


def test_normalize_round_scores_uses_expanded_bounds():
    scores = np.array(
        [
            targets.ROUND_SCORE_MIN,
            0.0,
            140.0,
            (targets.ROUND_SCORE_MIN + targets.ROUND_SCORE_RANGE),
        ],
        dtype=np.float32,
    )

    actual = targets.normalize_round_scores(scores)

    assert actual.shape == scores.shape
    assert actual[0] == pytest.approx(0.0)
    assert actual[-1] == pytest.approx(1.0)
    assert np.all(actual >= 0.0)
    assert np.all(actual <= 1.0)


def test_equivariant_skynet_value_returns_outcome_probability_simplex():
    model = skynet.EquivariantSkyNet(
        spatial_input_shape=(2, sj.ROW_COUNT, sj.COLUMN_COUNT, sj.FINGER_SIZE),
        non_spatial_input_shape=observations.get_non_spatial_input_shape(2),
        value_output_shape=(2,),
        policy_output_shape=(sj.MASK_SIZE,),
        device=torch.device("cpu"),
        embedding_dimensions=8,
        global_state_embedding_dimensions=16,
        num_heads=2,
    )
    batch_size = 4

    with torch.no_grad():
        output = model(
            torch.rand(
                batch_size,
                2,
                sj.ROW_COUNT,
                sj.COLUMN_COUNT,
                sj.FINGER_SIZE,
            ),
            torch.rand(batch_size, *observations.get_non_spatial_input_shape(2)),
            torch.ones(batch_size, sj.MASK_SIZE),
        )

    assert output.value.shape == (batch_size, 2)
    assert output.policy_logits.shape == (batch_size, sj.MASK_SIZE)
    assert torch.all(output.value >= 0)
    assert torch.allclose(output.value.sum(dim=1), torch.ones(batch_size), atol=1e-6)


def test_outcome_probability_tail_still_returns_probability_simplex():
    tail = skynet.SimpleOutcomeProbabilityTail(input_dimensions=3, players=4)

    output = tail(torch.randn(5, 3))

    assert output.shape == (5, 4)
    assert torch.all(output >= 0)
    assert torch.allclose(output.sum(dim=1), torch.ones(5), atol=1e-6)


def test_auxiliary_round_score_model_returns_round_score_output():
    model = skynet.EquivariantSkyNet(
        auxiliary_objectives={"round_score": 0.1},
        spatial_input_shape=(2, sj.ROW_COUNT, sj.COLUMN_COUNT, sj.FINGER_SIZE),
        non_spatial_input_shape=observations.get_non_spatial_input_shape(2),
        value_output_shape=(2,),
        policy_output_shape=(sj.MASK_SIZE,),
        device=torch.device("cpu"),
        embedding_dimensions=8,
        global_state_embedding_dimensions=16,
        num_heads=2,
    )
    batch_size = 4

    with torch.no_grad():
        output = model(
            torch.rand(
                batch_size,
                2,
                sj.ROW_COUNT,
                sj.COLUMN_COUNT,
                sj.FINGER_SIZE,
            ),
            torch.rand(batch_size, *observations.get_non_spatial_input_shape(2)),
            torch.ones(batch_size, sj.MASK_SIZE),
        )

    assert output.value.shape == (batch_size, 2)
    assert output.policy_logits.shape == (batch_size, sj.MASK_SIZE)
    assert output.auxiliary_outputs["round_score"].shape == (
        batch_size,
        2,
    )


def test_base_loss_uses_outcome_mse():
    value_output = torch.tensor([[0.25, 0.75]], dtype=torch.float32)
    value_target = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    policy_output = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    policy_target = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

    loss, details = losses.base_loss(
        skynet.ModelOutput(
            value_output,
            policy_output,
        ),
        {"value": value_target, "policy": policy_target},
        policy_scale=0.0,
    )

    expected = torch.tensor(0.5625)
    assert loss == pytest.approx(expected.item())
    assert details["outcome_value_loss"] == pytest.approx(expected.item())
