import numpy as np
import pytest
import torch

import skyjo as sj
from skyjo import mcts, parallel_mcts, skynet, train_utils


def test_scores_to_score_differential_value_two_players():
    scores = np.array([20, 35], dtype=np.float32)

    actual = skynet.scores_to_score_differential_value(scores)

    assert np.array_equal(actual, np.array([0, -15], dtype=np.float32))


def test_scores_to_score_differential_value_multi_player_tie_for_best():
    scores = np.array([10, 25, 10], dtype=np.float32)

    actual = skynet.scores_to_score_differential_value(scores)

    assert np.array_equal(actual, np.array([0, -15, 0], dtype=np.float32))


def test_normalize_round_scores_uses_expanded_bounds():
    scores = np.array(
        [skynet.ROUND_SCORE_MIN, 0.0, 140.0, skynet.ROUND_SCORE_MAX],
        dtype=np.float32,
    )

    actual = skynet.normalize_round_scores(scores)

    assert actual.shape == scores.shape
    assert actual[0] == pytest.approx(0.0)
    assert actual[-1] == pytest.approx(1.0)
    assert np.all(actual >= 0.0)
    assert np.all(actual <= 1.0)


def test_equivariant_skynet_value_returns_outcome_probability_simplex():
    model = skynet.EquivariantSkyNet(
        spatial_input_shape=(2, sj.ROW_COUNT, sj.COLUMN_COUNT, sj.FINGER_SIZE),
        non_spatial_input_shape=(sj.GAME_SIZE,),
        value_output_shape=(2,),
        policy_output_shape=(sj.MASK_SIZE,),
        device=torch.device("cpu"),
        embedding_dimensions=8,
        global_state_embedding_dimensions=16,
        num_heads=2,
    )
    batch_size = 4

    with torch.no_grad():
        value_output, policy_logits = model(
            torch.rand(
                batch_size,
                2,
                sj.ROW_COUNT,
                sj.COLUMN_COUNT,
                sj.FINGER_SIZE,
            ),
            torch.rand(batch_size, sj.GAME_SIZE),
            torch.ones(batch_size, sj.MASK_SIZE),
        )

    assert value_output.shape == (batch_size, 2)
    assert policy_logits.shape == (batch_size, sj.MASK_SIZE)
    assert torch.all(value_output >= 0)
    assert torch.allclose(value_output.sum(dim=1), torch.ones(batch_size), atol=1e-6)


def test_outcome_probability_tail_still_returns_probability_simplex():
    tail = skynet.SimpleOutcomeProbabilityTail(input_dimensions=3, players=4)

    output = tail(torch.randn(5, 3))

    assert output.shape == (5, 4)
    assert torch.all(output >= 0)
    assert torch.allclose(output.sum(dim=1), torch.ones(5), atol=1e-6)


def test_auxiliary_round_score_model_returns_round_score_output():
    model = skynet.EquivariantSkyNetWithRoundScoreAux(
        spatial_input_shape=(2, sj.ROW_COUNT, sj.COLUMN_COUNT, sj.FINGER_SIZE),
        non_spatial_input_shape=(sj.GAME_SIZE,),
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
            torch.rand(batch_size, sj.GAME_SIZE),
            torch.ones(batch_size, sj.MASK_SIZE),
        )

    assert output.value.shape == (batch_size, 2)
    assert output.policy_logits.shape == (batch_size, sj.MASK_SIZE)
    assert output.auxiliary_outputs[skynet.ROUND_SCORE_TARGET_NAME].shape == (
        batch_size,
        2,
    )


def test_base_loss_uses_outcome_mse():
    value_output = torch.tensor([[0.25, 0.75]], dtype=torch.float32)
    value_target = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    policy_output = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    policy_target = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

    loss, details = train_utils.base_loss(
        skynet.EquivariantOutput(
            value_output,
            policy_output,
        ),
        train_utils.TensorTrainingTargets(
            value_target,
            policy_target,
        ),
        policy_scale=0.0,
    )

    expected = torch.tensor(0.5625)
    assert loss == pytest.approx(expected.item())
    assert details["outcome_value_loss"] == pytest.approx(expected.item())


def test_mcts_terminal_node_uses_outcome_value(monkeypatch):
    expected = np.array([1.0, 0.0], dtype=np.float32)
    pre_terminal_state = sj.new(players=2)

    monkeypatch.setattr(mcts.sj, "apply_action", lambda state, action: state)
    monkeypatch.setattr(
        mcts.skynet,
        "skyjo_to_state_value",
        lambda terminal_state: expected,
    )

    node = mcts.TerminalStateNode(
        pre_terminal_state=pre_terminal_state,
        parent=None,
        action=0,
        is_random=False,
        initial_rollouts=1,
    )

    assert np.array_equal(node.state_value, expected)


def test_parallel_mcts_terminal_node_uses_outcome_value(monkeypatch):
    expected = np.array([1.0, 0.0], dtype=np.float32)
    pre_terminal_state = sj.new(players=2)

    monkeypatch.setattr(parallel_mcts.sj, "apply_action", lambda state, action: state)
    monkeypatch.setattr(
        parallel_mcts.skynet,
        "skyjo_to_state_value",
        lambda terminal_state: expected,
    )

    node = parallel_mcts.TerminalStateNode(
        pre_terminal_state=pre_terminal_state,
        parent=None,
        action=0,
        is_random=False,
        initial_outcome_realizations=1,
    )

    assert np.array_equal(node.state_value, expected)
