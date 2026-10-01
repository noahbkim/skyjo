from __future__ import annotations

import pathlib

import numpy as np
import pytest
import torch

import skyjo as sj
from skyjo import (
    buffer,
    checkpoint,
    mcts,
    parallel_mcts,
    play,
    predictor,
    skynet,
    train_utils,
)


def make_aux_model() -> skynet.EquivariantSkyNetWithAuxiliaryHeads:
    return skynet.EquivariantSkyNetWithAuxiliaryHeads(
        spatial_input_shape=(2, sj.ROW_COUNT, sj.COLUMN_COUNT, sj.FINGER_SIZE),
        non_spatial_input_shape=skynet.get_non_spatial_input_shape(2),
        value_output_shape=(2,),
        policy_output_shape=(sj.MASK_SIZE,),
        device=torch.device("cpu"),
        embedding_dimensions=8,
        global_state_embedding_dimensions=16,
        num_heads=2,
    )


def test_future_clear_target_rotates_players_and_masks_existing_columns() -> None:
    state = list(sj.new(players=2, top=0))
    table = state[1].copy()
    table[0, :, 1, :] = 0
    table[0, :, 1, sj.FINGER_CLEARED] = 1
    state[1] = table
    state[4] = 1
    fixed_final = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.5, 0.0]],
        dtype=np.float32,
    )

    target = play.future_clear_target_for_state(tuple(state), fixed_final)

    assert np.array_equal(
        target,
        np.array(
            [[0.0, -1.0, 0.5, 0.0], [1.0, 0.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
    )


def test_auxiliary_loss_masks_existing_clears_and_weights_positives() -> None:
    output = skynet.EquivariantAuxOutput(
        value=torch.tensor([[1.0, 0.0]]),
        policy_logits=torch.tensor([[0.0, 0.0]]),
        auxiliary_outputs={
            skynet.ROUND_SCORE_TARGET_NAME: torch.tensor([[0.2, 0.3]]),
            skynet.FUTURE_CLEAR_TARGET_NAME: torch.tensor(
                [[[100.0, 0.0], [0.0, 0.0]]]
            ),
        },
    )
    targets = train_utils.TensorTrainingTargets(
        value=torch.tensor([[1.0, 0.0]]),
        policy=torch.tensor([[1.0, 0.0]]),
        target_tensors={
            train_utils.VALUE_TARGET_NAME: torch.tensor([[1.0, 0.0]]),
            train_utils.POLICY_TARGET_NAME: torch.tensor([[1.0, 0.0]]),
            train_utils.FUTURE_CLEAR_TARGET_NAME: torch.tensor(
                [[[-1.0, 1.0], [0.0, 0.0]]]
            ),
        },
    )

    loss, details = train_utils.outcome_policy_auxiliary_loss(
        output,
        targets,
        value_scale=0.0,
        policy_scale=0.0,
        round_score_scale=0.0,
        future_clear_scale=1.0,
        clear_positive_weight=10.0,
    )

    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.zeros(3),
        torch.tensor([1.0, 0.0, 0.0]),
        pos_weight=torch.tensor(10.0),
    )
    assert loss.item() == pytest.approx(expected.item())
    assert details["future_clear_loss"] == pytest.approx(expected.item())


def test_zero_auxiliary_weights_are_exactly_the_base_loss() -> None:
    output = skynet.EquivariantOutput(
        torch.tensor([[0.25, 0.75]]),
        torch.tensor([[0.0, 0.0]]),
    )
    targets = train_utils.TensorTrainingTargets(
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[1.0, 0.0]]),
    )

    base, base_details = train_utils.base_loss(output, targets)
    actual, actual_details = train_utils.outcome_policy_auxiliary_loss(
        output,
        targets,
        round_score_scale=0.0,
        future_clear_scale=0.0,
    )

    assert torch.equal(actual, base)
    assert actual_details == base_details


def test_auxiliary_heads_preserve_column_symmetry() -> None:
    torch.manual_seed(4)
    model = make_aux_model().eval()
    spatial = torch.rand(
        2, 2, sj.ROW_COUNT, sj.COLUMN_COUNT, sj.FINGER_SIZE
    )
    non_spatial = torch.rand(2, *skynet.get_non_spatial_input_shape(2))
    mask = torch.ones(2, sj.MASK_SIZE)
    permutation = torch.tensor([2, 0, 3, 1])

    with torch.no_grad():
        output = model(spatial, non_spatial, mask)
        permuted = model(spatial[:, :, :, permutation], non_spatial, mask)

    assert output.auxiliary_outputs[skynet.ROUND_SCORE_TARGET_NAME].shape == (2, 2)
    assert output.auxiliary_outputs[skynet.FUTURE_CLEAR_TARGET_NAME].shape == (
        2,
        2,
        sj.COLUMN_COUNT,
    )
    assert torch.allclose(
        output.auxiliary_outputs[skynet.ROUND_SCORE_TARGET_NAME],
        permuted.auxiliary_outputs[skynet.ROUND_SCORE_TARGET_NAME],
        atol=1e-6,
    )
    assert torch.allclose(
        output.auxiliary_outputs[skynet.FUTURE_CLEAR_TARGET_NAME][:, :, permutation],
        permuted.auxiliary_outputs[skynet.FUTURE_CLEAR_TARGET_NAME],
        atol=1e-6,
    )


def test_local_predictor_preserves_score_output_for_search() -> None:
    client = predictor.LocalPredictorClient(make_aux_model(), max_batch_size=4)
    state = sj.new(players=2, top=0)

    prediction_id = client.put(state)
    client.send()
    returned_id, prediction = client.get()

    assert returned_id == prediction_id
    assert prediction.round_score_output is not None
    expected = prediction.value_output + 0.05 * (1.0 - prediction.round_score_output)
    assert np.allclose(prediction.search_value(0.05), expected)


def test_mcts_node_uses_score_utility_but_zero_weight_is_win_only() -> None:
    state = sj.new(players=2, top=0)
    prediction = skynet.SkyNetPrediction(
        value_output=np.array([0.6, 0.4], dtype=np.float32),
        policy_output=np.ones(sj.MASK_SIZE, dtype=np.float32) / sj.MASK_SIZE,
        auxiliary_outputs={
            skynet.ROUND_SCORE_TARGET_NAME: np.array([0.2, 0.8], dtype=np.float32)
        },
    )
    score_node = mcts.DecisionStateNode(
        state,
        parent=None,
        action=None,
        model_prediction=prediction,
        score_utility_weight=0.05,
    )
    baseline_node = mcts.DecisionStateNode(
        state,
        parent=None,
        action=None,
        model_prediction=prediction,
        score_utility_weight=0.0,
    )

    assert score_node.model_value_for_current_player == pytest.approx(0.64)
    assert np.allclose(score_node.state_value, [0.64, 0.41])
    assert np.array_equal(baseline_node.state_value, prediction.value_output)


def test_batched_mcts_forwards_score_utility_configuration() -> None:
    client = predictor.LocalPredictorClient(make_aux_model(), max_batch_size=4)
    state = sj.new(players=2, top=0)

    root = parallel_mcts.run_mcts(
        state,
        client,
        iterations=0,
        score_utility_weight=0.05,
    )

    assert root.score_utility_weight == 0.05
    assert root.model_prediction is not None
    assert np.allclose(
        root.state_value,
        skynet.to_state_value(
            root.model_prediction.search_value(0.05),
            sj.get_player(state),
        ),
    )


def test_predictor_output_queue_round_trips_score_predictions() -> None:
    queue = predictor.PredictorOutputQueue(queue_id=0, max_batch_size=2)
    ids = torch.tensor([4, 5])
    values = torch.tensor([[0.6, 0.4], [0.2, 0.8]])
    policies = torch.zeros(2, sj.MASK_SIZE)
    scores = torch.tensor([[0.1, 0.7], [0.3, 0.4]])

    queue.put(ids, values, policies, scores, batch_size=2)
    actual_ids, actual_values, actual_policies, actual_scores, batch_size = queue.get()

    assert batch_size == 2
    assert torch.equal(actual_ids, ids)
    assert torch.equal(actual_values, values)
    assert torch.equal(actual_policies, policies)
    assert torch.equal(actual_scores, scores)


def test_auxiliary_warm_start_loads_backbone_only(tmp_path: pathlib.Path) -> None:
    baseline = skynet.EquivariantSkyNet(
        spatial_input_shape=(2, sj.ROW_COUNT, sj.COLUMN_COUNT, sj.FINGER_SIZE),
        non_spatial_input_shape=skynet.get_non_spatial_input_shape(2),
        value_output_shape=(2,),
        policy_output_shape=(sj.MASK_SIZE,),
        device=torch.device("cpu"),
        embedding_dimensions=8,
        global_state_embedding_dimensions=16,
        num_heads=2,
    )
    path = checkpoint.save_checkpoint(
        tmp_path / "baseline.pth",
        model=baseline,
        optimizer=None,
    )
    auxiliary = make_aux_model()

    checkpoint.load_auxiliary_warm_start(path, model=auxiliary)

    auxiliary_state = auxiliary.state_dict()
    for name, value in baseline.state_dict().items():
        assert torch.equal(auxiliary_state[name], value)


def test_experiment_presets_and_auxiliary_target_schema() -> None:
    combined = train_utils.get_auxiliary_experiment_preset("combined")
    specs = buffer.auxiliary_target_specs(2, (sj.MASK_SIZE,))

    assert combined.round_score_scale == 0.1
    assert combined.future_clear_scale == 0.1
    assert combined.score_utility_weight == 0.05
    assert specs[-1] == buffer.TargetShapeSpec("future_clear", (2, sj.COLUMN_COUNT))


def test_replay_dataset_round_trips_auxiliary_targets(tmp_path: pathlib.Path) -> None:
    replay = buffer.ReplayBuffer(
        max_size=4,
        spatial_input_shape=(2, sj.ROW_COUNT, sj.COLUMN_COUNT, sj.FINGER_SIZE),
        non_spatial_input_shape=skynet.get_non_spatial_input_shape(2),
        action_mask_shape=(sj.MASK_SIZE,),
        target_specs=buffer.auxiliary_target_specs(2, (sj.MASK_SIZE,)),
    )
    state = sj.new(players=2, top=0)
    action_mask = sj.actions(state).astype(np.float32)
    future_clear = np.array(
        [[-1.0, 0.0, 0.5, 1.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    replay.add_game_data(
        [
            play.GameDataPoint(
                state,
                None,
                {
                    train_utils.VALUE_TARGET_NAME: np.array([1.0, 0.0]),
                    train_utils.POLICY_TARGET_NAME: action_mask / action_mask.sum(),
                    train_utils.ROUND_SCORE_TARGET_NAME: np.array([0.2, 0.6]),
                    train_utils.FUTURE_CLEAR_TARGET_NAME: future_clear,
                },
            )
        ],
        game_index=3,
        play_seed=4,
        target_seed=5,
    )

    replay.save(tmp_path / "dataset")
    loaded = buffer.ReplayBuffer.load(tmp_path / "dataset")

    assert np.array_equal(
        loaded.ordered_batch().targets[train_utils.FUTURE_CLEAR_TARGET_NAME][0],
        future_clear,
    )
