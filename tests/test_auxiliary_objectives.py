import pickle
import queue
import random
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

import skyjo as sj
from skyjo import buffer, factory, objectives, play, predictor, skynet, targets, train, train_utils


ABLATIONS = [{}, {"round_raw_score": 0.1}, {"round_doubled": 0.1},
             {"round_raw_score": 0.1, "round_doubled": 0.1}]


def terminal(scores, turn=0, stalled=()):
    state = list(sj.new(players=len(scores), rng=random.Random(0)))
    state[4], state[6] = turn, 0
    state[0][sj.GAME_LAST_REVEALED_TURNS:sj.GAME_LAST_REVEALED_TURNS + len(scores)] = turn
    for player in stalled:
        state[0][sj.GAME_LAST_REVEALED_TURNS + player] = 0
    state[1].fill(0)
    state[1][:len(scores), :, :, sj.FINGER_CLEARED] = 1
    for player, score in enumerate(scores):
        # Three visible cards in a column, deliberately not a matching triple.
        values = [score, 0, 0] if score <= 12 else [12, score - 12, 0]
        state[1][player, :, 0, :] = 0
        for row, value in enumerate(values):
            state[1][player, row, 0, value + 2] = 1
    return tuple(state)


@pytest.mark.parametrize("scores,turn,stalled,flags,final", [
    ([10, 20], 0, (), [False, False], [10, 20]),
    ([20, 10], 0, (), [True, False], [40, 10]),
    ([10, 10], 0, (), [True, False], [20, 10]),
    ([0, 0], 0, (), [True, False], [0, 0]),
    ([-1, -2], 0, (), [True, False], [-2, -2]),
    ([0, 10, 20], 63, (0, 1), [True, True, False], [0, 20, 20]),
])
def test_scoring_exposes_rule_flags_even_when_score_does_not_change(scores, turn, stalled, flags, final):
    state = terminal(scores, turn, stalled)
    raw, doubled = sj.get_round_score_components(state)
    np.testing.assert_array_equal(raw, scores)
    np.testing.assert_array_equal(doubled, flags)
    np.testing.assert_array_equal(sj.get_round_scores(state), final)


def test_terminal_sampling_averages_scored_realizations_not_mean_scores(monkeypatch):
    endings = [terminal([10, 20]), terminal([24, 20])]
    apply = Mock(side_effect=endings)
    monkeypatch.setattr(targets.sj, "apply_action", apply)
    monkeypatch.setattr(targets.sj, "is_action_random", lambda *args: True)
    summary = targets.summarize_terminal(sj.new(players=2), sj.MASK_FLIP, 2, rng=random.Random(1))
    np.testing.assert_array_equal(summary.raw_scores, [17, 20])
    np.testing.assert_array_equal(summary.doubled, [0.5, 0])
    np.testing.assert_array_equal(summary.scores, [29, 20])
    np.testing.assert_array_equal(summary.value, [-14, -5])
    np.testing.assert_array_equal(summary.outcome, [0.5, 0.5])
    assert apply.call_count == 2
    monkeypatch.setattr(targets.sj, "is_action_random", lambda *args: False)
    apply.reset_mock(side_effect=True)
    apply.return_value = endings[0]
    targets.summarize_terminal(endings[0], sj.MASK_REPLACE, 32)
    assert apply.call_count == 1


def quick_history(players=3):
    rng = random.Random(21)
    state = sj.start_round(sj.new(players=players, rng=rng), rng=rng)
    history = []
    while not sj.get_game_over(state):
        action = sj.quick_finish_action(state)
        policy = np.zeros(sj.MASK_SIZE, dtype=np.float32)
        policy[action] = 1
        history.append(play.GameHistoryEntry(state, action, policy))
        state = sj.apply_action(state, action, rng=rng)
    history.append(play.GameHistoryEntry(state, None, None))
    return history


@pytest.fixture(scope="module")
def history():
    return quick_history()


def model(config, players=3):
    return skynet.EquivariantSkyNet(
        spatial_input_shape=(players, sj.ROW_COUNT, sj.COLUMN_COUNT, sj.FINGER_SIZE),
        non_spatial_input_shape=(sj.GAME_SIZE,), value_output_shape=(players,),
        policy_output_shape=(sj.MASK_SIZE,), device=torch.device("cpu"),
        embedding_dimensions=8, global_state_embedding_dimensions=16, num_heads=2,
        auxiliary_objectives=config,
    )


def test_target_building_is_shared_rotated_and_randomness_isolated(history, monkeypatch):
    before = pickle.dumps(history)
    python_rng = random.getstate()
    numpy_rng = pickle.dumps(np.random.get_state())
    builder = Mock(wraps=targets.CONTEXT_BUILDERS["terminal"])
    monkeypatch.setitem(targets.CONTEXT_BUILDERS, "terminal", builder)
    rows, _ = targets.build_targets(history, ABLATIONS[-1], terminal_rollouts=4, rng=random.Random(8))
    assert builder.call_count == 1
    summary = targets.terminal_context(history, 4, random.Random(f"{random.Random(8).getrandbits(128)}:terminal"))
    for row in rows:
        shift = sj.get_player(row.state)
        np.testing.assert_array_equal(np.roll(row.targets["round_raw_score"], shift), summary.raw_scores)
        np.testing.assert_array_equal(np.roll(row.targets["round_doubled"], shift), summary.doubled)
    base, _ = targets.build_targets(history, {}, terminal_rollouts=4, rng=random.Random(8))
    for full, core in zip(rows, base):
        np.testing.assert_array_equal(full.targets["value"], core.targets["value"])
        np.testing.assert_array_equal(full.targets["policy"], core.targets["policy"])
        assert set(core.targets) == {"value", "policy"}
    assert pickle.dumps(history) == before
    assert random.getstate() == python_rng
    assert pickle.dumps(np.random.get_state()) == numpy_rng


@pytest.mark.parametrize("rollouts", [0, -1, 1.5])
def test_invalid_rollout_count_rejected(history, rollouts):
    with pytest.raises(ValueError, match="positive integer"):
        targets.build_targets(history, terminal_rollouts=rollouts)


@pytest.mark.parametrize("config", ABLATIONS)
def test_each_ablation_trains_and_reconstructs_for_inference(history, config, tmp_path):
    torch.manual_seed(7)
    baseline = model({})
    core_rng = torch.get_rng_state()
    torch.manual_seed(7)
    net = model(config)
    assert torch.equal(torch.get_rng_state(), core_rng)
    for name, tensor in baseline.state_dict().items():
        assert torch.equal(net.state_dict()[name], tensor)
    rows, _ = targets.build_targets(history, net.objectives, rng=random.Random(5))
    replay = buffer.ReplayBuffer.from_config(buffer.for_objectives(buffer.Config(
        max_size=len(rows), spatial_input_shape=net.spatial_input_shape,
        non_spatial_input_shape=net.non_spatial_input_shape, action_mask_shape=net.policy_output_shape,
    ), net.objectives))
    replay.add_game_data(rows)
    batch = replay.sample_batch(8)
    loss, details = train.train_step(net, batch, train_utils.base_loss, train.make_optimizer(net, 1e-3))
    assert np.isfinite(loss)
    assert set(net.auxiliary_heads) == set(config)
    assert net.card_embedder.weight.grad.abs().sum() > 0
    for name, head in net.auxiliary_heads.items():
        assert head.weight.grad.abs().sum() > 0
        assert details[f"{name}_weighted_loss"] == pytest.approx(config[name] * details[f"{name}_loss"])
    if not config:
        assert set(details) == {"score_differential_value_loss", "policy_loss"}
    saved = net.save(tmp_path)
    restored = skynet.EquivariantSkyNet.from_checkpoint(saved, torch.device("cpu"))
    assert restored.objectives.weights == config
    for name, tensor in net.state_dict().items():
        assert torch.equal(restored.state_dict()[name], tensor)
    # Both the direct prediction adapter and batched client expose only core outputs.
    direct = restored.predict(history[0].state)
    client = predictor.LocalPredictorClient(restored, max_batch_size=2)
    ids = [client.put(row.state) for row in rows[:2]]
    client.send()
    predictions = client.get_all()
    assert [item[0] for item in predictions] == ids
    np.testing.assert_allclose(predictions[0][1].value_output, direct.value_output, atol=1e-5)
    np.testing.assert_allclose(predictions[0][1].policy_output, direct.policy_output, atol=1e-6)
    # Factories propagate the initial model's config to reconstructed workers.
    model_factory = factory.SkyNetModelFactory(
        skynet.EquivariantSkyNet, players=3, models_dir=tmp_path, initial_model=net,
    )
    assert model_factory.get_latest_model().objectives.weights == config


def test_missing_auxiliary_labels_fail_but_extra_labels_are_usable(history):
    net = model(ABLATIONS[-1])
    core, _ = targets.build_targets(history)
    core_batch = train_utils.game_data_to_training_batch(core)
    with pytest.raises(ValueError, match="Missing target"):
        train.train_step(net, core_batch, train_utils.base_loss, train.make_optimizer(net, 1e-3))
    value_loss, policy_loss = train_utils.compute_model_loss_on_game_data(
        net, core, train_utils.policy_value_losses
    )
    assert torch.isfinite(value_loss + policy_loss)
    extra = core_batch._replace(target_arrays={**core_batch.targets, "unused": np.zeros((len(core), 1))})
    baseline = model({})
    train.train_step(baseline, extra, train_utils.base_loss, train.make_optimizer(baseline, 1e-3))
    assert objectives.resolve({"round_raw_score": 0}).weights == {}
    with pytest.raises(ValueError, match="nonnegative"):
        objectives.resolve({"round_doubled": -1})
    with pytest.raises(ValueError, match="Unknown"):
        objectives.resolve({"typo": 1})


def test_selfplay_worker_returns_history_without_constructing_targets(history, monkeypatch):
    monkeypatch.setattr(targets, "build_targets", Mock(side_effect=AssertionError("worker built targets")))
    worker = play.SelfplayGenerator(
        "test", player=None, player_count=3, game_data_queue=None,
        play_callable=lambda *args, **kwargs: history,
    )
    assert worker.generate_episode() is history


def test_learner_labels_queued_histories_before_replay_and_reuses_labels(history, tmp_path, monkeypatch):
    net = model(ABLATIONS[-1])
    replay = buffer.ReplayBuffer.from_config(buffer.for_objectives(buffer.Config(
        max_size=len(history), spatial_input_shape=net.spatial_input_shape,
        non_spatial_input_shape=net.non_spatial_input_shape,
        action_mask_shape=net.policy_output_shape, path=tmp_path / "buffer.pkl",
    ), net.objectives))
    histories = queue.Queue()
    histories.put(history)
    build = Mock(wraps=targets.build_targets)
    monkeypatch.setattr(targets, "build_targets", build)
    train.learn(
        model_factory=SimpleNamespace(get_latest_model=lambda: net),
        predictor_clients={}, training_data_buffer=replay, training_data_queue=histories,
        torch_device=torch.device("cpu"), learn_steps=1, games_generated_per_iteration=1,
        training_epochs=2, training_batch_size=8, training_learn_rate=1e-3,
        training_loss_function=train_utils.base_loss, loss_stats_function=None,
        validation_interval=None, validation_function=None, update_model_interval=None,
        model_faceoff_function=None, outcome_rollouts=2,
    )
    assert build.call_count == 1
    assert len(replay) == len(history) - 1
    assert set(replay.target_names) == {"value", "policy", *ABLATIONS[-1]}


def test_final_reveal_clears_column_before_raw_score_target():
    state = list(terminal([10, 5]))
    # Other player has a pair of -2 cards and one hidden card. Force the reveal
    # to -2 using a zero uniform draw; all earlier card types are exhausted.
    state[1][1, :, 0, :] = 0
    state[1][1, :2, 0, sj.CARD_N2] = 1
    state[1][1, 2, 0, sj.FINGER_HIDDEN] = 1
    state[0][sj.GAME_ACTION:sj.GAME_ACTION + sj.ACTION_SIZE] = 0
    state[0][sj.GAME_ACTION + sj.ACTION_REPLACE] = 1
    # Replace current player's visible 10 with the visible top 0; after rotation
    # the pending -2 reveal clears the other player's entire remaining column.
    state[0][sj.GAME_TOP:sj.GAME_TOP + sj.CARD_SIZE] = 0
    state[0][sj.GAME_TOP + sj.CARD_0] = 1
    state[2][:] = np.array(sj.CARD_COUNTS) - state[1][:, :, :, :sj.CARD_SIZE].sum(axis=(0, 1, 2))
    state[2][sj.CARD_0] -= 1
    state[6] = 1
    sj.validate(tuple(state))
    summary = targets.summarize_terminal(tuple(state), sj.MASK_REPLACE, 1,
                                         rng=SimpleNamespace(random=lambda: 0.0))
    np.testing.assert_array_equal(summary.raw_scores, [0, 0])
    assert summary.cleared_columns[4] == 1


def test_auxiliary_losses_use_normalized_mse_and_soft_label_bce():
    score_loss, metrics = objectives.raw_score_loss(torch.tensor([[144.0]]), torch.tensor([[0.0]]))
    assert score_loss.item() == pytest.approx(1.0)
    assert metrics["mae_points"] == 144.0
    doubled_loss, _ = objectives.doubled_loss(torch.tensor([[0.0]]), torch.tensor([[0.5]]))
    assert doubled_loss.item() == pytest.approx(np.log(2))
