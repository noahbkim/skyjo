import copy
import functools
import pickle
import random
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from skyjo import (
    buffer,
    checkpoint,
    experiment_config,
    objectives,
    play,
    player,
    predictor,
    skynet,
    targets,
    train,
    train_utils,
)
from skyjo import game as sj

ALL = {name: 0.1 for name in objectives.REGISTRY}


def terminal(scores, turn=0, stalled=()):
    state = list(sj.new(players=len(scores), rng=random.Random(0)))
    state[4], state[6] = turn, 0
    state[0][
        sj.GAME_LAST_REVEALED_TURNS : sj.GAME_LAST_REVEALED_TURNS + len(scores)
    ] = turn
    for seat in stalled:
        state[0][sj.GAME_LAST_REVEALED_TURNS + seat] = 0
    state[1].fill(0)
    state[1][: len(scores), :, :, sj.FINGER_CLEARED] = 1
    for seat, score in enumerate(scores):
        # Three visible cards in a column, deliberately not a matching triple.
        values = [score, 0, 0] if score <= 12 else [12, score - 12, 0]
        state[1][seat, :, 0, :] = 0
        for row, value in enumerate(values):
            state[1][seat, row, 0, value + 2] = 1
    return tuple(state)


@pytest.mark.parametrize(
    "scores,turn,stalled,flags,final",
    [
        ([10, 20], 0, (), [False, False], [10, 20]),
        ([20, 10], 0, (), [True, False], [40, 10]),
        ([10, 10], 0, (), [True, False], [20, 10]),
        ([0, 0], 0, (), [True, False], [0, 0]),
        ([-1, -2], 0, (), [True, False], [-2, -2]),
        ([0, 10, 20], 63, (0, 1), [True, True, False], [0, 20, 20]),
    ],
)
def test_scoring_exposes_rule_flags_even_when_score_does_not_change(
    scores, turn, stalled, flags, final
):
    state = terminal(scores, turn, stalled)
    raw, doubled = sj.get_round_score_components(state)
    np.testing.assert_array_equal(raw, scores)
    np.testing.assert_array_equal(doubled, flags)
    np.testing.assert_array_equal(sj.get_round_scores(state), final)


@pytest.fixture(scope="module")
def result():
    random.seed(21)
    np.random.seed(21)
    return play.play_game([player.NaiveQuickFinishPlayer() for _ in range(3)])


def model(config, players=3):
    return skynet.EquivariantSkyNet(
        spatial_input_shape=(players, sj.ROW_COUNT, sj.COLUMN_COUNT, sj.FINGER_SIZE),
        non_spatial_input_shape=skynet.get_non_spatial_input_shape(players),
        value_output_shape=(players,),
        policy_output_shape=(sj.MASK_SIZE,),
        device=torch.device("cpu"),
        embedding_dimensions=8,
        global_state_embedding_dimensions=16,
        num_heads=2,
        auxiliary_objectives=config,
    )


def test_round_local_labels_preserve_full_game_targets_and_rng(result, monkeypatch):
    assert len(result.rounds) > 1
    before = pickle.dumps(result)
    rng = pickle.dumps(checkpoint.capture_rng_state())
    baseline, stats = play.game_result_to_game_data(result)
    builder = Mock(wraps=targets.summarize_round)
    monkeypatch.setitem(targets.CONTEXT_BUILDERS, "terminal", builder)
    observed, _ = play.game_result_to_game_data(result, ALL)
    assert builder.call_count == len(result.rounds)
    offset = 0
    raw_by_round = []
    for round_result in result.rounds:
        final = round_result.history[-1].state
        raw = np.roll([sj.get_score(final, i) for i in range(3)], sj.get_player(final))
        raw_by_round.append(tuple(raw))
        for row in observed[offset : offset + len(round_result.history) - 1]:
            shift = sj.get_player(row.state)
            np.testing.assert_allclose(
                np.roll(row.targets["round_raw_score"], shift) * 144, raw, atol=1e-5
            )
            np.testing.assert_allclose(
                np.roll(row.targets["round_score"], shift) * 336 - 48,
                round_result.round_scores,
                atol=1e-5,
            )
        offset += len(round_result.history) - 1
    assert len(set(raw_by_round)) > 1
    sampled, sample_stats = play.game_result_to_game_data(
        result, ALL, mode="resampled", samples=8, seed=5
    )
    subset, _ = play.game_result_to_game_data(
        result, {"round_raw_score": 0.1}, mode="resampled", samples=8, seed=5
    )
    for core, full, resampled, selected in zip(
        baseline, observed, sampled, subset, strict=True
    ):
        for name in ("value", "policy"):
            np.testing.assert_array_equal(core.targets[name], full.targets[name])
            np.testing.assert_array_equal(core.targets[name], resampled.targets[name])
        np.testing.assert_array_equal(
            resampled.targets["round_raw_score"], selected.targets["round_raw_score"]
        )
    assert pickle.dumps(stats) == pickle.dumps(sample_stats)
    assert pickle.dumps(result) == before
    # torch RNG pickle serializes storage identifiers; compare subsequent draws instead below.
    saved = pickle.loads(rng)
    assert random.getstate() == saved["python"]
    np.testing.assert_equal(np.random.get_state(), saved["numpy"])
    assert torch.equal(torch.get_rng_state(), saved["torch_cpu"])


def test_correlated_sample_penalties_and_deterministic_shortcut(result, monkeypatch):
    endings = [terminal([10, 20, 21]), terminal([24, 20, 21])]
    apply = Mock(side_effect=endings)
    monkeypatch.setattr(targets.sj, "apply_action", apply)
    monkeypatch.setattr(targets.sj, "is_action_random", lambda *args: True)
    summary = targets.summarize_round(result.rounds[0], mode="resampled", samples=2)
    np.testing.assert_array_equal(summary.raw_scores, [17, 20, 21])
    np.testing.assert_array_equal(summary.doubled, [0.5, 0, 0])
    np.testing.assert_array_equal(summary.scores, [29, 20, 21])
    assert apply.call_count == 2
    # Resampling radically different endings still leaves the observed game
    # winners, statistics, and labels for the core outcome untouched.
    baseline, stats = play.game_result_to_game_data(result)
    apply.side_effect = endings * len(result.rounds)
    sampled, sampled_stats = play.game_result_to_game_data(
        result, ALL, mode="resampled", samples=2
    )
    assert pickle.dumps(stats) == pickle.dumps(sampled_stats)
    for row, core in zip(sampled, baseline, strict=True):
        np.testing.assert_array_equal(row.targets["value"], core.targets["value"])
        fixed_raw = np.roll(row.targets["round_raw_score"], sj.get_player(row.state))
        np.testing.assert_allclose(fixed_raw * 144, [17, 20, 21])
    monkeypatch.setattr(targets.sj, "is_action_random", lambda *args: False)
    apply.reset_mock(side_effect=True)
    apply.return_value = endings[0]
    targets.summarize_round(result.rounds[0], mode="resampled", samples=32)
    assert apply.call_count == 1
    with pytest.raises(ValueError, match="positive integer"):
        targets.summarize_round(result.rounds[0], samples=0)


@pytest.mark.parametrize(
    "configuration",
    [
        {},
        {"round_raw_score": 0},
        {"round_score": 0.1},
        {"round_raw_score": 0.1},
        {"round_doubled": 0.1},
        {"round_raw_score": 0.1, "round_doubled": 0.1},
        ALL,
    ],
)
def test_objectives_through_replay_training_and_inference(
    result, configuration, tmp_path
):
    torch.manual_seed(7)
    baseline = model({})
    core_rng = torch.get_rng_state()
    torch.manual_seed(7)
    net = model(configuration)
    assert torch.equal(torch.get_rng_state(), core_rng)
    for name, tensor in baseline.state_dict().items():
        assert torch.equal(net.state_dict()[name], tensor)
    active = objectives.resolve(configuration).weights
    rows, _ = play.game_result_to_game_data(result, configuration)
    replay = buffer.ReplayBuffer.from_config(
        buffer.Config(
            max_size=len(rows),
            spatial_input_shape=net.spatial_input_shape,
            non_spatial_input_shape=net.non_spatial_input_shape,
            action_mask_shape=net.policy_output_shape,
            target_specs=experiment_config.target_specs(3, configuration),
            path=tmp_path / "data",
        )
    )
    replay.add_game_data(rows)
    replay.save()
    replay = buffer.ReplayBuffer.load(tmp_path / "data")
    assert set(replay.target_names) == {"value", "policy", *active}
    before = net.predict(rows[0].state)
    expected = baseline.predict(rows[0].state)
    np.testing.assert_array_equal(before.value_output, expected.value_output)
    np.testing.assert_array_equal(before.policy_output, expected.policy_output)
    loss_fn = functools.partial(
        objectives.configured_loss, auxiliary_objectives=configuration
    )
    loss, details = train.train_step(
        net, replay.sample_batch(8), loss_fn, train.make_optimizer(net, 1e-3)
    )
    assert np.isfinite(loss)
    assert set(net.auxiliary_heads) == set(active)
    assert net.card_embedder.weight.grad.abs().sum() > 0
    for name, head in net.auxiliary_heads.items():
        assert all(
            p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters()
        )
        assert details[f"{name}_weighted_loss"] == pytest.approx(
            active[name] * details[f"{name}_loss"]
        )
    if not active:
        assert set(details) == {"total_loss", "outcome_value_loss", "policy_loss"}
    saved = checkpoint.save_checkpoint(
        tmp_path / "model.pth",
        model=net,
        optimizer=None,
        configuration={"auxiliary_objectives": configuration},
    )
    restored = model(configuration)
    checkpoint.load_checkpoint(saved, model=restored, restore_rng=False)
    direct = restored.predict(rows[0].state)
    client = predictor.LocalPredictorClient(restored, max_batch_size=2)
    ids = [client.put(row.state) for row in rows[:2]]
    client.send()
    predictions = client.get_all()
    assert [item[0] for item in predictions] == ids
    np.testing.assert_allclose(
        predictions[0][1].value_output, direct.value_output, atol=1e-5
    )
    np.testing.assert_allclose(
        predictions[0][1].policy_output, direct.policy_output, atol=1e-6
    )


def test_disabled_loss_and_diagnostic_preserve_optimizer_update(result):
    torch.manual_seed(11)
    baseline = model({})
    net = copy.deepcopy(baseline)
    rows, _ = play.game_result_to_game_data(result, ALL)
    batch = train_utils.game_data_to_training_batch(
        rows[:8], target_names=("value", "policy", *ALL)
    )
    rng = torch.get_rng_state()
    expected = train.train_step(
        baseline, batch, train_utils.base_loss, train.make_optimizer(baseline, 1e-3)
    )
    scales = {}
    actual = train.train_step(
        net,
        batch,
        functools.partial(
            objectives.configured_loss, auxiliary_objectives={"round_raw_score": 0}
        ),
        train.make_optimizer(net, 1e-3),
        gradient_scales=scales,
    )
    assert expected == actual
    assert scales["core_weighted_norm"] > 0
    assert torch.equal(torch.get_rng_state(), rng)
    for name, tensor in baseline.state_dict().items():
        assert torch.equal(net.state_dict()[name], tensor)


def test_auxiliary_diagnostic_uses_shared_graph_without_changing_gradients(result):
    from skyjo.gradient_diagnostic import measure

    net = model(ALL)
    rows, _ = play.game_result_to_game_data(result, ALL)
    batch = train_utils.game_data_to_training_batch(
        rows[:4], target_names=("value", "policy", *ALL)
    )
    t = train_utils.numpy_targets_to_tensors(batch.targets, device=net.device)
    out = net(
        torch.tensor(batch.spatial_inputs),
        torch.tensor(batch.non_spatial_inputs),
        torch.tensor(batch.action_masks),
    )
    for p in net.parameters():
        p.grad = torch.ones_like(p)
    rng = torch.get_rng_state()
    scales = measure(net, out, t, auxiliary_objectives=ALL)
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(p.grad, torch.ones_like(p)) for p in net.parameters())
    for name in ALL:
        assert scales[f"{name}/unweighted_norm"] > 0
        assert scales[f"{name}/weighted_norm"] == pytest.approx(
            0.1 * scales[f"{name}/unweighted_norm"]
        )
    loss, _ = objectives.configured_loss(out, t, auxiliary_objectives=ALL)
    loss.backward()  # Graph remains usable for the real optimizer step.


def test_normalized_losses_and_missing_labels():
    for fn, points in (
        (objectives.raw_score_loss, 144),
        (objectives.charged_loss, 336),
    ):
        loss, metrics = fn(torch.ones(1, 2), torch.zeros(1, 2))
        assert loss.item() == 1
        assert metrics["mae_points"] == points
    loss, _ = objectives.doubled_loss(torch.zeros(1, 2), torch.full((1, 2), 0.5))
    assert loss.item() == pytest.approx(np.log(2))
    with pytest.raises(ValueError, match="Missing target"):
        objectives.resolve(ALL).add_losses(torch.tensor(0.0), {}, SimpleNamespace(), {})
    with pytest.raises(ValueError, match="nonnegative"):
        objectives.resolve({"round_doubled": -1})


def test_final_reveal_clears_column_before_raw_score_target():
    state = list(terminal([10, 5]))
    # Other player has a pair of -2 cards and one hidden card. Force the reveal
    # to -2 using a zero uniform draw; all earlier card types are exhausted.
    state[1][1, :, 0, :] = 0
    state[1][1, :2, 0, sj.CARD_N2] = 1
    state[1][1, 2, 0, sj.FINGER_HIDDEN] = 1
    state[0][sj.GAME_ACTION : sj.GAME_ACTION + sj.ACTION_SIZE] = 0
    state[0][sj.GAME_ACTION + sj.ACTION_REPLACE] = 1
    # Replace current player's visible 10 with the visible top 0; after rotation
    # the pending -2 reveal clears the other player's entire remaining column.
    state[0][sj.GAME_TOP : sj.GAME_TOP + sj.CARD_SIZE] = 0
    state[0][sj.GAME_TOP + sj.CARD_0] = 1
    state[2][:] = np.array(sj.CARD_COUNTS) - state[1][:, :, :, : sj.CARD_SIZE].sum(
        axis=(0, 1, 2)
    )
    state[2][sj.CARD_0] -= 1
    state[6] = 1
    sj.validate(tuple(state))
    final = sj.apply_action(
        tuple(state), sj.MASK_REPLACE, rng=SimpleNamespace(random=lambda: 0.0)
    )
    summary = targets.terminal_summary(final)
    np.testing.assert_array_equal(summary.raw_scores, [0, 0])
    assert sj.get_table(final)[0, :, 0, sj.FINGER_CLEARED].all()
