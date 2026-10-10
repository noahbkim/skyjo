import copy
import dataclasses
import functools
import pickle
import random
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from helpers import NaiveQuickFinishPlayer

from skyjo.engine import game as sj
from skyjo.learning import buffer, checkpoint, losses, objectives, observations
from skyjo.learning import predictor, replay_io, skynet, targets, train
from skyjo.simulation import play

ALL = {name: 0.1 for name in objectives.OBJECTIVE_NAMES}


def terminal(scores, turn=0, stalled=()):
    state = sj.new(players=len(scores), rng=random.Random(0))
    state = dataclasses.replace(state, turn=turn, countdown=0)
    state.game[
        sj.GAME_LAST_REVEALED_TURNS : sj.GAME_LAST_REVEALED_TURNS + len(scores)
    ] = turn
    for seat in stalled:
        state.game[sj.GAME_LAST_REVEALED_TURNS + seat] = 0
    state.table.fill(0)
    state.table[: len(scores), :, :, sj.FINGER_CLEARED] = 1
    for seat, score in enumerate(scores):
        # Three visible cards in a column, deliberately not a matching triple.
        values = [score, 0, 0] if score <= 12 else [12, score - 12, 0]
        state.table[seat, :, 0, :] = 0
        for row, value in enumerate(values):
            state.table[seat, row, 0, value + 2] = 1
    return state


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


def model(config, players=3):
    return skynet.EquivariantSkyNet(
        spatial_input_shape=(players, sj.ROW_COUNT, sj.COLUMN_COUNT, sj.FINGER_SIZE),
        non_spatial_input_shape=observations.get_non_spatial_input_shape(players),
        value_output_shape=(players,),
        policy_output_shape=(sj.MASK_SIZE,),
        device=torch.device("cpu"),
        embedding_dimensions=8,
        global_state_embedding_dimensions=16,
        num_heads=2,
        auxiliary_objectives=config,
    )


@pytest.fixture(scope="module")
def result():
    return play.play_game(
        [NaiveQuickFinishPlayer() for _ in range(3)],
        environment_rng=np.random.default_rng(21),
        action_rng=np.random.default_rng(22),
    )


def test_round_labels_preserve_observed_targets_history_and_rng(result, monkeypatch):
    before = pickle.dumps(result)
    rng = checkpoint.capture_rng_state()
    baseline = targets.build_training_batch(result)
    builder = Mock(wraps=targets.summarize_round)
    monkeypatch.setattr(targets, "summarize_round", builder)
    observed = targets.build_training_batch(result, ALL)
    assert builder.call_count == len(result.rounds)
    offset = 0
    for round_result in result.rounds:
        raw = np.roll(
            [sj.get_score(round_result.history[-1].state, i) for i in range(3)],
            sj.get_player(round_result.history[-1].state),
        )
        for entry in round_result.history[:-1]:
            shift = sj.get_player(entry.state)
            np.testing.assert_allclose(
                np.roll(observed.targets["round_raw_score"][offset], shift) * 144,
                raw,
                atol=1e-5,
            )
            np.testing.assert_allclose(
                np.roll(observed.targets["round_score"][offset], shift) * 336 - 48,
                round_result.round_scores,
                atol=1e-5,
            )
            offset += 1
    sampled = targets.build_training_batch(
        result, ALL, mode="resampled", samples=8, seed=5
    )
    subset = targets.build_training_batch(
        result, {"round_raw_score": 0.1}, mode="resampled", samples=8, seed=5
    )
    for name in ("value", "policy"):
        np.testing.assert_array_equal(baseline.targets[name], observed.targets[name])
        np.testing.assert_array_equal(baseline.targets[name], sampled.targets[name])
    np.testing.assert_array_equal(
        sampled.targets["round_raw_score"], subset.targets["round_raw_score"]
    )
    assert pickle.dumps(result) == before
    assert random.getstate() == rng["python"]
    np.testing.assert_equal(np.random.get_state(), rng["numpy"])
    assert torch.equal(torch.get_rng_state(), rng["torch_cpu"])


def test_resampling_preserves_correlation_and_skips_deterministic_work(
    result, monkeypatch
):
    endings = [terminal([10, 20, 21]), terminal([24, 20, 21])]
    apply = Mock(side_effect=endings)
    monkeypatch.setattr(targets.sj, "apply_action", apply)
    monkeypatch.setattr(targets.sj, "is_action_random", lambda *args: True)
    summary = targets.summarize_round(result.rounds[0], mode="resampled", samples=2)
    np.testing.assert_array_equal(summary.raw_scores, [17, 20, 21])
    np.testing.assert_array_equal(summary.doubled, [0.5, 0, 0])
    np.testing.assert_array_equal(summary.scores, [29, 20, 21])
    monkeypatch.setattr(targets.sj, "is_action_random", lambda *args: False)
    apply.reset_mock(side_effect=True)
    apply.return_value = endings[0]
    targets.summarize_round(result.rounds[0], mode="resampled", samples=32)
    assert apply.call_count == 1


@pytest.mark.parametrize(
    "configuration",
    [
        {},
        {"round_raw_score": 0},
        {"round_score": 0.1},
        {"round_raw_score": 0.1},
        {"round_doubled": 0.1},
        ALL,
    ],
)
def test_optional_objectives_through_encoding_replay_fit_and_checkpoint(
    result, configuration, tmp_path
):
    torch.manual_seed(7)
    baseline = model({})
    expected_rng = torch.get_rng_state()
    torch.manual_seed(7)
    net = model(configuration)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(net.state_dict()[name], value, rtol=0, atol=0)
    active = objectives.resolve(configuration).weights
    batch = targets.build_training_batch(result, configuration)
    replay = buffer.ReplayBuffer(
        len(batch),
        net.spatial_input_shape,
        net.non_spatial_input_shape,
        net.policy_output_shape,
        (
            *buffer.core_target_specs(3, net.policy_output_shape),
            *(buffer.TargetShapeSpec(name, (3,)) for name in active),
        ),
    )
    replay.append(batch, buffer.GameProvenance(0))
    replay_io.save(replay, tmp_path / "replay")
    replay = replay_io.load(tmp_path / "replay")
    state = result.rounds[0].history[0].state
    before = predictor.LocalPredictor(net, 2).evaluate([state])[0]
    expected = predictor.LocalPredictor(baseline, 2).evaluate([state])[0]
    np.testing.assert_array_equal(before.value, expected.value)
    np.testing.assert_array_equal(before.policy, expected.policy)
    loss, details = train.train_step(
        net,
        replay.sample_batch(8, rng=np.random.default_rng(0)),
        functools.partial(losses.configured_loss, auxiliary_objectives=configuration),
        train.make_optimizer(net, 1e-3),
    )
    assert np.isfinite(loss)
    assert net.card_embedder.weight.grad.abs().sum() > 0
    for name, head in net.auxiliary_heads.items():
        assert all(
            p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters()
        )
        assert details[f"{name}_weighted_loss"] == pytest.approx(
            active[name] * details[f"{name}_loss"]
        )
    saved = checkpoint.save_checkpoint(
        tmp_path / "model.pth", model=net, optimizer=None
    )
    restored = model(configuration)
    checkpoint.load_checkpoint(saved, model=restored, restore_rng=False)
    expected = predictor.LocalPredictor(net, 2).evaluate([state])[0]
    actual = predictor.LocalPredictor(restored, 2).evaluate([state])[0]
    np.testing.assert_array_equal(actual.value, expected.value)
    np.testing.assert_array_equal(actual.policy, expected.policy)


def test_gradient_diagnostic_does_not_change_optimizer_update(result):
    torch.manual_seed(11)
    net = model(ALL)
    baseline = copy.deepcopy(net)
    batch = targets.build_training_batch(result, ALL)[:8]
    loss_fn = functools.partial(losses.configured_loss, auxiliary_objectives=ALL)
    scales = {}
    expected = train.train_step(
        baseline, batch, loss_fn, train.make_optimizer(baseline, 1e-3)
    )
    actual = train.train_step(
        net, batch, loss_fn, train.make_optimizer(net, 1e-3), gradient_scales=scales
    )
    assert expected == actual
    assert scales["core_weighted_norm"] > 0
    for name in ALL:
        assert scales[f"{name}/weighted_norm"] == pytest.approx(
            0.1 * scales[f"{name}/unweighted_norm"]
        )
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(net.state_dict()[name], value, rtol=0, atol=0)


def test_normalized_losses_and_missing_labels():
    for fn, points in ((losses.raw_score_loss, 144), (losses.charged_loss, 336)):
        loss, metrics = fn(torch.ones(1, 2), torch.zeros(1, 2))
        assert loss.item() == 1 and metrics["mae_points"] == points
    loss, _ = losses.doubled_loss(torch.zeros(1, 2), torch.full((1, 2), 0.5))
    assert loss.item() == pytest.approx(np.log(2))
    output = skynet.ModelOutput(torch.zeros(1, 2), torch.zeros(1, 2))
    with pytest.raises(ValueError, match="Missing target"):
        losses.configured_loss(
            output,
            {"value": torch.zeros(1, 2), "policy": torch.full((1, 2), 0.5)},
            auxiliary_objectives=ALL,
        )
    with pytest.raises(ValueError, match="nonnegative"):
        objectives.resolve({"round_doubled": -1})


def test_final_reveal_clears_column_before_raw_score_target():
    state = terminal([10, 5])
    # Other player has a pair of -2 cards and one hidden card. Force the reveal
    # to -2 using a zero uniform draw; all earlier card types are exhausted.
    state.table[1, :, 0, :] = 0
    state.table[1, :2, 0, sj.CARD_N2] = 1
    state.table[1, 2, 0, sj.FINGER_HIDDEN] = 1
    state.game[sj.GAME_ACTION : sj.GAME_ACTION + sj.ACTION_SIZE] = 0
    state.game[sj.GAME_ACTION + sj.ACTION_REPLACE] = 1
    # Replace current player's visible 10 with the visible top 0; after rotation
    # the pending -2 reveal clears the other player's entire remaining column.
    state.game[sj.GAME_TOP : sj.GAME_TOP + sj.CARD_SIZE] = 0
    state.game[sj.GAME_TOP + sj.CARD_0] = 1
    state.deck[:] = np.array(sj.CARD_COUNTS) - state.table[:, :, :, : sj.CARD_SIZE].sum(
        axis=(0, 1, 2)
    )
    state.deck[sj.CARD_0] -= 1
    state = dataclasses.replace(state, countdown=1)
    sj.validate(state)
    final = sj.apply_action(
        state, sj.MASK_REPLACE, rng=SimpleNamespace(random=lambda: 0.0)
    )
    summary = targets.terminal_summary(final)
    np.testing.assert_array_equal(summary.raw_scores, [0, 0])
    assert sj.get_table(final)[0, :, 0, sj.FINGER_CLEARED].all()


def test_configured_auxiliary_heads_preserve_column_symmetry():
    net = model(ALL).eval()
    spatial = torch.rand(2, 3, sj.ROW_COUNT, sj.COLUMN_COUNT, sj.FINGER_SIZE)
    nonspatial = torch.rand(2, *observations.get_non_spatial_input_shape(3))
    mask = torch.ones(2, sj.MASK_SIZE)
    permutation = torch.tensor([2, 0, 3, 1])
    with torch.inference_mode():
        original = net(spatial, nonspatial, mask)
        permuted = net(spatial[:, :, :, permutation], nonspatial, mask)
    for name in ALL:
        assert original.auxiliary_outputs[name].shape == (2, 3)
        torch.testing.assert_close(
            original.auxiliary_outputs[name],
            permuted.auxiliary_outputs[name],
            atol=1e-6,
            rtol=1e-5,
        )
