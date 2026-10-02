"""Numerical diagnostics, observational evaluation, and bounded progress reporting."""

import copy
import logging
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from skyjo import checkpoint, experiment_training, explain, skynet, train, train_utils
from skyjo import game as sj


def test_policy_diagnostics_weight_positions_and_ignore_masked_actions():
    logits = torch.full((4, sj.MASK_SIZE), 100.0, requires_grad=True)
    masks = torch.zeros_like(logits)
    # One example from every phase, each with two equiprobable legal choices.
    for i, pair in enumerate(((0, 1), (2, 3), (4, 16), (16, 17))):
        masks[i, list(pair)] = 1
    targets = masks / 2
    targets[1, 2:4] = torch.tensor([1.0, 0.0])
    stats = train_utils.TrainingDiagnostics()
    stats.update(logits[:1], targets[:1], masks[:1])
    stats.update(logits[1:], targets[1:], masks[1:])
    report = stats.summary()

    assert report["policy/all/positions"] == 4
    assert report["policy/all/target_entropy"] == pytest.approx(0.75 * np.log(2))
    assert report["policy/all/predicted_entropy"] == pytest.approx(np.log(2))
    assert report["policy/all/target_kl"] == pytest.approx(0.25 * np.log(2))
    for phase in ("initial_reveal", "draw_take", "flip_replace", "replace_only"):
        assert report[f"policy/{phase}/positions"] == 1
    assert report["policy/draw_take/target_kl"] == pytest.approx(np.log(2))
    assert all(not total.requires_grad for total in stats.totals.values())


def small_model():
    return skynet.EquivariantSkyNet(
        spatial_input_shape=(2, 3, 4, sj.FINGER_SIZE),
        non_spatial_input_shape=skynet.get_non_spatial_input_shape(2),
        value_output_shape=(2,),
        policy_output_shape=(sj.MASK_SIZE,),
        device=torch.device("cpu"),
        embedding_dimensions=8,
        global_state_embedding_dimensions=16,
        num_heads=1,
    )


def test_observing_training_preserves_updates_rng_and_module_modes():
    torch.set_num_threads(1)
    model = small_model()
    model.train()
    # Preserve mixed modes as well as the top-level flag.
    model.card_embedder.eval()
    modes = [module.training for module in model.modules()]
    initial = copy.deepcopy(model)
    rng = checkpoint.capture_rng_state()
    report = explain.evaluate_concepts(model)
    assert len(report.examples) == 10
    assert [module.training for module in model.modules()] == modes
    actual_random = (random.random(), np.random.random(), torch.rand(3))
    checkpoint.restore_rng_state(rng)
    expected_random = (random.random(), np.random.random(), torch.rand(3))
    assert actual_random[:2] == expected_random[:2]
    torch.testing.assert_close(actual_random[2], expected_random[2], rtol=0, atol=0)

    state = explain.create_almost_clear_position()
    mask = sj.actions(state).astype(np.float32)
    batch = train_utils.game_data_to_training_batch(
        [
            explain_game_point(state, mask),
            explain_game_point(state, mask),
        ]
    )
    replay = SimpleNamespace(sample_batch=lambda batch_size: batch)
    first_optimizer = train.make_optimizer(model, 0.001)
    second_optimizer = train.make_optimizer(initial, 0.001)
    stats = train_utils.TrainingDiagnostics()
    checkpoint.restore_rng_state(rng)
    observed = train.train_steps(
        model, replay, 2, 3, first_optimizer, train_utils.base_loss, diagnostics=stats
    )
    observed_rng = torch.get_rng_state()
    checkpoint.restore_rng_state(rng)
    plain = train.train_steps(
        initial, replay, 2, 3, second_optimizer, train_utils.base_loss
    )
    assert observed == plain
    torch.testing.assert_close(model.state_dict(), initial.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(
        first_optimizer.state_dict(), second_optimizer.state_dict(), rtol=0, atol=0
    )
    torch.testing.assert_close(observed_rng, torch.get_rng_state(), rtol=0, atol=0)
    assert stats.summary()["policy/all/positions"] == 6


def explain_game_point(state, mask):
    from skyjo.play import GameDataPoint

    return GameDataPoint(
        state,
        None,
        {"value": np.array([1.0, 0.0], dtype=np.float32), "policy": mask / mask.sum()},
    )


def test_progress_is_timed_including_idle_intervals_and_final_completion(
    monkeypatch, caplog
):
    now = [0.0]
    monkeypatch.setattr(experiment_training.time, "perf_counter", lambda: now[0])
    progress = experiment_training.GenerationProgress(8, interval=30)
    with caplog.at_level(logging.INFO):
        now[0] = 5
        progress.games = 2
        progress.decisions = 200
        progress.report()
        assert not caplog.records
        assert progress.wait_seconds() == 25
        now[0] = 30
        progress.report()
        now[0] = 60
        progress.report()  # No newly completed task: still report the stall.
        now[0] = 61
        progress.games = 8
        progress.decisions = 800
        progress.report(final=True)
    assert len(caplog.records) == 3
    assert "2/8 games" in caplog.messages[0]
    assert "0.067 games/s" in caplog.messages[0]
    assert "6.7 decisions/s" in caplog.messages[0]
    assert "ETA 90.0s" in caplog.messages[0]
    assert "complete" in caplog.messages[-1]


def test_default_progress_reports_only_completion_even_after_long_idle(
    monkeypatch, caplog
):
    now = [0.0]
    monkeypatch.setattr(experiment_training.time, "perf_counter", lambda: now[0])
    config = experiment_training.ObservationConfig()
    progress = experiment_training.GenerationProgress(8, config.progress_interval_seconds)
    with caplog.at_level(logging.INFO):
        for elapsed, games in ((300, 0), (600, 4), (900, 4)):
            now[0] = elapsed
            progress.games = games
            progress.report()
            assert progress.wait_seconds() is None
        assert not caplog.records
        now[0] = 1000
        progress.games = 8
        progress.decisions = 800
        progress.report(final=True)
    assert len(caplog.records) == 1
    assert "8/8 games in 1000.0s" in caplog.messages[0]
    assert "0.008 games/s" in caplog.messages[0]
    assert "0.8 decisions/s" in caplog.messages[0]
    assert "complete" in caplog.messages[0]


@pytest.mark.parametrize(
    "interval,expected", [(0, []), (5, [0, 5, 10, 12]), (1, list(range(13)))]
)
def test_concept_schedule_includes_boundaries_without_duplicates(interval, expected):
    config = experiment_training.ObservationConfig(concept_interval=interval)
    assert [i for i in range(13) if config.concepts_due(i, 12)] == expected
