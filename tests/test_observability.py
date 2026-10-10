"""Numerical diagnostics, observational evaluation, and bounded progress reporting."""

import copy
import dataclasses
import logging
import random

import numpy as np
import pytest
import torch

from skyjo.learning import batches, buffer
from skyjo.learning import checkpoint
from skyjo.experiments import experiment_training
from skyjo.analytics import explain
from skyjo.learning import losses
from skyjo.learning import observations
from skyjo.learning import skynet
from skyjo.learning import train
from skyjo.engine import game as sj


def test_policy_diagnostics_weight_positions_and_ignore_masked_actions():
    logits = torch.full((4, sj.MASK_SIZE), 100.0, requires_grad=True)
    masks = torch.zeros_like(logits)
    # One example from every phase, each with two equiprobable legal choices.
    for i, pair in enumerate(((0, 1), (2, 3), (4, 16), (16, 17))):
        masks[i, list(pair)] = 1
    targets = masks / 2
    targets[1, 2:4] = torch.tensor([1.0, 0.0])
    stats = train.TrainingDiagnostics()
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
        non_spatial_input_shape=observations.get_non_spatial_input_shape(2),
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

    states = [
        explain.create_almost_clear_position(),
        explain.create_negative_clear_position(),
    ]
    inputs = batches.states_to_batch(states)
    mask = inputs.action_masks.astype(np.float32)
    batch = dataclasses.replace(
        inputs,
        targets={
            "value": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            "policy": mask / mask.sum(axis=1, keepdims=True),
        },
    )
    replay = buffer.ReplayBuffer(
        2,
        inputs.spatial_inputs.shape[1:],
        inputs.non_spatial_inputs.shape[1:],
        inputs.action_masks.shape[1:],
    )
    replay.append(batch, buffer.GameProvenance(0))
    observed_sampling, plain_sampling = (
        np.random.default_rng(8),
        np.random.default_rng(8),
    )
    first_optimizer = train.make_optimizer(model, 0.001)
    second_optimizer = train.make_optimizer(initial, 0.001)
    stats = train.TrainingDiagnostics()
    checkpoint.restore_rng_state(rng)
    observed = train.train_steps(
        model,
        replay,
        2,
        3,
        first_optimizer,
        losses.base_loss,
        diagnostics=stats,
        sampling_rng=observed_sampling,
    )
    observed_rng = torch.get_rng_state()
    checkpoint.restore_rng_state(rng)
    plain = train.train_steps(
        initial,
        replay,
        2,
        3,
        second_optimizer,
        losses.base_loss,
        sampling_rng=plain_sampling,
    )
    assert observed == plain
    assert observed_sampling.bit_generator.state == plain_sampling.bit_generator.state
    torch.testing.assert_close(model.state_dict(), initial.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(
        first_optimizer.state_dict(), second_optimizer.state_dict(), rtol=0, atol=0
    )
    torch.testing.assert_close(observed_rng, torch.get_rng_state(), rtol=0, atol=0)
    assert stats.summary()["policy/all/positions"] == 6


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
    progress = experiment_training.GenerationProgress(
        8, config.progress_interval_seconds
    )
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
