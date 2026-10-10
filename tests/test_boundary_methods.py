"""Protect the engine/perspective and held-out-data contracts of the comparison."""

import dataclasses
import json
import random

import numpy as np
import pytest
import torch

from skyjo.engine import game as sj
from skyjo.learning import observations
from skyjo.search.evaluator import Prediction
from skyjo.learning import checkpoint as checkpoint_io
from skyjo.learning.boundary_data import load_boundaries
from skyjo.experiments.boundary_methods import (
    next_deal,
    next_deal_predictions,
    validate_holdout,
)


@pytest.mark.parametrize("players,starter", [(2, 0), (2, 1), (3, 2)])
def test_next_deal_matches_engine_boundary_without_mutation(players, starter):
    # Cleared boards contribute zero, making the starting totals explicit.
    completed = sj.new(players=players, top=sj.CARD_P12)
    completed.table[:players] = 0
    completed.table[:players, :, :, sj.FINGER_CLEARED] = 1
    completed.game[sj.GAME_SCORES : sj.GAME_SCORES + players] = (9, 42, 23)[:players]
    completed = dataclasses.replace(completed, turn=players * 4 + starter, countdown=0)
    assert sj.validate(completed)
    before = sj.hash_skyjo(completed)
    totals = sj.get_game_scores(completed)
    original_totals = totals.copy()

    expected = sj.start_next_round(completed, rng=random.Random(19))
    actual = next_deal(totals, starter, random.Random(19))

    assert sj.hash_skyjo(actual) == sj.hash_skyjo(expected)
    assert sj.get_player(actual) == starter
    np.testing.assert_array_equal(
        observations.get_non_spatial_state_numpy(actual),
        observations.get_non_spatial_state_numpy(expected),
    )
    np.testing.assert_array_equal(
        observations.get_spatial_state_numpy(actual),
        observations.get_spatial_state_numpy(expected),
    )
    np.testing.assert_array_equal(sj.actions(actual), sj.actions(expected))
    np.testing.assert_array_equal(totals, original_totals)
    assert sj.hash_skyjo(completed) == before
    assert not np.shares_memory(actual.game, totals)


class RecordingPredictor:
    """Card-dependent probabilities expressed in the input's player order."""

    def __init__(self):
        self.states = []
        self.values = []

    def evaluate(self, states):
        predictions = []
        for state in states:
            # Every fresh player has one visible card. Unequal seat weights make
            # an accidental rotation observable even when cards happen to match.
            visible = sj.get_table(state)[:, :, :, : sj.CARD_SIZE]
            weights = (visible * np.arange(1, sj.CARD_SIZE + 1)).sum(axis=(1, 2, 3))
            weights += np.arange(1, state.players + 1) * 17
            value = (weights / weights.sum()).astype(np.float32)
            self.states.append(state)
            self.values.append(value)
            predictions.append(
                Prediction(
                    value=np.roll(value, sj.get_player(state)),
                    policy=np.zeros(sj.MASK_SIZE, dtype=np.float32),
                )
            )
        return predictions


def test_deal_means_use_nested_samples_and_preserve_starter_perspective():
    scores = np.array([[14, 63, 32], [72, 28, 49]], dtype=np.float32)
    starters = np.array([2, 1])
    samples = {}
    repeats = 2
    for count in (1, 3):
        inference = RecordingPredictor()
        actual, _ = next_deal_predictions(
            scores,
            starters,
            inference,
            sample_counts=(count,),
            repeats=repeats,
            seed=91,
            batch_size=2,
        )
        estimates = actual[f"next_deal_{count}"]
        assert estimates.shape == (repeats, len(scores), 3)
        samples[count] = []
        for row, (totals, starter) in enumerate(zip(scores, starters, strict=True)):
            selected = [
                (state, value)
                for state, value in zip(inference.states, inference.values, strict=True)
                if np.array_equal(sj.get_game_scores(state), totals)
            ]
            assert len(selected) == repeats * count
            assert all(sj.get_player(state) == starter for state, _ in selected)
            scalar_values = np.array([value for _, value in selected]).reshape(
                repeats, count, 3
            )
            np.testing.assert_allclose(
                estimates[:, row],
                scalar_values.mean(axis=1),
                atol=1e-7,
                rtol=0,
            )
            hashes = np.array([sj.hash_skyjo(state) for state, _ in selected])
            samples[count].append(hashes.reshape(repeats, count))

    for row in range(len(scores)):
        np.testing.assert_array_equal(samples[1][row][:, 0], samples[3][row][:, 0])
    # Sampling does not depend on inference chunking, and seeds are reproducible.
    repeated, _ = next_deal_predictions(
        scores,
        starters,
        RecordingPredictor(),
        sample_counts=(3,),
        repeats=repeats,
        seed=91,
        batch_size=7,
    )
    np.testing.assert_array_equal(repeated["next_deal_3"], estimates)


@pytest.mark.parametrize(
    "violation",
    [None, "model_saw_game", "wrong_generator", "score_overlap", "changed_log"],
)
def test_holdout_requires_unseen_games_and_matching_unchanged_provenance(
    tmp_path, violation
):
    checkpoint = tmp_path / "model.pth"
    checkpoint_io.save_checkpoint(
        checkpoint,
        model=torch.nn.Linear(1, 1),
        optimizer=None,
        configuration={},
        progress=checkpoint_io.TrainingProgress(generated_games=10, iteration=3),
    )
    game_index = 9 if violation == "model_saw_game" else 10
    generator = tmp_path / "other.pth" if violation == "wrong_generator" else checkpoint
    records = []
    for number, (scores, cumulative) in enumerate(
        [([22, 40], [22, 40]), ([79, 40], [101, 80])], start=1
    ):
        records.append(
            {
                "run_id": "evaluation",
                "game_index": game_index,
                "iteration": 4,
                "play_seed": 5,
                "round_number": number,
                "scores": scores,
                "cumulative_scores": cumulative,
                "ending_player": 1,
                "checkpoint_path": str(generator),
                "partial_start": False,
            }
        )
    source = tmp_path / "rounds.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in records))
    dataset = load_boundaries([source])
    boundary_run = tmp_path / "boundary"
    (boundary_run / "data").mkdir(parents=True)
    training_ids = (
        dataset.game_ids if violation == "score_overlap" else np.array(['["old",0]'])
    )
    np.savez(boundary_run / "data/boundaries.npz", game_ids=training_ids)
    if violation == "changed_log":
        source.write_text(source.read_text() + "\n")

    if violation is not None:
        with pytest.raises(ValueError):
            validate_holdout(dataset, boundary_run, checkpoint, [source])
    else:
        audit = validate_holdout(dataset, boundary_run, checkpoint, [source])
        assert audit["next_starters"] == [1]
        assert audit["evaluation_game_index_min"] == 10
        assert audit["recorded_generator_matches_evaluator"] is True
        assert audit["boundary_dataset_game_overlap"] == 0
