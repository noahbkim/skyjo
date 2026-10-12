"""Game identity, worker scheduling, and replay ingestion stay deterministic."""

import dataclasses
import multiprocessing
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_full_game import completed_round

from skyjo.engine import game as sj
from skyjo.experiments import generation, selfplay_training
from skyjo.experiments.contestants import ContestantConfig
from skyjo.experiments.training_setup import initialize_training_data_buffer
from skyjo.learning import batches, buffer, models, observations, replay_io
from skyjo.search.mcts import SearchConfig
from skyjo.simulation import play
from skyjo.simulation.jobs import GeneratedGame, derive_game_seed


def test_game_seeds_are_stable_and_separate_games_and_streams():
    seed = derive_game_seed(7, 12)
    assert seed == derive_game_seed(7, 12)
    assert 0 <= seed <= np.iinfo(np.uint32).max
    assert len({seed, derive_game_seed(7, 13), derive_game_seed(7, 12, 1)}) == 3


def _worker_thread_counts():
    return torch.get_num_threads(), torch.get_num_interop_threads()


def _assert_same_games(expected, actual):
    for left, right in zip(expected, actual, strict=True):
        assert (left.global_game_index, left.play_seed) == (
            right.global_game_index,
            right.play_seed,
        )
        for left_round, right_round in zip(
            left.result.rounds, right.result.rounds, strict=True
        ):
            assert left_round.cumulative_scores == right_round.cumulative_scores
            for a, b in zip(left_round.history, right_round.history, strict=True):
                assert sj.hash_skyjo(a.state) == sj.hash_skyjo(b.state)
                assert a.action == b.action
                np.testing.assert_array_equal(
                    a.action_probabilities, b.action_probabilities
                )


def test_real_games_are_independent_of_task_grouping_and_worker_count():
    torch.set_num_threads(1)
    torch.manual_seed(1)
    model_settings = {
        "embedding_dimensions": 4,
        "global_state_embedding_dimensions": 8,
        "num_heads": 1,
    }
    model = models.build(model_settings, players=2, device="cpu")
    common = {
        "model_settings": model_settings,
        "model_state_dict": model.state_dict(),
        "model_player_config": ContestantConfig(
            iterations=1, temperature=1, search=SearchConfig(dirichlet_epsilon=0.25)
        ),
        "players": 2,
        "run_seed": 31,
    }
    expected = generation.play_games_locally(
        **common, number_of_games=2, first_game_index=20
    )
    split = [
        game
        for index in (20, 21)
        for game in generation.play_games_locally(
            **common, number_of_games=1, first_game_index=index
        )
    ]
    _assert_same_games(expected, split)
    for workers in (1, 2):
        with multiprocessing.get_context("spawn").Pool(
            workers, initializer=generation.configure_torch_worker, initargs=(1,)
        ) as pool:
            assert pool.apply(_worker_thread_counts) == (1, 1)
            actual = generation.generate_iteration(
                pool,
                total_games=2,
                games_per_task=1,
                first_game_index=20,
                worker_kwargs=common,
            )
        _assert_same_games(expected, actual)


def make_buffer():
    return buffer.ReplayBuffer(
        max_size=8,
        spatial_input_shape=(2, sj.ROW_COUNT, sj.COLUMN_COUNT, sj.FINGER_SIZE),
        non_spatial_input_shape=observations.get_non_spatial_input_shape(2),
        action_mask_shape=(sj.MASK_SIZE,),
    )


def test_fresh_replay_import_does_not_overwrite_its_source(tmp_path):
    state = sj.new(players=2, top=0)
    inputs = batches.states_to_batch([state])
    batch = dataclasses.replace(
        inputs,
        targets={
            "value": np.array([[1.0, 0.0]], dtype=np.float32),
            "policy": inputs.action_masks.astype(np.float32)
            / inputs.action_masks.sum(axis=1, keepdims=True),
        },
    )
    source = make_buffer()
    source.append(batch, buffer.GameProvenance(12))
    source_path = replay_io.save(source, tmp_path / "source")
    original = {
        path.name: path.read_bytes() for path in source_path.iterdir() if path.is_file()
    }
    config = buffer.Config(
        8,
        inputs.spatial_inputs.shape[1:],
        inputs.non_spatial_inputs.shape[1:],
        inputs.action_masks.shape[1:],
    )
    seeded = initialize_training_data_buffer(config, source_path)
    assert seeded.game_indices == (12,)
    seeded.append(batch, buffer.GameProvenance(13))
    destination = replay_io.save(seeded, tmp_path / "destination")
    assert replay_io.load(destination).game_indices == (12, 13)
    assert original == {
        path.name: path.read_bytes() for path in source_path.iterdir() if path.is_file()
    }


def _observed_game(index):
    completed = completed_round(((1, 2, 3), (4, 5, 6)), (95 + index, 95))
    state = sj.apply_action(dataclasses.replace(completed, countdown=2), sj.MASK_TAKE)
    probabilities = np.zeros(sj.MASK_SIZE, dtype=np.float32)
    probabilities[sj.MASK_REPLACE] = 1
    final = sj.apply_action(state, sj.MASK_REPLACE)
    result = play.GameResult(
        (
            play.RoundResult(
                [
                    play.RoundHistoryEntry(state, sj.MASK_REPLACE, probabilities),
                    play.RoundHistoryEntry(final, None, None),
                ],
                tuple(sj.get_fixed_perspective_round_scores(final)),
                tuple(sj.get_fixed_perspective_game_scores(final)),
                sj.get_player(final),
            ),
        )
    )
    return GeneratedGame(index, derive_game_seed(9, index), result)


def test_target_ingestion_sorts_games_and_uses_observed_results(monkeypatch):
    generated = [_observed_game(index) for index in range(3)]

    def forbidden(*args, **kwargs):
        raise AssertionError("Observed labels must not replay game transitions")

    monkeypatch.setattr(sj, "apply_action", forbidden)
    ordered, reversed_ = make_buffer(), make_buffer()
    selfplay_training.add_generated_games_to_buffer(generated, ordered)
    selfplay_training.add_generated_games_to_buffer(reversed(generated), reversed_)
    assert ordered.game_indices == reversed_.game_indices == (0, 1, 2)
    for name, expected in ordered.ordered_batch().targets.items():
        np.testing.assert_array_equal(expected, reversed_.ordered_batch().targets[name])


def test_generation_restores_order_after_out_of_order_completions():
    class CompletingPool:
        def __init__(self):
            self.tasks = []

        def apply_async(self, worker, *, kwds, callback, error_callback):
            self.tasks.append((kwds, callback))
            if len(self.tasks) == 3:
                for settings, complete in reversed(self.tasks):
                    index = settings["first_game_index"]
                    complete(
                        [
                            GeneratedGame(
                                index,
                                index,
                                SimpleNamespace(
                                    rounds=[SimpleNamespace(history=[None] * 3)]
                                ),
                            )
                        ]
                    )

    result = generation.generate_iteration(
        CompletingPool(),
        total_games=3,
        games_per_task=1,
        first_game_index=10,
        worker_kwargs={},
    )
    assert [game.global_game_index for game in result] == [10, 11, 12]


def test_generation_reports_later_worker_failure_without_waiting_for_first():
    class FailingPool:
        def apply_async(self, worker, *, kwds, callback, error_callback):
            if kwds["first_game_index"] == 1:
                error_callback(RuntimeError("worker failed"))
            # Task zero never completes.

    with pytest.raises(RuntimeError, match="worker failed"):
        generation.generate_iteration(
            FailingPool(),
            total_games=2,
            games_per_task=1,
            first_game_index=0,
            worker_kwargs={},
        )
