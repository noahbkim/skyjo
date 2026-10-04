from __future__ import annotations

import numpy as np
import torch

from skyjo import batches, buffer, observations, play, selfplay_training
from skyjo import game as sj


def test_torch_worker_uses_bounded_cpu_parallelism(monkeypatch):
    configured_threads = []
    configured_interop_threads = []
    monkeypatch.setattr(torch, "set_num_threads", configured_threads.append)
    monkeypatch.setattr(
        torch,
        "set_num_interop_threads",
        configured_interop_threads.append,
    )

    selfplay_training.configure_torch_worker(1)

    assert configured_threads == [1]
    assert configured_interop_threads == [1]


def test_game_seeds_are_stable_and_distinct():
    seed_a = selfplay_training.derive_game_seed(
        7, 12, selfplay_training.PLAY_SEED_STREAM
    )
    seed_b = selfplay_training.derive_game_seed(
        7, 12, selfplay_training.PLAY_SEED_STREAM
    )
    target_seed = selfplay_training.derive_game_seed(
        7, 13, selfplay_training.PLAY_SEED_STREAM
    )

    assert seed_a == seed_b
    assert 0 <= seed_a <= np.iinfo(np.uint32).max
    assert seed_a != target_seed


def test_real_games_are_independent_of_task_batching():
    torch.manual_seed(1)
    model_settings = {
        "embedding_dimensions": 8,
        "global_state_embedding_dimensions": 16,
        "num_heads": 1,
    }
    model = selfplay_training.models.build(model_settings, players=2, device="cpu")
    player_config = selfplay_training.player.ModelPlayerConfig(
        action_softmax_temperature=1.0,
        mcts_iterations=1,
        mcts_dirichlet_epsilon=0.25,
        mcts_after_state_evaluate_all_children=False,
    )
    common = {
        "model_settings": model_settings,
        "model_state_dict": model.state_dict(),
        "model_player_config": player_config,
        "players": 2,
        "run_seed": 31,
    }
    one_task = selfplay_training.play_games_locally(
        **common,
        number_of_games=2,
        first_game_index=20,
    )
    multiple_tasks = [
        *selfplay_training.play_games_locally(
            **common, number_of_games=1, first_game_index=20
        ),
        *selfplay_training.play_games_locally(
            **common, number_of_games=1, first_game_index=21
        ),
    ]
    for left, right in zip(one_task, multiple_tasks, strict=True):
        assert left.global_game_index == right.global_game_index
        assert left.play_seed == right.play_seed
        assert len(left.result.rounds) == len(right.result.rounds)
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


def make_buffer() -> buffer.ReplayBuffer:
    return buffer.ReplayBuffer(
        max_size=8,
        spatial_input_shape=(2, sj.ROW_COUNT, sj.COLUMN_COUNT, sj.FINGER_SIZE),
        non_spatial_input_shape=observations.get_non_spatial_input_shape(2),
        action_mask_shape=(sj.MASK_SIZE,),
    )


def test_fresh_run_can_seed_buffer_without_overwriting_source(tmp_path):
    source_path = tmp_path / "source" / "dataset"
    source = make_buffer()
    state = sj.new(players=2, top=0)
    action_mask = sj.actions(state).astype(np.float32)
    source.add(
        state,
        {
            batches.VALUE_TARGET_NAME: np.array([1.0, 0.0], dtype=np.float32),
            batches.POLICY_TARGET_NAME: action_mask / action_mask.sum(),
        },
        game_index=12,
    )
    source.save(source_path)
    destination_path = tmp_path / "destination" / "dataset"
    config = buffer.Config(
        max_size=8,
        spatial_input_shape=(2, sj.ROW_COUNT, sj.COLUMN_COUNT, sj.FINGER_SIZE),
        non_spatial_input_shape=observations.get_non_spatial_input_shape(2),
        action_mask_shape=(sj.MASK_SIZE,),
        path=destination_path,
    )

    seeded = selfplay_training.initialize_training_data_buffer(config, source_path)

    assert seeded.game_indices == (12,)
    assert seeded.path == destination_path
    assert not destination_path.exists()
    assert source_path.exists()


def test_target_generation_is_sorted_and_does_not_need_randomness(monkeypatch):
    state = sj.new(players=2, top=0)
    action_mask = sj.actions(state).astype(np.float32)
    policy = action_mask / action_mask.sum()

    def fake_conversion(result):
        marker = float(result)
        data = [
            play.GameDataPoint(
                state,
                None,
                {
                    batches.VALUE_TARGET_NAME: np.array(
                        [marker, -marker], dtype=np.float32
                    ),
                    batches.POLICY_TARGET_NAME: policy,
                },
            )
        ]
        return data, object()

    monkeypatch.setattr(
        selfplay_training.play,
        "game_result_to_game_data",
        fake_conversion,
    )
    generated = [
        selfplay_training.GeneratedGame(
            global_game_index=index,
            play_seed=selfplay_training.derive_game_seed(
                9, index, selfplay_training.PLAY_SEED_STREAM
            ),
            result=index,
        )
        for index in range(3)
    ]

    ordered_buffer = make_buffer()
    selfplay_training.add_generated_games_to_buffer(
        generated,
        ordered_buffer,
    )
    reversed_buffer = make_buffer()
    selfplay_training.add_generated_games_to_buffer(
        reversed(generated),
        reversed_buffer,
    )

    assert ordered_buffer.game_indices == (0, 1, 2)
    assert reversed_buffer.game_indices == (0, 1, 2)
    assert np.array_equal(
        ordered_buffer.ordered_batch().targets["value"],
        reversed_buffer.ordered_batch().targets["value"],
    )


def test_generation_collects_out_of_order_completions_and_restores_game_order():
    from types import SimpleNamespace

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
                            selfplay_training.GeneratedGame(
                                index,
                                index,
                                SimpleNamespace(
                                    rounds=[SimpleNamespace(history=[None] * 3)]
                                ),
                            )
                        ]
                    )

    result = selfplay_training.generate_iteration(
        CompletingPool(),
        total_games=3,
        games_per_task=1,
        first_game_index=10,
        worker_kwargs={},
    )
    assert [game.global_game_index for game in result] == [10, 11, 12]


def test_generation_propagates_later_worker_failure_without_waiting_for_first():
    import pytest

    class FailingPool:
        def apply_async(self, worker, *, kwds, callback, error_callback):
            if kwds["first_game_index"] == 1:
                error_callback(RuntimeError("worker failed"))
            # Task zero never completes.

    with pytest.raises(RuntimeError, match="worker failed"):
        selfplay_training.generate_iteration(
            FailingPool(),
            total_games=2,
            games_per_task=1,
            first_game_index=0,
            worker_kwargs={},
        )
