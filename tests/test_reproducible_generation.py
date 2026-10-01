from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from skyjo import buffer, play, skynet, train_utils
from skyjo import game as sj

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import distributed_main  # noqa: E402


def test_torch_worker_uses_bounded_cpu_parallelism(monkeypatch):
    configured_threads = []
    configured_interop_threads = []
    monkeypatch.setattr(torch, "set_num_threads", configured_threads.append)
    monkeypatch.setattr(
        torch,
        "set_num_interop_threads",
        configured_interop_threads.append,
    )

    distributed_main.configure_torch_worker(1)

    assert configured_threads == [1]
    assert configured_interop_threads == [1]


def test_game_seeds_are_stable_and_distinct():
    seed_a = distributed_main.derive_game_seed(
        7, 12, distributed_main.PLAY_SEED_STREAM
    )
    seed_b = distributed_main.derive_game_seed(
        7, 12, distributed_main.PLAY_SEED_STREAM
    )
    target_seed = distributed_main.derive_game_seed(
        7, 13, distributed_main.PLAY_SEED_STREAM
    )

    assert seed_a == seed_b
    assert 0 <= seed_a <= np.iinfo(np.uint32).max
    assert seed_a != target_seed


def test_real_games_are_independent_of_task_batching():
    torch.manual_seed(1)
    model_kwargs = {
        "embedding_dimensions": 8, "global_state_embedding_dimensions": 16, "num_heads": 1,
    }
    model = distributed_main.build_local_model(skynet.EquivariantSkyNet, model_kwargs, 2)
    player_config = distributed_main.player.ModelPlayerConfig(
        action_softmax_temperature=1.0, mcts_iterations=1,
        mcts_dirichlet_epsilon=0.25, mcts_after_state_evaluate_all_children=False,
    )
    common = {
        "model_callable": skynet.EquivariantSkyNet,
        "model_kwargs": model_kwargs,
        "model_state_dict": model.state_dict(),
        "model_player_config": player_config,
        "players": 2,
        "run_seed": 31,
    }
    one_task = distributed_main.play_games_locally(
        **common, number_of_games=2, first_game_index=20,
    )
    multiple_tasks = [
        *distributed_main.play_games_locally(**common, number_of_games=1, first_game_index=20),
        *distributed_main.play_games_locally(**common, number_of_games=1, first_game_index=21),
    ]
    for left, right in zip(one_task, multiple_tasks, strict=True):
        assert left.global_game_index == right.global_game_index
        assert left.play_seed == right.play_seed
        assert len(left.result.rounds) == len(right.result.rounds)
        for left_round, right_round in zip(left.result.rounds, right.result.rounds, strict=True):
            assert left_round.cumulative_scores == right_round.cumulative_scores
            for a, b in zip(left_round.history, right_round.history, strict=True):
                assert sj.hash_skyjo(a.state) == sj.hash_skyjo(b.state)
                assert a.action == b.action
                np.testing.assert_array_equal(a.action_probabilities, b.action_probabilities)


def make_buffer() -> buffer.ReplayBuffer:
    return buffer.ReplayBuffer(
        max_size=8,
        spatial_input_shape=(2, sj.ROW_COUNT, sj.COLUMN_COUNT, sj.FINGER_SIZE),
        non_spatial_input_shape=skynet.get_non_spatial_input_shape(2),
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
            train_utils.VALUE_TARGET_NAME: np.array([1.0, 0.0], dtype=np.float32),
            train_utils.POLICY_TARGET_NAME: action_mask / action_mask.sum(),
        },
        game_index=12,
    )
    source.save(source_path)
    destination_path = tmp_path / "destination" / "dataset"
    config = buffer.Config(
        max_size=8,
        spatial_input_shape=(2, sj.ROW_COUNT, sj.COLUMN_COUNT, sj.FINGER_SIZE),
        non_spatial_input_shape=skynet.get_non_spatial_input_shape(2),
        action_mask_shape=(sj.MASK_SIZE,),
        path=destination_path,
    )

    seeded = distributed_main.initialize_training_data_buffer(config, source_path)

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
                    train_utils.VALUE_TARGET_NAME: np.array(
                        [marker, -marker], dtype=np.float32
                    ),
                    train_utils.POLICY_TARGET_NAME: policy,
                },
            )
        ]
        return data, object()

    monkeypatch.setattr(
        distributed_main.play,
        "game_result_to_game_data",
        fake_conversion,
    )
    generated = [
        distributed_main.GeneratedGame(
            global_game_index=index,
            play_seed=distributed_main.derive_game_seed(
                9, index, distributed_main.PLAY_SEED_STREAM
            ),
            result=index,
        )
        for index in range(3)
    ]

    ordered_buffer = make_buffer()
    distributed_main.add_generated_games_to_buffer(
        generated,
        ordered_buffer,
    )
    reversed_buffer = make_buffer()
    distributed_main.add_generated_games_to_buffer(
        reversed(generated),
        reversed_buffer,
    )

    assert ordered_buffer.game_indices == (0, 1, 2)
    assert reversed_buffer.game_indices == (0, 1, 2)
    assert np.array_equal(
        ordered_buffer.ordered_batch().value_targets,
        reversed_buffer.ordered_batch().value_targets,
    )
