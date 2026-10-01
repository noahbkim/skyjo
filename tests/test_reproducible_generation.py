from __future__ import annotations

import random
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from skyjo import buffer, play, skynet, train_utils
from skyjo import game as sj

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import distributed_main  # noqa: E402


@pytest.mark.parametrize("rounds_per_task", [1, 2, 4])
@pytest.mark.parametrize("wins_per_pair,passed", [((1, 1), False), ((2, 0), True)])
def test_faceoff_batching_preserves_pair_seeds_and_strict_win_rule(
    rounds_per_task, wins_per_pair, passed
):
    seeds = []

    class Pool:
        def apply_async(self, function, args):
            # Worker input defines a consecutive range of paired-game seeds.
            count, first_seed = args[5:7]
            seeds.extend(range(first_seed, first_seed + count))
            return types.SimpleNamespace(
                get=lambda: tuple(count * wins for wins in wins_per_pair)
            )

    result = distributed_main.validate_model_faceoff(
        pool=Pool(),
        model=types.SimpleNamespace(state_dict=lambda: {}),
        players=2,
        previous_model_state_dict={},
        model_callable=None,
        model_kwargs={},
        rounds=10,
        rounds_per_task=rounds_per_task,
        model_player_config=None,
    )
    assert seeds == list(range(10))
    assert result == {
        "candidate_wins": 10 * wins_per_pair[0],
        "champion_wins": 10 * wins_per_pair[1],
        "passed": passed,
    }


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


def test_game_seeds_are_stable_distinct_streams():
    seed_a = distributed_main.derive_game_seed(
        7, 12, distributed_main.PLAY_SEED_STREAM
    )
    seed_b = distributed_main.derive_game_seed(
        7, 12, distributed_main.PLAY_SEED_STREAM
    )
    target_seed = distributed_main.derive_game_seed(
        7, 12, distributed_main.TARGET_SEED_STREAM
    )

    assert seed_a == seed_b
    assert 0 <= seed_a <= np.iinfo(np.uint32).max
    assert seed_a != target_seed


def test_games_are_independent_of_task_batching(monkeypatch):
    monkeypatch.setattr(
        distributed_main,
        "build_local_model",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        distributed_main.predictor,
        "LocalPredictorClient",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        distributed_main.player,
        "ModelPlayer",
        lambda *args, **kwargs: object(),
    )

    def fake_distributed_play(players, start_state=None, number_of_games=1):
        del players, start_state
        assert number_of_games == 1
        return [
            [
                (
                    random.random(),
                    float(np.random.random()),
                    float(torch.rand(1).item()),
                )
            ]
        ]

    monkeypatch.setattr(
        distributed_main.play,
        "distributed_play",
        fake_distributed_play,
    )
    player_config = types.SimpleNamespace(kwargs=lambda: {})
    common_arguments = {
        "model_callable": object(),
        "model_state_dict": {},
        "model_kwargs": {},
        "model_player_config": player_config,
        "players": 2,
        "run_seed": 31,
    }

    one_task = distributed_main.play_games_locally(
        **common_arguments,
        number_of_games=3,
        first_game_index=20,
    )
    multiple_tasks = [
        *distributed_main.play_games_locally(
            **common_arguments,
            number_of_games=1,
            first_game_index=20,
        ),
        *distributed_main.play_games_locally(
            **common_arguments,
            number_of_games=2,
            first_game_index=21,
        ),
    ]

    assert one_task == multiple_tasks


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


def test_target_generation_is_seeded_and_sorted_before_buffering(monkeypatch):
    state = sj.new(players=2, top=0)
    action_mask = sj.actions(state).astype(np.float32)
    policy = action_mask / action_mask.sum()

    def fake_conversion(history, terminal_rollouts, include_future_clear_target=True):
        del history, terminal_rollouts, include_future_clear_target
        marker = (
            random.random()
            + float(np.random.random())
            + float(torch.rand(1).item())
        )
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
        "game_history_to_game_data",
        fake_conversion,
    )
    generated = [
        distributed_main.GeneratedGameHistory(
            global_game_index=index,
            play_seed=distributed_main.derive_game_seed(
                9, index, distributed_main.PLAY_SEED_STREAM
            ),
            target_seed=distributed_main.derive_game_seed(
                9, index, distributed_main.TARGET_SEED_STREAM
            ),
            history=[],
        )
        for index in range(3)
    ]

    ordered_buffer = make_buffer()
    distributed_main.add_generated_games_to_buffer(
        generated,
        ordered_buffer,
        outcome_rollouts=2,
    )
    reversed_buffer = make_buffer()
    distributed_main.add_generated_games_to_buffer(
        reversed(generated),
        reversed_buffer,
        outcome_rollouts=2,
    )

    assert ordered_buffer.game_indices == (0, 1, 2)
    assert reversed_buffer.game_indices == (0, 1, 2)
    assert np.array_equal(
        ordered_buffer.ordered_batch().value_targets,
        reversed_buffer.ordered_batch().value_targets,
    )
