from __future__ import annotations

import random
import sys
import types
from pathlib import Path

import numpy as np
import torch

from skyjo import buffer, play, train_utils
from skyjo import game as sj

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import distributed_main  # noqa: E402


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
        non_spatial_input_shape=(sj.GAME_SIZE,),
        action_mask_shape=(sj.MASK_SIZE,),
    )


def test_target_generation_is_seeded_and_sorted_before_buffering(monkeypatch):
    state = sj.new(players=2, top=0)
    action_mask = sj.actions(state).astype(np.float32)
    policy = action_mask / action_mask.sum()

    def fake_conversion(history, terminal_rollouts):
        del history, terminal_rollouts
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
