"""Random caps select replay rows without changing full-game outcomes or IDs."""

import dataclasses
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from helpers import NaiveQuickFinishPlayer

from skyjo.engine import game
from skyjo.experiments import experiment_config, selfplay_training
from skyjo.experiments.contestants import ContestantConfig, PreparedContestant
from skyjo.experiments.playout_budget import RandomizedBudgetPlayer
from skyjo.learning import buffer, targets
from skyjo.search.configuration import SearchConfig
from skyjo.simulation import play
from skyjo.simulation.decisions import PlayerDecision
from skyjo.simulation.jobs import GeneratedGame


def test_caps_noise_and_selection_use_an_independent_random_stream(monkeypatch):
    calls = []

    def player(prepared, search_rng):
        def decide(state, *, diagnostics):
            calls.append(prepared.config)
            # Changing search's RNG consumption cannot change the cap sequence.
            search_rng.random(prepared.config.iterations)
            mask = game.actions(state).astype(float)
            return PlayerDecision(mask / mask.sum())

        return SimpleNamespace(get_decision=decide)

    monkeypatch.setattr(PreparedContestant, "player", player)
    prepared = PreparedContestant(
        ContestantConfig(
            iterations=128, temperature=1, search=SearchConfig(dirichlet_epsilon=0.25)
        ),
        None,
    )
    agent = RandomizedBudgetPlayer(
        prepared,
        fast_iterations=16,
        full_probability=0.25,
        search_rng=np.random.default_rng(8),
        budget_rng=np.random.default_rng(0),
    )
    state = game.start_round(game.new(players=2), rng=np.random.default_rng(1))
    decisions = [agent.get_decision(state, diagnostics=False) for _ in range(4)]
    assert [d.trainable for d in decisions] == [False, False, True, True]
    assert [c.iterations for c in calls] == [16, 16, 128, 128]
    assert [c.temperature for c in calls] == [0, 0, 1, 1]
    assert [c.search.dirichlet_epsilon for c in calls] == [0, 0, 0.25, 0.25]


def test_only_selected_decisions_enter_replay_but_all_games_count(
    tmp_path, monkeypatch
):
    result = play.play_game(
        [NaiveQuickFinishPlayer()] * 2,
        environment_rng=np.random.default_rng(7),
        action_rng=np.random.default_rng(8),
    )
    original = targets.build_training_batch(result)
    selected = []
    rounds = []
    index = 0
    for round_result in result.rounds:
        history = []
        for entry in round_result.history[:-1]:
            keep = index % 4 == 0
            selected.append(keep)
            history.append(entry._replace(trainable=keep))
            index += 1
        rounds.append(
            dataclasses.replace(
                round_result, history=history + [round_result.history[-1]]
            )
        )
    filtered = dataclasses.replace(result, rounds=tuple(rounds))
    actual = targets.build_training_batch(filtered)
    for name in original.targets:
        np.testing.assert_array_equal(
            actual.targets[name], original.targets[name][selected]
        )
    empty = dataclasses.replace(
        result,
        rounds=tuple(
            dataclasses.replace(
                r, history=[e._replace(trainable=False) for e in r.history]
            )
            for r in result.rounds
        ),
    )
    assert len(targets.build_training_batch(empty)) == 0
    replay = buffer.ReplayBuffer(
        max_size=len(original) * 2,
        spatial_input_shape=original.spatial_inputs.shape[1:],
        non_spatial_input_shape=original.non_spatial_inputs.shape[1:],
        action_mask_shape=original.action_masks.shape[1:],
    )
    prepared = selfplay_training.prepare_iteration(
        [GeneratedGame(4, 41, filtered), GeneratedGame(5, 42, empty)], replay
    )
    assert len(prepared.games) == 2
    assert prepared.positions == 2 * len(original)
    assert prepared.training_positions == len(actual) == len(replay)
    assert prepared.training_games == replay.game_count == 1
    assert replay.game_indices == (4,)

    # A fresh run must also survive an iteration with no retained samples.
    monkeypatch.setattr(selfplay_training.mp, "Pool", lambda **kw: nullcontext(None))
    monkeypatch.setattr(
        selfplay_training,
        "generate_iteration",
        lambda pool, **kw: [GeneratedGame(0, 42, empty)],
    )
    root = Path(__file__).resolve().parents[1]
    config = experiment_config.load_configuration(root / "configs/smoke.toml")[1]
    config["budget"]["iterations"] = 1
    config["validation"]["concept_interval"] = 0
    path = tmp_path / "empty-selection.json"
    path.write_text(json.dumps(config))
    run = selfplay_training.launch(
        path, tmp_path / "runs", repository=root, allow_dirty=True
    )
    assert run.progress["generated_games"] == 1
    assert run.progress["optimizer_steps"] == 0
    assert run.checkpoint.path.is_file()


@pytest.mark.parametrize("fast,probability", [(0, 0.25), (16, 1), (32, 0.25), (16, 0)])
def test_invalid_randomized_budget_is_rejected_before_launch(
    tmp_path, fast, probability
):
    with pytest.raises(ValueError):
        experiment_config.resolve_configuration(
            {
                "search": {"iterations": 32},
                "selfplay": {
                    "fast_search_iterations": fast,
                    "full_search_probability": probability,
                },
            },
            base_directory=tmp_path,
        )
