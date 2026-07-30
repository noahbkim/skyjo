from __future__ import annotations

import numpy as np

from skyjo import faceoff, player


def model_player_config() -> player.ModelPlayerConfig:
    return player.ModelPlayerConfig(
        action_softmax_temperature=0.0,
        mcts_iterations=4,
        mcts_dirichlet_epsilon=0.0,
        mcts_after_state_evaluate_all_children=False,
        mcts_terminal_state_initial_rollouts=1,
    )


def test_mcts_faceoff_uses_model_players_and_swaps_seats(monkeypatch) -> None:
    candidate = object()
    champion = object()
    monkeypatch.setattr(
        faceoff.predictor,
        "LocalPredictorClient",
        lambda model, max_batch_size: model,
    )
    monkeypatch.setattr(
        faceoff.player,
        "ModelPlayer",
        lambda model, **kwargs: model,
    )
    observed_seats = []

    def fake_game(players, start_state=None):
        del start_state
        observed_seats.append(tuple(players))
        return np.array([1.0, 0.0], dtype=np.float32), np.array([0, 1])

    monkeypatch.setattr(faceoff, "single_game_faceoff", fake_game)
    completed_games = []
    wins = faceoff.model_mcts_faceoff(
        candidate,
        champion,
        model_player_config(),
        paired_rounds=2,
        seed=10,
        game_completed_callback=lambda: completed_games.append(None),
    )
    assert observed_seats == [
        (candidate, champion),
        (champion, candidate),
        (candidate, champion),
        (champion, candidate),
    ]
    assert wins == (2, 2)
    assert len(completed_games) == 4


def test_promotion_requires_strict_majority(monkeypatch) -> None:
    config = faceoff.MCTSPromotionConfig(model_player_config(), paired_rounds=100)
    monkeypatch.setattr(faceoff, "model_mcts_faceoff", lambda *args, **kwargs: (100, 100))
    assert not faceoff.passes_mcts_promotion(object(), object(), config)
    monkeypatch.setattr(faceoff, "model_mcts_faceoff", lambda *args, **kwargs: (101, 99))
    assert faceoff.passes_mcts_promotion(object(), object(), config)
