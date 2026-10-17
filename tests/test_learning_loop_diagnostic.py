"""Protect target reconstruction, exact scoring, and continuation perspective."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from skyjo.engine import game
from skyjo.experiments import learning_loop_diagnostic as diagnostic
from skyjo.experiments.search_positions import load_suite
from skyjo.experiments.search_strategy_benchmark import load_cases
from skyjo.search import runner
from skyjo.search.evaluator import Prediction

ROOT = Path(__file__).resolve().parents[1]


class UniformEvaluator:
    def evaluate(self, states):
        return [
            Prediction(np.array([0.5, 0.5]), game.actions(s) / game.actions(s).sum())
            for s in states
        ]


def test_traced_visit_targets_match_the_training_runner(monkeypatch):
    evaluator = UniformEvaluator()
    monkeypatch.setattr(diagnostic, "_plan", {"budgets": [32]})
    monkeypatch.setattr(
        diagnostic, "agent", lambda name: SimpleNamespace(evaluator=evaluator)
    )
    for position in load_suite(ROOT / "data/search-positions"):
        position = diagnostic.dataclasses.replace(position, tags=("tactical",))
        row = diagnostic.trial((position, {}, "test", "base", 32, 0))
        searched = runner.search(
            position.state,
            evaluator,
            32,
            config=diagnostic.BASE,
            rng=np.random.default_rng(0),
        )
        np.testing.assert_allclose(row["target"], searched.training_policy, atol=1e-7)
        assert sum(row["target"]) == pytest.approx(1)


def test_oracle_measures_distribution_quality_separately_from_choice():
    result = diagnostic.oracle_metrics(
        {0: 0.2, 1: 0.8}, np.array([0.9, 0.1]), np.array([0.25, 0.75]), 1, 0
    )
    assert result == pytest.approx(
        {
            "policy_expected_regret": 0.54,
            "target_expected_regret": 0.15,
            "choice_regret": 0,
            "q_choice_regret": 0.6,
        }
    )


def test_forced_terminal_action_uses_original_fixed_seat_and_does_not_restart(
    monkeypatch,
):
    _, cases = load_cases(ROOT / "configs/benchmarks/search_positions.json")
    case = next(c for c in cases if c.name == "terminal-183111")
    assert game.get_player(case.state) == 1
    # Every outcome ends the game; the original actor loses even though the
    # perspective rotates at the terminal transition.
    prepared = SimpleNamespace(player=lambda rng: None)
    monkeypatch.setattr(
        diagnostic.play, "play_game", lambda *a, **k: pytest.fail("Game already ended")
    )
    monkeypatch.setattr(
        diagnostic.play,
        "play_round",
        lambda *a, **k: pytest.fail("Round already ended"),
    )
    result = diagnostic.continue_action(case.state, 9, prepared, 123)
    assert result["win_credit"] == 0
    assert result["game_margin"] < 0
