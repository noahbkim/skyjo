from pathlib import Path

import numpy as np
import pytest

from skyjo.engine import game
from skyjo.experiments.search_strategy_benchmark import (
    Variant,
    exact_action_values,
    load_cases,
    recorded_selection,
    run_trial,
)
from skyjo.search.evaluator import Prediction
from skyjo.search.mcts import SearchConfig

FIXTURES = (
    Path(__file__).resolve().parents[1] / "configs/benchmarks/search_positions.json"
)


@pytest.mark.parametrize(
    "name,score_before,score_after", [("giveaway_9", 22, 4), ("giveaway_12", 50, 26)]
)
def test_recorded_giveaway_enables_clear_despite_hidden_card_randomness(
    name, score_before, score_after
):
    _, cases = load_cases(FIXTURES)
    case = next(c for c in cases if c.name == name)
    assert game.get_score(case.state, 1) == score_before
    for seed in range(8):
        state = case.state
        for action in case.metadata["clear_sequence"]:
            state = game.apply_action(state, action, rng=np.random.default_rng(seed))
        # The original opponent rotates back to relative seat1 after their clear.
        assert state.table[1, :, 3, game.FINGER_CLEARED].all()
        assert game.get_score(state, 1) == score_after
        assert state.countdown is not None


def test_saved_visits_and_values_distinguish_the_two_failure_modes():
    source, cases = load_cases(FIXTURES)
    first, second = [recorded_selection(c) for c in cases if c.cohort == "tactical"]
    assert (
        source["checkpoint_sha256"]
        == "dcbe5ec51a529460de5c8f6ad91d37fcdef5e0e83f183474832976c3be5d976f"
    )
    assert (first["visit_choice"], first["value_choice"]) == (18, 21)
    assert first["value_advantage"] == pytest.approx(0.013392865657806396)
    assert (second["visit_choice"], second["value_choice"]) == (27, 27)


def test_frozen_exact_endgame_references_still_match_engine_outcomes():
    _, cases = load_cases(FIXTURES)
    exact = [c for c in cases if c.metadata.get("exact_action_values")]
    assert len(exact) == 17
    for case in exact:
        expected = {
            int(a): value for a, value in case.metadata["exact_action_values"].items()
        }
        assert exact_action_values(case.state) == pytest.approx(expected), case.name


class UniformEvaluator:
    def evaluate(self, states):
        return [
            Prediction(
                np.full(state.players, 1 / state.players, dtype=np.float32),
                game.actions(state) / game.actions(state).sum(),
            )
            for state in states
        ]


def test_trial_accounts_for_budget_and_scores_the_selected_action_against_oracle():
    _, cases = load_cases(FIXTURES)
    case = next(c for c in cases if c.cohort == "exact_terminal")
    variant = Variant("test", SearchConfig(boundary_samples=2))
    result = run_trial(case, variant, 32, 17, UniformEvaluator(), enumerated=True)
    assert sum(action["visits"] for action in result["actions"]) == 32
    expected = case.metadata["exact_action_values"]
    assert result["exact_regret"] == pytest.approx(
        max(expected.values()) - expected[str(result["choice"])]
    )
    assert sum(
        action["action_probability"] for action in result["actions"]
    ) == pytest.approx(1)
    assert sum(
        action["training_probability"] for action in result["actions"]
    ) == pytest.approx(1)
    assert result["evaluator_states"] == 1
    assert result["boundary_samples"] >= 32
