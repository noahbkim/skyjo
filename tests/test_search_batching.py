"""Batch independent searches without changing their per-tree traversal order."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_mcts_symmetry import FixedEvaluator

from skyjo.engine import game
from skyjo.experiments.search_performance import result_digest
from skyjo.experiments.search_strategy_benchmark import load_cases
from skyjo.search import runner
from skyjo.search.batching import EvaluationRequest, evaluate_concurrently
from skyjo.search.configuration import SearchConfig


@pytest.mark.parametrize(
    "config",
    [
        SearchConfig(dirichlet_epsilon=0.25, boundary_samples=3),
        SearchConfig(after_state_evaluate_all_children=True, boundary_samples=3),
        SearchConfig(root_strategy="gumbel", action_strategy="gumbel"),
        SearchConfig(replicas=2),
    ],
)
def test_batched_search_preserves_policies_values_and_rng(config):
    _, cases = load_cases(Path("configs/benchmarks/search_positions.json"))
    continuing = replace(cases[2].state, game=cases[2].state.game.copy())
    continuing.game[game.GAME_SCORES : game.GAME_SCORES + 2] = 0
    states = [cases[0].state, cases[2].state, continuing]
    evaluator = FixedEvaluator()
    expected = []
    for index, state in enumerate(states):
        rng = np.random.default_rng(index)
        result = runner.search(state, evaluator, 64, config=config, rng=rng)
        expected.append(result_digest(result, rng.bit_generator.state))
    streams = [np.random.default_rng(i) for i in range(len(states))]
    results = evaluate_concurrently(
        (
            runner.search_steps(state, evaluator, 64, config=config, rng=rng)
            for state, rng in zip(states, streams, strict=True)
        ),
        2,
    )
    assert expected == [
        result_digest(result, rng.bit_generator.state)
        for result, rng in zip(results, streams, strict=True)
    ]


def test_scheduler_refills_slots_and_preserves_evaluator_identity_and_result_order():
    class Evaluator:
        def __init__(self):
            self.batches = []

        def evaluate(self, states):
            self.batches.append(tuple(states))
            return states

    first, second = Evaluator(), Evaluator()

    def task(evaluator, name, count):
        for i in range(count):
            assert (yield EvaluationRequest(evaluator, [(name, i)])) == [(name, i)]
        return name

    results = evaluate_concurrently(
        [task(first, "a", 1), task(first, "b", 3), task(second, "c", 1)], 2
    )
    assert results == ["a", "b", "c"]
    assert first.batches == [(("a", 0), ("b", 0)), (("b", 1),), (("b", 2),)]
    assert second.batches == [(("c", 0),)]


def test_failed_evaluation_closes_all_active_searches():
    closed = []

    class Evaluator:
        def evaluate(self, states):
            raise RuntimeError("evaluation failed")

    evaluator = Evaluator()

    def task(index):
        try:
            yield EvaluationRequest(evaluator, [index])
        finally:
            closed.append(index)

    with pytest.raises(RuntimeError, match="evaluation failed"):
        evaluate_concurrently((task(i) for i in range(3)), 2)
    assert closed == [0, 1]
