"""Behavior and ownership checks for the opt-in search storage experiment."""

from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest
from test_action_symmetry import board_state, with_slack
from test_mcts_symmetry import FixedEvaluator

from skyjo.engine import game
from skyjo.experiments.search_core_poc import VARIANTS, search_variant
from skyjo.experiments.search_performance import result_digest
from skyjo.experiments.search_strategy_benchmark import load_cases
from skyjo.search import mcts, runner


@pytest.mark.parametrize("exact_chance", [False, True])
def test_storage_variants_preserve_search_results_and_rng(exact_chance):
    _, cases = load_cases(Path("configs/benchmarks/search_positions.json"))
    continuing = replace(cases[2].state, game=cases[2].state.game.copy())
    continuing.game[game.GAME_SCORES : game.GAME_SCORES + 2] = 0
    config = mcts.SearchConfig(
        after_state_evaluate_all_children=exact_chance,
        boundary_samples=3,
        dirichlet_epsilon=0.25,
    )
    for state in (cases[0].state, cases[2].state, continuing):
        digests = []
        for variant in VARIANTS:
            rng = np.random.default_rng(41)
            evaluator = FixedEvaluator()
            with search_variant(variant):
                result = runner.search(state, evaluator, 32, config=config, rng=rng)
            digests.append(
                (
                    result_digest(result, rng.bit_generator.state),
                    tuple(game.hash_skyjo(s) for s in evaluator.states),
                )
            )
        assert len(set(digests)) == 1


@pytest.mark.parametrize("variant", VARIANTS[1:])
def test_retained_search_snapshots_are_immutable(variant):
    saved = []

    def retain(root):
        if root.visit_count == 1:
            view = root.view()
            saved.append(
                (
                    view,
                    {
                        f.name: getattr(view, f.name).copy()
                        for f in fields(view)
                        if isinstance(getattr(view, f.name), np.ndarray)
                    },
                )
            )

    with search_variant(variant):
        root = mcts.run_mcts(
            board_state(),
            FixedEvaluator(),
            32,
            rng=np.random.default_rng(4),
            observer=retain,
        )
    assert root.visit_count == 32
    assert saved
    for view, arrays in saved:
        for name, expected in arrays.items():
            np.testing.assert_array_equal(getattr(view, name), expected)
            assert not getattr(view, name).flags.writeable


@pytest.mark.parametrize("exact_chance", [False, True])
def test_recycling_retry_and_context_cleanup(exact_chance):
    state = with_slack(
        board_state(
            (("H", 0, "X", "X"), (2, 2, "X", "X"), (3, 3, "X", "X")),
            phase=game.ACTION_DRAW_OR_TAKE,
        ),
        2,
    )
    original = mcts.DecisionStateNode, mcts.backpropagate
    digests = []
    for variant in VARIANTS:
        rng = np.random.default_rng(4)
        with search_variant(variant):
            result = runner.search(
                state,
                FixedEvaluator(force_draw_path=True),
                16,
                config=mcts.SearchConfig(
                    fpu_reduction=1, after_state_evaluate_all_children=exact_chance
                ),
                rng=rng,
            )
            assert not result.roots[0].effective_merge_symmetric_actions
            digests.append(result_digest(result, rng.bit_generator.state))
        assert (mcts.DecisionStateNode, mcts.backpropagate) == original
    assert len(set(digests)) == 1
    with pytest.raises(RuntimeError), search_variant("combined"):
        raise RuntimeError("evaluation failed")
    assert (mcts.DecisionStateNode, mcts.backpropagate) == original
