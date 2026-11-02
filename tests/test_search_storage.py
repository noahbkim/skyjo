"""Search storage must preserve decisions, RNG consumption and snapshot ownership."""

from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest
from test_action_symmetry import board_state, with_slack
from test_mcts_symmetry import FixedEvaluator

from skyjo.engine import game
from skyjo.experiments.search_performance import result_digest
from skyjo.experiments.search_strategy_benchmark import load_cases
from skyjo.search import mcts, runner
from skyjo.search.configuration import resolve_components
from skyjo.search.strategies import CurrentMixture


class ReferenceMixture(CurrentMixture):
    """Use the generic immutable-snapshot path with identical mixture math."""


def components(config, optimized):
    result = resolve_components(config)
    return result if optimized else replace(result, valuation=ReferenceMixture())


@pytest.mark.parametrize("exact_chance", [False, True])
def test_storage_preserves_search_results_and_rng(exact_chance):
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
        for optimized in (False, True):
            rng = np.random.default_rng(41)
            evaluator = FixedEvaluator()
            result = runner.search(
                state,
                evaluator,
                32,
                config=config,
                rng=rng,
                components=components(config, optimized),
            )
            digests.append(
                (
                    result_digest(result, rng.bit_generator.state),
                    tuple(game.hash_skyjo(s) for s in evaluator.states),
                )
            )
        assert len(set(digests)) == 1


def test_retained_search_snapshots_are_immutable():
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
def test_recycling_retry_preserves_search_and_rng(exact_chance):
    state = with_slack(
        board_state(
            (("H", 0, "X", "X"), (2, 2, "X", "X"), (3, 3, "X", "X")),
            phase=game.ACTION_DRAW_OR_TAKE,
        ),
        2,
    )
    config = mcts.SearchConfig(
        fpu_reduction=1, after_state_evaluate_all_children=exact_chance
    )
    digests = []
    for optimized in (False, True):
        rng = np.random.default_rng(4)
        result = runner.search(
            state,
            FixedEvaluator(force_draw_path=True),
            16,
            config=config,
            rng=rng,
            components=components(config, optimized),
        )
        assert not result.roots[0].effective_merge_symmetric_actions
        digests.append(result_digest(result, rng.bit_generator.state))
    assert len(set(digests)) == 1


def test_root_expansion_and_snapshot_do_not_construct_unselected_children(monkeypatch):
    applied = []
    original = game.apply_action

    def counted(state, action, **kwargs):
        applied.append(action)
        return original(state, action, **kwargs)

    state = board_state(phase=game.ACTION_REPLACE)
    monkeypatch.setattr(game, "apply_action", counted)
    root = mcts.run_mcts(state, FixedEvaluator(), 0)
    view = root.view()
    assert not applied
    assert not view.visits.any()
    # The public mapping still exposes actual nodes when inspected explicitly.
    action = next(a for a in root.children if not game.is_action_random(a, state))
    child = root.children[action]
    assert child.action == action
    assert root.children[action] is child
    assert applied == [action]
