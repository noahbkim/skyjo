"""Experimental speedups must preserve results and ownership boundaries."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch
from test_action_symmetry import board_state, with_slack
from test_mcts_symmetry import FixedEvaluator

from skyjo.engine import game
from skyjo.experiments.search_performance import result_digest
from skyjo.experiments.search_performance_poc import (
    VARIANTS,
    CachedDecisionNode,
    PackedInputs,
    make_predictor,
    snapshot_variant,
)
from skyjo.experiments.search_strategy_benchmark import load_cases
from skyjo.learning import batches, models
from skyjo.search import mcts, runner
from skyjo.search.configuration import preset


def small_model(players=2):
    torch.set_num_threads(1)
    torch.manual_seed(1)
    return models.build(
        {
            "embedding_dimensions": 4,
            "global_state_embedding_dimensions": 8,
            "num_heads": 2,
        },
        players=players,
        device="cpu",
    )


@pytest.mark.parametrize("players", [2, 4, 8])
def test_packing_and_predictions_preserve_inputs_and_retained_results(players):
    rng = np.random.default_rng(3)
    state = game.start_round(game.new(players=players, rng=rng), rng=rng)
    states = []
    for _ in range(7):
        states.append(state)
        state = game.apply_action(
            state, int(rng.choice(np.flatnonzero(game.actions(state)))), rng=rng
        )
    packed = PackedInputs(7, players).fill(states)
    expected_batch = batches.states_to_batch(states)
    for field in ("spatial_inputs", "non_spatial_inputs", "action_masks"):
        np.testing.assert_array_equal(
            getattr(packed, field), getattr(expected_batch, field)
        )
    model = small_model(players).train()
    expected = make_predictor(model, "baseline", max_batch_size=3).evaluate(states)
    original_states = [s.table.copy() for s in states]
    for variant in ("mode", "packing", "sharing", "inference"):
        evaluator = make_predictor(model, variant, max_batch_size=3)
        assert evaluator.evaluate([]) == []
        result = evaluator.evaluate(states)
        evaluator.evaluate(states[::-1] * 2)  # Refill all buffers and partial chunks.
        for actual, reference in zip(result, expected, strict=True):
            np.testing.assert_array_equal(actual.value, reference.value)
            np.testing.assert_array_equal(actual.policy, reference.policy)
    assert model.training  # The experimental evaluator does not own the caller's model.
    for state, table in zip(states, original_states, strict=True):
        np.testing.assert_array_equal(state.table, table)


@pytest.mark.parametrize("strategy", ["legacy", "alphazero", "gumbel"])
@pytest.mark.parametrize("exact_chance", [False, True])
def test_search_variants_preserve_results_and_random_streams(strategy, exact_chance):
    _, cases = load_cases(Path("configs/benchmarks/search_positions.json"))
    # A tactical position, terminal boundary, and continuing-round boundary.
    states = [
        cases[0].state,
        cases[2].state,
        dataclasses.replace(cases[2].state, game=cases[2].state.game.copy()),
    ]
    states[-1].game[game.GAME_SCORES : game.GAME_SCORES + 2] = 0
    config = preset(
        strategy, after_state_evaluate_all_children=exact_chance, boundary_samples=3
    )
    model = small_model()
    for state in states:
        signatures = []
        for variant in VARIANTS:
            evaluator = make_predictor(model, variant)
            rng = np.random.default_rng(41)
            with snapshot_variant(variant):
                result = runner.search(state, evaluator, 16, config=config, rng=rng)
                signatures.append(result_digest(result, rng.bit_generator.state))
        assert len(set(signatures)) == 1


def test_snapshot_cache_invalidates_without_changing_retained_snapshots():
    state = board_state()
    with snapshot_variant("snapshots"):
        root = mcts.run_mcts(state, FixedEvaluator(), 2, rng=np.random.default_rng(4))
        assert isinstance(root, CachedDecisionNode)
        saved = root.view()
        retained = {
            field.name: getattr(saved, field.name).copy()
            for field in dataclasses.fields(saved)
            if isinstance(getattr(saved, field.name), np.ndarray)
        }
        child = next(iter(root.children.values()))
        child.visit_count += 1
        child._state_value[:] = 0.75
        root.visit_count += 1
        root.action_priors[:] = 0.25
        root.model_prediction.value[:] = 0.4
        root.refresh_value()
        current = root.view()
        reference = CachedDecisionNode.__mro__[1].view(root)
        for field in dataclasses.fields(current):
            actual, expected = (
                getattr(current, field.name),
                getattr(reference, field.name),
            )
            if isinstance(actual, np.ndarray):
                np.testing.assert_array_equal(actual, expected)
                assert not actual.flags.writeable
            else:
                assert actual == expected
        for name, array in retained.items():
            np.testing.assert_array_equal(getattr(saved, name), array)


@pytest.mark.parametrize("exact_chance", [False, True])
def test_snapshot_variant_preserves_recycling_retry_and_restores_class(exact_chance):
    board = (("H", 0, "X", "X"), (2, 2, "X", "X"), (3, 3, "X", "X"))
    state = with_slack(board_state(board, phase=game.ACTION_DRAW_OR_TAKE), 2)
    original = mcts.DecisionStateNode
    signatures = []
    for variant in ("baseline", "snapshots"):
        rng = np.random.default_rng(4)
        evaluator = FixedEvaluator(force_draw_path=True)
        with snapshot_variant(variant):
            result = runner.search(
                state,
                evaluator,
                16,
                config=mcts.SearchConfig(
                    fpu_reduction=1, after_state_evaluate_all_children=exact_chance
                ),
                rng=rng,
            )
            assert not result.roots[0].effective_merge_symmetric_actions
            signatures.append(
                (result_digest(result, rng.bit_generator.state), len(evaluator.states))
            )
        assert mcts.DecisionStateNode is original
    assert signatures[0] == signatures[1]
    with pytest.raises(RuntimeError), snapshot_variant("snapshots"):
        raise RuntimeError("test cleanup")
    assert mcts.DecisionStateNode is original
