"""Search accounting and cache safety without any neural model dependency."""

import dataclasses
import random

import numpy as np
import pytest

from skyjo.engine import game as sj
from skyjo.search import mcts
from skyjo.search.evaluator import Prediction


class UniformEvaluator:
    def evaluate(self, states):
        return [
            Prediction(
                np.full(state.players, 1 / state.players, dtype=np.float32),
                sj.actions(state).astype(np.float32) / sj.actions(state).sum(),
            )
            for state in states
        ]


def make_state():
    rng = np.random.default_rng(10)
    return sj.start_round(sj.new(players=2, top=sj.CARD_0), rng=rng)


@pytest.mark.parametrize("exact_chance", [False, True])
def test_search_preserves_visits_and_legal_policy(exact_chance):
    root = mcts.run_mcts(
        make_state(),
        UniformEvaluator(),
        8,
        config=mcts.SearchConfig(after_state_evaluate_all_children=exact_chance),
        rng=np.random.default_rng(10),
    )
    assert (
        root.visit_count
        == sum(child.visit_count for child in root.children.values())
        == 8
    )
    assert np.isfinite(root.state_value).all()
    policy = root.policy_targets()
    assert policy.sum() == pytest.approx(1)
    assert not policy[~sj.actions(root.state).astype(bool)].any()
    if exact_chance:
        chances = [
            child
            for child in root.children.values()
            if isinstance(child, mcts.AfterStateNode) and child.is_expanded
        ]
        assert chances
        assert all(
            len(chance.children) > 1 and chance.child_weight_total == 1
            for chance in chances
        )


def test_ucb_uses_fixed_seat_value_and_prior_on_first_selection():
    rng = np.random.default_rng(1)
    state = make_state()
    for _ in range(2):
        state = sj.apply_action(state, sj.MASK_FLIP_SECOND_RIGHT, rng=rng)
    config = mcts.SearchConfig(c_puct=1.5, fpu_reduction=0.1)
    root = mcts.DecisionStateNode(state, None, None, mcts.SearchContext(config))
    policy = np.zeros(sj.MASK_SIZE, dtype=np.float32)
    policy[[sj.MASK_DRAW, sj.MASK_TAKE]] = [0.8, 0.2]
    root.expand(Prediction(np.array([0.7, 0.3], dtype=np.float32), policy))
    assert mcts.ucb_score(root.children[sj.MASK_DRAW], root) == pytest.approx(1.8)
    assert mcts.ucb_score(root.children[sj.MASK_TAKE], root) == pytest.approx(0.9)
    assert root.select_child() is root.children[sj.MASK_DRAW]


def test_exact_chance_update_replaces_propagated_return():
    state = make_state()
    for _ in range(2):
        state = sj.apply_action(state, sj.MASK_FLIP_SECOND_RIGHT)
    root = mcts.DecisionStateNode(state, None, None)
    root.expand(UniformEvaluator().evaluate([state])[0])
    chance = root.children[sj.MASK_DRAW]
    child_state = sj.draw(sj.preordain(state, sj.CARD_P5))
    child = mcts.DecisionStateNode(child_state, chance, sj.MASK_DRAW)
    child.expand(Prediction(np.array([0.6, 0.4]), sj.actions(child_state)))
    key = sj.hash_skyjo(child_state)
    chance.children, chance.child_weights = {key: child}, {key: 0.25}
    chance.child_weight_total = 1
    chance.state_value_total = np.array([0.45, 0.55], dtype=np.float32)
    chance.all_children_discovered = chance.is_expanded = True
    mcts.backpropagate([root, chance, child], np.array([0.8, 0.2]))
    np.testing.assert_allclose(child.state_value, [0.8, 0.2])
    np.testing.assert_allclose(chance.state_value, [0.5, 0.5])
    np.testing.assert_allclose(root.state_value, [0.5, 0.5])


def test_reused_root_rejects_changed_chance_or_evaluators_before_mutation():
    state, evaluator = make_state(), UniformEvaluator()
    config = mcts.SearchConfig(
        after_state_evaluate_all_children=True, dirichlet_epsilon=0.2
    )
    rng = np.random.default_rng(1)
    root = mcts.run_mcts(state, evaluator, 3, config=config, rng=rng)
    weights = [
        (dict(child.child_weights), child.child_weight_total)
        for child in root.children.values()
    ]
    noise, value, random_state = (
        root.dirichlet_noise.copy(),
        root.state_value_total.copy(),
        rng.bit_generator.state,
    )
    for kwargs in (
        {
            "config": dataclasses.replace(
                config, after_state_evaluate_all_children=False
            )
        },
        {"config": dataclasses.replace(config, c_puct=2)},
        {"evaluator": UniformEvaluator()},
        {"boundary_evaluator": object()},
        {"rng": np.random.default_rng(2)},
    ):
        args = {
            "config": config,
            "evaluator": evaluator,
            "iterations": 1,
            "root_node": root,
        } | kwargs
        with pytest.raises(ValueError):
            mcts.run_mcts(state, **args)
        assert root.visit_count == 3
        assert rng.bit_generator.state == random_state
        assert weights == [
            (dict(child.child_weights), child.child_weight_total)
            for child in root.children.values()
        ]
        np.testing.assert_array_equal(root.dirichlet_noise, noise)
        np.testing.assert_array_equal(root.state_value_total, value)
    assert mcts.run_mcts(state, evaluator, 2, config=config, root_node=root) is root
    assert root.visit_count == 5


def test_policy_temperature_handles_extremes_and_unvisited_roots():
    root = mcts.run_mcts(make_state(), UniformEvaluator(), 0)
    actions = list(root.children)
    for temperature in (0, 1):
        with pytest.raises(ValueError, match="visited legal"):
            root.policy_targets(temperature)
    root.children[actions[0]].visit_count = 31
    root.children[actions[1]].visit_count = 1
    for temperature in (0, 0.01, 1e-320, 1):
        policy = root.policy_targets(temperature)
        assert np.isfinite(policy).all()
        assert policy.sum() == pytest.approx(1)
        assert policy.argmax() == actions[0]
    for temperature in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            root.policy_targets(temperature)


@pytest.mark.parametrize("predetermined", [False, True])
def test_engine_chance_outcomes_recycle_discards_without_changing_input(predetermined):
    rng = random.Random(8)
    state = make_state()
    for _ in range(2):
        state = sj.apply_action(state, sj.MASK_FLIP_SECOND_RIGHT, rng=rng)
    remaining = state.deck.sum() - 1
    state.game[sj.GAME_DISCARDS : sj.GAME_DISCARDS + sj.CARD_SIZE] = state.deck
    expected = state.deck.copy() / state.deck.sum()
    state.deck.fill(0)
    if predetermined:
        state, expected = sj.preordain(state, sj.CARD_P1), [1.0]
    before = sj.hash_skyjo(state)
    outcomes = sj.chance_outcomes(state, sj.MASK_DRAW)
    assert sorted(probability for _, probability in outcomes) == pytest.approx(
        sorted(expected)
    )
    assert sj.hash_skyjo(state) == before
    assert all(
        sj.validate(child) and child.deck.sum() == remaining for child, _ in outcomes
    )
    assert sj.hash_skyjo(sj.apply_action(state, sj.MASK_DRAW)) in {
        sj.hash_skyjo(child) for child, _ in outcomes
    }
