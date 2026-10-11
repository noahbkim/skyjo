"""Search accounting and fresh-tree ownership without a neural model dependency."""

import dataclasses
import random

import numpy as np
import pytest
from test_action_symmetry import board_state

from skyjo.engine import game as sj
from skyjo.search import mcts
from skyjo.search.evaluator import Prediction
from skyjo.search.player import SearchPlayer


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


@pytest.mark.parametrize("exact_chance", [False, True])
@pytest.mark.parametrize("iterations", [1, 3, 4])
def test_chance_initialization_and_boundary_backups(
    monkeypatch, exact_chance, iterations
):
    state = dataclasses.replace(
        board_state(
            ((0, "X", "X", "X"), (3, "X", "X", "X"), (6, "X", "X", "X")),
            phase=sj.ACTION_DRAW_OR_TAKE,
            countdown=2,
            deck_counts={1: 1, 2: 3},
        ),
        turn=1,
    )
    draws = iter([sj.CARD_P1] * (2 if exact_chance else 3) + [sj.CARD_P2])
    apply = sj.apply_action

    def sample(observed, action, **kwargs):
        if observed is state and action == sj.MASK_DRAW:
            observed = sj.preordain(observed, next(draws))
        return apply(observed, action, **kwargs)

    class Evaluator:
        def __init__(self):
            self.states = []

        def evaluate(self, states):
            self.states.extend(states)
            predictions = []
            for observed in states:
                action = sj.MASK_DRAW if observed is state else sj.MASK_REPLACE
                policy = np.zeros(sj.MASK_SIZE, dtype=np.float32)
                policy[action] = 1
                value = 0.2 if sj.get_top(observed) == sj.CARD_P1 else 0.6
                predictions.append(Prediction(np.array([value, 1 - value]), policy))
            return predictions

    class Boundary:
        def __init__(self):
            self.values = iter([0, 0, 1, 0.2, 0.2])
            self.batches = []

        def evaluate(self, states, rng):
            assert all(sj.get_round_over(s) and sj.get_player(s) == 0 for s in states)
            self.batches.append(len(states))
            return np.array([[v, 1 - v] for v in (next(self.values) for _ in states)])

    monkeypatch.setattr(sj, "apply_action", sample)
    evaluator, boundary = Evaluator(), Boundary()
    config = mcts.SearchConfig(
        after_state_evaluate_all_children=exact_chance,
        boundary_samples=2,
        fpu_reduction=1,
        merge_symmetric_actions=False,
    )
    root = mcts.run_mcts(
        state, evaluator, iterations, config=config, boundary_evaluator=boundary
    )
    chance = root.children[sj.MASK_DRAW]
    assert root.visit_count == chance.visit_count == iterations
    if iterations == 1:
        assert sum(c.visit_count for c in chance.children.values()) == (
            0 if exact_chance else 1
        )
        assert root.state_value[0] == pytest.approx(0.5 if exact_chance else 0.2)
        return

    child = next(
        c for c in chance.children.values() if sj.get_top(c.state) == sj.CARD_P1
    )
    assert child.visit_count == (2 if exact_chance else 3)
    assert child.state_value[0] == pytest.approx(0.5 if exact_chance else 0.4)
    np.testing.assert_allclose(
        child.children[sj.MASK_REPLACE].state_value, [1 / 3, 2 / 3]
    )

    if iterations == 3:
        assert chance.state_value[0] == pytest.approx(0.575 if exact_chance else 0.4)
        expected = (0.5 + 0.45 + 0.575) / 3 if exact_chance else 0.4
        np.testing.assert_allclose(root.state_value, [expected, 1 - expected])
        assert boundary.batches == [2, 1]
        return
    second = next(
        c for c in chance.children.values() if sj.get_top(c.state) == sj.CARD_P2
    )
    assert second.visit_count == 1
    assert second.state_value[0] == pytest.approx(0.2 if exact_chance else 0.6)
    assert boundary.batches == ([2, 1, 2] if exact_chance else [2, 1])
    assert len(evaluator.states) == 3  # Root and each distinct outcome, once.
    np.testing.assert_allclose(root.state_value, [0.45, 0.55])


def test_each_player_search_builds_a_fresh_tree():
    state = make_state()

    class Evaluator(UniformEvaluator):
        root_evaluations = 0

        def evaluate(self, states):
            self.root_evaluations += sum(observed is state for observed in states)
            return super().evaluate(states)

    evaluator = Evaluator()
    player = SearchPlayer(evaluator, 4, rng=np.random.default_rng(8))
    first = player.run_mcts(state)
    first_value = first.state_value.copy()
    first_policy = first.policy_targets().copy()
    second = player.run_mcts(state)
    assert first is not second
    assert first.parent is second.parent is None
    assert first.visit_count == second.visit_count == 4
    assert evaluator.root_evaluations == 2
    np.testing.assert_array_equal(first.state_value, first_value)
    np.testing.assert_array_equal(first.policy_targets(), first_policy)


def test_policy_temperature_handles_extremes_and_unvisited_roots():
    root = mcts.run_mcts(make_state(), UniformEvaluator(), 0)
    assert root.visit_count == sum(c.visit_count for c in root.children.values()) == 0
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
