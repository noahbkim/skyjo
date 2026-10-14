"""Search composition contracts, fresh evidence, and independently chosen targets."""

import dataclasses

import numpy as np
import pytest
from test_action_symmetry import board_state
from test_mcts_core import UniformEvaluator

from skyjo.engine import game as sj
from skyjo.search import mcts
from skyjo.search.strategies import (
    BestResponse,
    CompletedQPolicy,
    CurrentMixture,
    Evidence,
    GumbelAllocation,
    GumbelContext,
    GumbelPolicy,
    MeanReturn,
    NodeView,
    PolicyInput,
    ValuePolicy,
    VisitPolicy,
    frozen_array,
)
from skyjo.search.symmetry import ActionGroups


def view(**changes):
    """Two action groups with fixed-seat values; seat one prefers group two."""
    values = {
        "groups": ActionGroups(((16, 20), (17,))),
        "actions": (16, 17),
        "priors": frozen_array([0.75, 0.25]),
        "model_priors": frozen_array([0.75, 0.25]),
        "visits": frozen_array([1, 1]),
        "values": frozen_array([[0.8, 0.2], [0.1, 0.9]]),
        "available": frozen_array([True, True]),
        "player": 1,
        "state_value": frozen_array([0.9, 0.1]),
        "model_value": frozen_array([0.9, 0.1]),
        "visit_count": 2,
        "is_root": True,
    }
    values.update(changes)
    return NodeView(**values)


@pytest.mark.parametrize("name", ["legacy", "alphazero", "gumbel"])
def test_uniform_group_search_ignores_model_policy_for_all_strategies(name):
    from skyjo.search.configuration import preset
    from skyjo.search.runner import search

    class SkewedEvaluator(UniformEvaluator):
        def __init__(self, reverse):
            self.reverse = reverse

        def evaluate(self, states):
            predictions = super().evaluate(states)
            weights = np.arange(1, sj.MASK_SIZE + 1, dtype=np.float32)
            if self.reverse:
                weights = weights[::-1]
            for state, prediction in zip(states, predictions, strict=True):
                prediction.policy[:] = sj.actions(state) * weights
                prediction.policy[:] /= prediction.policy.sum()
            return predictions

    results = [
        search(
            board_state(),
            SkewedEvaluator(reverse),
            16,
            config=preset(name, uniform_group_priors=True),
            rng=np.random.default_rng(4),
        )
        for reverse in (False, True)
    ]
    a, b = results
    assert not np.allclose(a.summary.model_priors, b.summary.model_priors)
    np.testing.assert_array_equal(a.summary.visits, b.summary.visits)
    np.testing.assert_array_equal(a.action_probabilities, b.action_probabilities)
    np.testing.assert_array_equal(a.training_policy, b.training_policy)


@pytest.mark.parametrize(
    "strategy, expected",
    [(CurrentMixture(), 0.6), (MeanReturn(), 0.45), (BestResponse(), 0.1)],
)
def test_backups_distinguish_fresh_returns_current_estimates_and_opponent_choice(
    strategy, expected
):
    current = view()
    estimate = strategy.initialize(current.model_value)
    estimate.update(
        dataclasses.replace(
            current,
            visits=frozen_array([1, 0]),
            available=frozen_array([True, False]),
            visit_count=1,
        ),
        Evidence(np.array([0.8, 0.2])),
    )
    estimate.update(current, Evidence(np.array([0.1, 0.9])))
    np.testing.assert_allclose(estimate.value, [expected, 1 - expected])


def test_mean_return_ignores_refreshed_estimates_as_independent_evidence():
    current = view()
    estimate = MeanReturn().initialize(current.model_value)
    estimate.update(current, Evidence(np.array([0.2, 0.8])))
    estimate.update(current, Evidence(np.array([0.9, 0.1]), kind="refresh"))
    estimate.update(current, Evidence(np.array([0.4, 0.6])))
    np.testing.assert_allclose(estimate.value, [0.3, 0.7])


@pytest.mark.parametrize(
    "player, expected", [(1, [0.2, 0.7, 0.1]), (2, [0.6, 0.1, 0.3])]
)
def test_best_response_preserves_the_selected_full_multiplayer_vector(player, expected):
    current = view(
        player=player,
        values=frozen_array([[0.2, 0.7, 0.1], [0.6, 0.1, 0.3]]),
        state_value=frozen_array([1 / 3] * 3),
        model_value=frozen_array([1 / 3] * 3),
    )
    estimate = BestResponse().initialize(current.model_value)
    estimate.update(current, Evidence(current.model_value))
    np.testing.assert_allclose(estimate.value, expected)


def test_action_policies_use_group_mass_fixed_seats_and_replica_uncertainty():
    current = view(visits=frozen_array([9, 4]))
    probabilities = VisitPolicy().probabilities(PolicyInput(current), 2)
    np.testing.assert_allclose(probabilities[[16, 20, 17]], [0.3, 0.3, 0.4])
    assert probabilities.sum() == pytest.approx(1)
    greedy = VisitPolicy().probabilities(PolicyInput(current), 0)
    assert greedy[16] == 1  # Deterministic representative, not a split group.
    assert ValuePolicy().probabilities(PolicyInput(current), 0)[17] == 1

    uncertain = view(
        values=frozen_array([[0.2, 0.8], [0.3, 0.7]]),
        standard_error=frozen_array([[0.2, 0.2], [0.01, 0.01]]),
    )
    assert ValuePolicy(0).probabilities(PolicyInput(uncertain), 0)[16] == 1
    assert ValuePolicy(1).probabilities(PolicyInput(uncertain), 0)[17] == 1


def test_improved_target_excludes_gumbel_noise_and_acting_temperature():
    current = view()
    left = PolicyInput(current, GumbelContext(frozen_array([100.0, 0.0])))
    right = PolicyInput(current, GumbelContext(frozen_array([0.0, 100.0])))
    assert GumbelPolicy().probabilities(left, 0)[16] == 1
    assert GumbelPolicy().probabilities(right, 0)[17] == 1
    target = CompletedQPolicy()
    np.testing.assert_array_equal(
        target.probabilities(left, 1), target.probabilities(right, 1)
    )
    probabilities = target.probabilities(left, 1)
    assert probabilities.sum() == pytest.approx(1)
    assert np.count_nonzero(probabilities) == 3


@pytest.mark.parametrize("budget", [1, 3, 8, 17])
def test_gumbel_sessions_consume_the_budget_and_reproduce_seeded_results(budget):
    def allocate():
        current = view(
            visits=frozen_array([0, 0]),
            available=frozen_array([False, False]),
            visit_count=0,
        )
        session = GumbelAllocation(max_candidates=16).start(
            budget, np.random.default_rng(8)
        )
        for _ in range(budget):
            action = session.select(current)
            assert action in current.actions
            counts = current.visits.copy()
            counts[current.actions.index(action)] += 1
            current = dataclasses.replace(
                current,
                visits=frozen_array(counts),
                available=frozen_array(counts > 0),
                visit_count=current.visit_count + 1,
            )
            if session.observe is not None:
                session.observe(current)
        result = PolicyInput(current, session.finish(current))
        action = GumbelPolicy().probabilities(result, 0)
        assert current.visits.sum() == current.visit_count == budget
        assert action.sum() == 1
        assert current.visits[current.actions.index(int(action.argmax()))] > 0
        return current.visits, action, CompletedQPolicy().probabilities(result, 1)

    first, second = allocate(), allocate()
    for before, after in zip(first, second, strict=True):
        np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize("budget", [1, 2, 7])
def test_gumbel_candidate_limit_keeps_all_work_on_the_initial_top_candidates(budget):
    current = view(
        groups=ActionGroups(((16,), (17,), (18,), (19,))),
        actions=(16, 17, 18, 19),
        priors=frozen_array([0.1, 0.4, 0.2, 0.3]),
        model_priors=frozen_array([0.1, 0.4, 0.2, 0.3]),
        visits=frozen_array([0, 0, 0, 0]),
        values=frozen_array([[0.5, 0.5]] * 4),
        available=frozen_array([False] * 4),
        visit_count=0,
    )
    session = GumbelAllocation(max_candidates=2, scale=0).start(
        budget, np.random.default_rng(2)
    )
    selected = set()
    for _ in range(budget):
        action = session.select(current)
        selected.add(action)
        counts = current.visits.copy()
        counts[current.actions.index(action)] += 1
        current = dataclasses.replace(
            current,
            visits=frozen_array(counts),
            available=frozen_array(counts > 0),
            visit_count=current.visit_count + 1,
        )
    assert selected == ({17} if budget == 1 else {17, 19})
    assert current.visits.sum() == budget


class FirstAction:
    def start(self, budget, rng):
        return self

    def select(self, current):
        return current.actions[0]

    def observe(self, current):
        pass

    def finish(self, current):
        return None


def test_configured_dirichlet_concentration_only_changes_root_priors():
    state = board_state(phase=sj.ACTION_DRAW_OR_TAKE)
    root = mcts.run_mcts(
        state,
        UniformEvaluator(),
        8,
        config=mcts.SearchConfig(dirichlet_epsilon=1, dirichlet_alpha=0.03),
        rng=np.random.default_rng(5),
    )
    expected = np.random.default_rng(5).dirichlet([0.03, 0.03])
    np.testing.assert_allclose(root.action_priors[list(root.children)], expected)
    descendants = list(root.children.values())
    expanded = 0
    while descendants:
        node = descendants.pop()
        if isinstance(node, mcts.DecisionStateNode) and node.is_expanded:
            expanded += 1
            np.testing.assert_array_equal(node.action_priors, node.model_action_priors)
            assert not node.dirichlet_noise.any()
        if not isinstance(node, mcts.RoundBoundaryNode):
            descendants.extend(node.children.values())
    assert expanded > 0


@pytest.mark.parametrize(
    "valuation, expected", [(CurrentMixture(), 7 / 18), (MeanReturn(), 0.5)]
)
def test_fresh_boundary_batches_do_not_back_up_successive_pooled_means(
    valuation, expected
):
    from skyjo.search.configuration import resolve_components

    state = board_state(
        ((0, "X", "X", "X"), (3, "X", "X", "X"), (6, "X", "X", "X")),
        phase=sj.ACTION_REPLACE,
        countdown=1,
    )

    class Boundary:
        def __init__(self):
            self.samples = iter([0.0, 0.0, 1.0])

        def evaluate(self, states, rng):
            return np.array([[x, 1 - x] for x in (next(self.samples) for _ in states)])

    config = mcts.SearchConfig(boundary_samples=2)
    components = dataclasses.replace(
        resolve_components(config),
        valuation=valuation,
        root_exploration=FirstAction(),
    )
    root = mcts.run_mcts(
        state,
        UniformEvaluator(),
        2,
        config=config,
        components=components,
        boundary_evaluator=Boundary(),
    )
    boundary = next(iter(root.children.values()))
    assert boundary.visit_count == root.visit_count == 2
    assert boundary.sample_count == 3
    np.testing.assert_allclose(boundary.state_value, [1 / 3, 2 / 3])
    np.testing.assert_allclose(root.state_value, [expected, 1 - expected])


@pytest.mark.parametrize("valuation", [CurrentMixture(), MeanReturn(), BestResponse()])
def test_all_valuations_preserve_probability_weighted_chance(valuation):
    from skyjo.search.configuration import resolve_components

    state = board_state(phase=sj.ACTION_DRAW_OR_TAKE, deck_counts={1: 1, 2: 3})

    class Evaluator:
        def evaluate(self, states):
            predictions = UniformEvaluator().evaluate(states)
            for observed, prediction in zip(states, predictions, strict=True):
                q = (
                    0.9
                    if observed is state
                    else {sj.CARD_P1: 0.2, sj.CARD_P2: 0.6}[sj.get_top(observed)]
                )
                prediction.value[:] = [q, 1 - q]
            return predictions

    config = mcts.SearchConfig(after_state_evaluate_all_children=True)
    components = dataclasses.replace(
        resolve_components(config),
        valuation=valuation,
        root_exploration=FirstAction(),
    )
    root = mcts.run_mcts(state, Evaluator(), 1, config=config, components=components)
    chance = root.children[sj.MASK_DRAW]
    np.testing.assert_allclose(chance.state_value, [0.5, 0.5])
    expected = 0.7 if isinstance(valuation, CurrentMixture) else 0.5
    np.testing.assert_allclose(root.state_value, [expected, 1 - expected])


def test_mean_return_backs_up_fresh_sampled_outcomes_instead_of_chance_means(
    monkeypatch,
):
    from skyjo.search.configuration import resolve_components

    state = board_state(phase=sj.ACTION_DRAW_OR_TAKE)
    outcomes = iter([sj.CARD_P1, sj.CARD_P2])
    apply_action = sj.apply_action

    def sample(observed, action, **kwargs):
        if observed is state and action == sj.MASK_DRAW:
            observed = sj.preordain(observed, next(outcomes))
        return apply_action(observed, action, **kwargs)

    class Evaluator:
        def evaluate(self, states):
            predictions = UniformEvaluator().evaluate(states)
            for observed, prediction in zip(states, predictions, strict=True):
                q = (
                    0.9
                    if observed is state
                    else {sj.CARD_P1: 0.2, sj.CARD_P2: 0.6}[sj.get_top(observed)]
                )
                prediction.value[:] = [q, 1 - q]
            return predictions

    monkeypatch.setattr(sj, "apply_action", sample)
    config = mcts.SearchConfig(valuation="mean_return")
    components = dataclasses.replace(
        resolve_components(config), root_exploration=FirstAction()
    )
    root = mcts.run_mcts(state, Evaluator(), 2, config=config, components=components)
    # Fresh outcomes .2 and .6 average to .4; successive chance means would give .3.
    np.testing.assert_allclose(root.state_value, [0.4, 0.6])
    np.testing.assert_allclose(root.children[sj.MASK_DRAW].state_value, [0.4, 0.6])


def test_custom_strategies_work_with_immutable_inputs_without_runner_changes():
    from skyjo.search.configuration import resolve_components
    from skyjo.search.runner import search

    class RoundRobin:
        def __init__(self):
            self.sessions = []

        def start(self, budget, rng):
            session = Session()
            self.sessions.append(session)
            return session

    class Session:
        def __init__(self):
            self.observations = []
            self.node_id = None

        def select(self, current):
            assert not current.visits.flags.writeable
            assert not current.values.flags.writeable
            if self.node_id is None:
                self.node_id = current.node_id
            assert current.node_id == self.node_id
            return current.actions[current.visit_count % len(current.actions)]

        def observe(self, current):
            self.observations.append(current.visit_count)

        def finish(self, current):
            return tuple(self.observations)

    class LastActionTarget:
        def probabilities(self, result, temperature):
            assert result.context == (1, 2, 3, 4)
            assert not result.action_probabilities.flags.writeable
            probabilities = np.zeros(sj.MASK_SIZE, dtype=np.float32)
            probabilities[result.summary.actions[-1]] = 1
            return probabilities

    class CheckedValuation:
        def initialize(self, model_value):
            with pytest.raises(ValueError):
                model_value[:] = 0
            return CheckedAccumulator(MeanReturn().initialize(model_value))

    class CheckedAccumulator:
        def __init__(self, delegate):
            self.delegate = delegate

        @property
        def value(self):
            return self.delegate.value

        def update(self, view, evidence):
            with pytest.raises(ValueError):
                evidence.value[:] = 0
            self.delegate.update(view, evidence)

    state = board_state(phase=sj.ACTION_DRAW_OR_TAKE)
    config = mcts.SearchConfig()
    allocation = RoundRobin()
    components = dataclasses.replace(
        resolve_components(config),
        root_exploration=allocation,
        valuation=CheckedValuation(),
        target_policy=LastActionTarget(),
    )
    outputs = [
        search(state, UniformEvaluator(), 4, components=components, temperature=0)
        for _ in range(2)
    ]
    assert len(allocation.sessions) == 2
    assert allocation.sessions[0] is not allocation.sessions[1]
    for result in outputs:
        np.testing.assert_array_equal(result.summary.visits, [2, 2])
        assert result.action_probabilities[sj.MASK_DRAW] == 1
        assert result.training_policy[sj.MASK_TAKE] == 1
        assert not result.action_probabilities.flags.writeable
        assert not result.training_policy.flags.writeable


def test_replica_coverage_spends_one_total_budget_and_reports_uncertainty():
    from skyjo.search.runner import search

    state = board_state(phase=sj.ACTION_DRAW_OR_TAKE)
    config = mcts.SearchConfig(
        replicas=3, action_strategy="lower_confidence", target_strategy="visits"
    )

    def run():
        return search(
            state, UniformEvaluator(), 11, config=config, rng=np.random.default_rng(3)
        )

    result = run()
    assert (
        result.summary.visit_count
        == sum(root.visit_count for root in result.roots)
        == 11
    )
    assert [root.visit_count for root in result.roots] == [4, 4, 3]
    assert all(
        all(child.visit_count for child in root.children.values())
        for root in result.roots
    )
    np.testing.assert_allclose(result.summary.standard_error, 0)
    assert result.action_probabilities.sum() == pytest.approx(1)
    assert result.training_policy.sum() == pytest.approx(1)
    replayed = run()
    np.testing.assert_array_equal(result.summary.visits, replayed.summary.visits)
    np.testing.assert_array_equal(
        result.action_probabilities, replayed.action_probabilities
    )


def test_insufficient_replica_coverage_budget_fails_before_evaluation():
    from skyjo.search.runner import search

    class Evaluator(UniformEvaluator):
        calls = 0

        def evaluate(self, states):
            self.calls += 1
            return super().evaluate(states)

    evaluator = Evaluator()
    with pytest.raises(ValueError, match="coverage|cover"):
        search(
            board_state(phase=sj.ACTION_DRAW_OR_TAKE),
            evaluator,
            5,
            config=mcts.SearchConfig(replicas=3),
        )
    assert evaluator.calls == 0


def test_alphazero_acting_temperature_does_not_change_visit_targets():
    from skyjo.search.configuration import preset
    from skyjo.search.runner import search

    state = board_state(phase=sj.ACTION_DRAW_OR_TAKE)
    outputs = [
        search(
            state,
            UniformEvaluator(),
            8,
            config=preset("alphazero"),
            temperature=temperature,
            rng=np.random.default_rng(3),
        )
        for temperature in [0, 2]
    ]
    assert np.count_nonzero(outputs[0].action_probabilities) == 1
    assert np.count_nonzero(outputs[1].action_probabilities) == 2
    np.testing.assert_array_equal(
        outputs[0].training_policy, outputs[1].training_policy
    )
    assert np.count_nonzero(outputs[0].training_policy) == 2
