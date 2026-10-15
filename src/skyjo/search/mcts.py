"""Within-round Monte Carlo search with explicit evaluator and RNG ownership."""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Callable, Iterator
from copy import deepcopy

import numpy as np

from skyjo.engine import game as sj

from . import symmetry
from .configuration import DEFAULT_SEARCH_CONFIG, SearchConfig, resolve_components
from .evaluator import BoundaryEvaluator, Evaluator, NextDealEvaluator, Prediction
from .strategies import (
    AllocationSession,
    Evidence,
    NodeView,
    SearchComponents,
    ValueAccumulator,
    frozen_array,
)


class _SymmetryRecyclingRisk(Exception):
    """An explored grouped path reached the reserve needed for final reveals."""


@dataclasses.dataclass(frozen=True, slots=True)
class SearchContext:
    """Settings and dependencies shared within one search tree."""

    config: SearchConfig = dataclasses.field(default_factory=SearchConfig)
    boundary_evaluator: BoundaryEvaluator | None = None
    rng: np.random.Generator = dataclasses.field(default_factory=np.random.default_rng)
    components: SearchComponents | None = None
    root_session: AllocationSession | None = None
    interior_session: AllocationSession | None = None
    node_ids: Iterator[int] = dataclasses.field(default_factory=itertools.count)

    def __post_init__(self):
        if self.components is None:
            object.__setattr__(self, "components", resolve_components(self.config))
        if self.root_session is None:
            object.__setattr__(
                self,
                "root_session",
                self.components.root_exploration.start(0, self.rng),
            )
        if self.interior_session is None:
            object.__setattr__(
                self,
                "interior_session",
                self.components.interior_exploration.start(0, self.rng),
            )


def ucb_score(child: MCTSNode, parent: DecisionStateNode) -> float:
    """Compatibility accessor for the configured PUCT score."""
    from .strategies import PUCT

    config = parent.context.config
    view = parent.view()
    return float(
        PUCT(config.c_puct, config.fpu_reduction, config.fpu_mode).scores(view)[
            view.actions.index(child.action)
        ]
    )


@dataclasses.dataclass(slots=True)
class DecisionStateNode:
    state: sj.Skyjo
    parent: DecisionStateNode | AfterStateNode | None
    action: sj.SkyjoAction | None
    context: SearchContext | None = None
    effective_merge_symmetric_actions: bool = False
    _state_value: np.ndarray = dataclasses.field(init=False)
    _valuation_state: ValueAccumulator | None = dataclasses.field(
        init=False, default=None
    )
    _node_id: int = dataclasses.field(init=False)
    model_prediction: Prediction | None = None
    children: dict[int, MCTSNode] = dataclasses.field(default_factory=dict)
    visit_count: int = 0
    is_expanded: bool = False
    dirichlet_noise: np.ndarray = dataclasses.field(init=False)
    action_groups: symmetry.ActionGroups | None = None
    model_action_priors: np.ndarray | None = None
    base_action_priors: np.ndarray | None = None
    action_priors: np.ndarray | None = None

    def __post_init__(self):
        if self.context is None:
            self.context = (
                self.parent.context if self.parent is not None else SearchContext()
            )
        self._node_id = next(self.context.node_ids)
        self._state_value = np.zeros(self.state.players, dtype=np.float32)
        self.dirichlet_noise = np.zeros(sj.MASK_SIZE, dtype=np.float32)

    @property
    def has_value_estimate(self):
        return self.model_prediction is not None

    @property
    def state_value(self):
        return self._state_value

    def view(self):
        actions = tuple(self.children)
        return NodeView(
            self.action_groups,
            actions,
            frozen_array([self.action_priors[a] for a in actions]),
            frozen_array([self.model_action_priors[a] for a in actions]),
            frozen_array([self.children[a].visit_count for a in actions]),
            frozen_array([self.children[a].state_value for a in actions]),
            frozen_array([self.children[a].has_value_estimate for a in actions]),
            sj.get_player(self.state),
            frozen_array(self.state_value),
            frozen_array(self.model_prediction.value),
            self.visit_count,
            self.parent is None,
            node_id=self._node_id,
            base_priors=frozen_array([self.base_action_priors[a] for a in actions]),
        )

    def refresh_value(self, evidence=None):
        self._valuation_state.update(
            self.view(),
            evidence if evidence is not None else Evidence(self.state_value, "refresh"),
        )
        self._state_value = self._valuation_state.value

    def expand(self, model_prediction: Prediction):
        assert not self.is_expanded
        if (
            self.effective_merge_symmetric_actions
            and not symmetry.safe_to_merge_actions(self.state, 0)
        ):
            raise _SymmetryRecyclingRisk
        self.model_prediction = model_prediction
        self._valuation_state = self.context.components.valuation.initialize(
            frozen_array(model_prediction.value)
        )
        self._state_value = self._valuation_state.value
        self.action_groups = symmetry.ActionGroups.from_state(
            self.state, merge=self.effective_merge_symmetric_actions
        )
        self.model_action_priors = self.action_groups.aggregate(model_prediction.policy)
        self.base_action_priors = self.model_action_priors
        if self.context.config.uniform_group_priors:
            self.base_action_priors = np.zeros(sj.MASK_SIZE, dtype=np.float64)
            representatives = self.action_groups.representatives
            self.base_action_priors[list(representatives)] = 1 / len(representatives)
        self.action_priors = self.base_action_priors
        self.is_expanded = True
        self.children = {
            action: self.create_child_node(action)
            for action in self.action_groups.representatives
        }

    def create_child_node(self, action):
        if sj.get_round_about_to_end(self.state):
            return RoundBoundaryNode(self.state, self, action)
        if sj.is_action_random(action, self.state):
            return AfterStateNode(self.state, action, self)
        return DecisionStateNode(
            sj.apply_action(self.state, action, rng=self.context.rng),
            self,
            action,
            effective_merge_symmetric_actions=self.effective_merge_symmetric_actions,
        )

    def select_child(self, **kwargs):
        session = (
            self.context.root_session
            if self.parent is None
            else self.context.interior_session
        )
        action = session.select(self.view())
        if action not in self.children:
            raise ValueError("Exploration selected an illegal action group")
        return self.children[action]

    def policy_targets(self, temperature=1.0):
        assert self.action_groups is not None
        return self.action_groups.policy_from_visits(
            {action: child.visit_count for action, child in self.children.items()},
            temperature,
        )

    def action_probability(self, action):
        assert self.action_priors is not None
        return float(self.action_priors[action])

    def refresh_root_noise(self):
        assert self.action_groups is not None
        self.dirichlet_noise.fill(0)
        epsilon = self.context.config.dirichlet_epsilon
        self.action_priors = self.base_action_priors
        if epsilon:
            actions = np.flatnonzero(sj.actions(self.state))
            self.dirichlet_noise[actions] = self.context.rng.dirichlet(
                np.full(
                    len(actions),
                    self.context.config.dirichlet_alpha or 10 / len(actions),
                )
            )
            self.action_priors = (
                1 - epsilon
            ) * self.base_action_priors + epsilon * self.action_groups.aggregate(
                self.dirichlet_noise
            )


@dataclasses.dataclass(slots=True)
class AfterStateNode:
    state: sj.Skyjo
    action: sj.SkyjoAction
    parent: DecisionStateNode
    _state_value: np.ndarray = dataclasses.field(init=False)
    children: dict[int, DecisionStateNode] = dataclasses.field(default_factory=dict)
    child_weights: dict[int, float] = dataclasses.field(default_factory=dict)
    child_weight_total: float = 0.0
    visit_count: int = 0
    is_expanded: bool = False
    all_children_discovered: bool = False

    def __post_init__(self):
        self._state_value = np.zeros(self.state.players, dtype=np.float32)

    @property
    def context(self):
        return self.parent.context

    @property
    def has_value_estimate(self):
        return self.is_expanded

    @property
    def state_value(self):
        return self._state_value

    def refresh_value(self):
        """Use fixed probabilities or empirical frequencies, never work counts."""
        total = np.zeros(self.state.players, dtype=np.float32)
        for key, child in self.children.items():
            total += self.child_weights[key] * child.state_value
        self._state_value = total / self.child_weight_total

    def _create_child(self, state):
        assert not sj.get_round_over(state)
        return DecisionStateNode(
            state,
            self,
            self.action,
            effective_merge_symmetric_actions=self.parent.effective_merge_symmetric_actions,
        )

    def _realize_outcome(self):
        state = sj.apply_action(self.state, self.action, rng=self.context.rng)
        key = sj.hash_skyjo(state)
        if key not in self.children:
            self.children[key] = self._create_child(state)
        return key

    def discover(self, discover_all_children=False):
        assert not self.children
        self.all_children_discovered = discover_all_children
        if discover_all_children:
            for state, probability in sj.chance_outcomes(self.state, self.action):
                key = sj.hash_skyjo(state)
                self.children[key] = self._create_child(state)
                self.child_weights[key] = self.child_weights.get(key, 0.0) + probability
            self.child_weight_total = 1.0
        else:
            key = self._realize_outcome()
            self.child_weights[key] = 1.0
            self.child_weight_total = 1.0

    def expand(self):
        assert not self.is_expanded
        self.refresh_value()
        self.is_expanded = True

    def select_child(self, update_child_weights=True):
        key = self._realize_outcome()
        if update_child_weights:
            self.child_weights[key] = self.child_weights.get(key, 0) + 1
            self.child_weight_total += 1
        return self.children[key]


@dataclasses.dataclass(slots=True)
class RoundBoundaryNode:
    """A revisitable leaf pooling outcomes independently of traversal counts."""

    pre_terminal_state: sj.Skyjo
    parent: DecisionStateNode
    action: sj.SkyjoAction
    visit_count: int = 0
    is_expanded: bool = False
    sample_count: int = 0
    sample_value_total: np.ndarray = dataclasses.field(init=False)
    deterministic_completed_state: sj.Skyjo | None = None

    def __post_init__(self):
        self.sample_value_total = np.zeros(self.state.players, dtype=np.float32)

    @property
    def context(self):
        return self.parent.context

    @property
    def state(self):
        return self.pre_terminal_state

    @property
    def has_value_estimate(self):
        return self.sample_count > 0

    @property
    def state_value(self):
        return (
            self.sample_value_total / self.sample_count
            if self.sample_count
            else np.zeros(self.state.players, dtype=np.float32)
        )

    def evaluate(self):
        """Pool fresh outcome evidence independently of traversal counts."""
        first = self.deterministic_completed_state
        if first is not None:
            if sj.get_game_over(first) and self.has_value_estimate:
                return Evidence(self.state_value.copy(), "exact")
        else:
            deterministic = not self.state.table[
                :, :, :, sj.FINGER_HIDDEN
            ].any() and not sj.is_action_random(self.action, self.state)
            first = sj.apply_action(self.state, self.action, rng=self.context.rng)
            if deterministic:
                self.deterministic_completed_state = first
        samples = self.context.config.boundary_samples if not self.sample_count else 1
        if self.deterministic_completed_state is not None:
            # A deterministic continuing ending still needs fresh next-deal values.
            completed = [first] * (1 if sj.get_game_over(first) else samples)
        else:
            completed = [first] + [
                sj.apply_action(self.state, self.action, rng=self.context.rng)
                for _ in range(samples - 1)
            ]
        assert self.context.boundary_evaluator is not None
        predictions = self.context.boundary_evaluator.evaluate(
            completed, self.context.rng
        )
        batch_total = predictions.sum(axis=0, dtype=np.float32)
        self.sample_value_total += batch_total
        self.sample_count += len(completed)
        self.is_expanded = True
        return Evidence(batch_total / len(completed))


MCTSNode = DecisionStateNode | AfterStateNode | RoundBoundaryNode


def find_leaf(root: MCTSNode, update_after_state_child_weights=False):
    path = [root]
    node = root
    while node.is_expanded and not isinstance(node, RoundBoundaryNode):
        node = node.select_child(update_child_weights=update_after_state_child_weights)
        path.append(node)
    return path


def backpropagate(search_path: list[MCTSNode], evidence: Evidence | None = None):
    """Count one traversal and refresh mixtures from the updated descendants."""
    if evidence is None:
        evidence = Evidence(search_path[-1].state_value.copy())
    for node in reversed(search_path):
        node.visit_count += 1
        if isinstance(node, DecisionStateNode):
            node.refresh_value(evidence)
        elif isinstance(node, AfterStateNode):
            node.refresh_value()


def run_mcts(
    game_state: sj.Skyjo,
    evaluator: Evaluator,
    iterations: int,
    *,
    config: SearchConfig = DEFAULT_SEARCH_CONFIG,
    boundary_evaluator: BoundaryEvaluator | None = None,
    rng: np.random.Generator | None = None,
    components: SearchComponents | None = None,
    cover_root: bool = False,
    symmetry_budget: int | None = None,
    observer: Callable[[DecisionStateNode], None] | None = None,
) -> DecisionStateNode:
    """Build and search a fresh tree from game_state, using fixed-seat values.

    An optional observer reads the live root after every completed backup. It
    must not mutate the tree or consume search RNG; no snapshots are copied here.
    A symmetry-safety retry supplies a new root starting at visit one; observers
    must discard statistics from the abandoned root.
    """
    if type(iterations) is not int or iterations < 0:
        raise ValueError("iterations must be a nonnegative integer")
    components = components if components is not None else resolve_components(config)
    rng = rng if rng is not None else np.random.default_rng()
    merging = config.merge_symmetric_actions and symmetry.safe_to_merge_actions(
        game_state, 0 if symmetry_budget is None else symmetry_budget
    )

    def build_tree(merge):
        return _search_tree(
            game_state,
            evaluator,
            iterations,
            config=config,
            boundary_evaluator=boundary_evaluator,
            rng=rng,
            components=components,
            cover_root=cover_root,
            merge=merge,
            observer=observer,
        )

    if merging and symmetry_budget is None:
        rng_state = deepcopy(rng.bit_generator.state)
        try:
            return build_tree(True)
        except _SymmetryRecyclingRisk:
            # Discard all grouped values and allocation sessions before retrying.
            rng.bit_generator.state = rng_state
        return build_tree(False)
    return build_tree(merging)


def _search_tree(
    game_state,
    evaluator,
    iterations,
    *,
    config,
    boundary_evaluator,
    rng,
    components,
    cover_root,
    merge,
    observer,
):
    coverage = (
        symmetry.ActionGroups.from_state(game_state, merge=merge).representatives
        if cover_root
        else ()
    )
    if len(coverage) > iterations:
        raise ValueError("Budget is too small to cover every root action group")
    context = SearchContext(
        config=config,
        boundary_evaluator=boundary_evaluator
        if boundary_evaluator is not None
        else NextDealEvaluator(evaluator),
        rng=rng,
        components=components,
        root_session=components.root_exploration.start(iterations - len(coverage), rng),
        interior_session=components.interior_exploration.start(iterations, rng),
    )
    root_node = DecisionStateNode(
        game_state,
        None,
        None,
        context,
        effective_merge_symmetric_actions=merge,
    )
    root_node.expand(evaluator.evaluate([game_state])[0])
    root_node.refresh_root_noise()
    for iteration in range(iterations):
        start = (
            root_node.children[coverage[iteration]]
            if iteration < len(coverage)
            else root_node
        )
        path = find_leaf(
            start,
            update_after_state_child_weights=not config.after_state_evaluate_all_children,
        )
        if start is not root_node:
            path.insert(0, root_node)
        leaf = path[-1]
        if isinstance(leaf, RoundBoundaryNode):
            evidence = leaf.evaluate()
        elif isinstance(leaf, AfterStateNode):
            leaf.discover(config.after_state_evaluate_all_children)
            children = list(leaf.children.values())
            for child, prediction in zip(
                children,
                evaluator.evaluate([child.state for child in children]),
                strict=True,
            ):
                child.expand(prediction)
            leaf.expand()
            evidence = Evidence(
                leaf.state_value.copy(),
                "expectation" if leaf.all_children_discovered else "sample",
            )
            if not leaf.all_children_discovered:
                # The first sampled outcome is part of this traversal, too.
                leaf = children[0]
                path.append(leaf)
        else:
            leaf.expand(evaluator.evaluate([leaf.state])[0])
            evidence = Evidence(leaf.model_prediction.value.copy())
        backpropagate(path, evidence)
        if context.root_session.observe is not None:
            context.root_session.observe(root_node.view())
        if context.interior_session.observe is not None:
            for visited in path[1:]:
                if isinstance(visited, DecisionStateNode):
                    context.interior_session.observe(visited.view())
        if observer is not None:
            observer(root_node)
    return root_node


def visualize_children(node: MCTSNode):
    if isinstance(node, RoundBoundaryNode):
        raise TypeError("Round boundaries have no children")
    for action, child in node.children.items():
        print(action, child.visit_count, child.state_value)
