"""Within-round Monte Carlo search with explicit evaluator and RNG ownership."""

from __future__ import annotations

import dataclasses

import numpy as np

from skyjo.engine import game as sj

from . import symmetry
from .evaluator import BoundaryEvaluator, Evaluator, NextDealEvaluator, Prediction


@dataclasses.dataclass(frozen=True, slots=True)
class SearchConfig:
    """Search settings; boundary_samples controls the first visit's batch only."""

    dirichlet_epsilon: float = 0.0
    after_state_evaluate_all_children: bool = False
    c_puct: float = 1.5
    fpu_reduction: float = 0.0
    boundary_samples: int = 1
    merge_symmetric_actions: bool = True

    def __post_init__(self):
        if (
            not np.isfinite(self.dirichlet_epsilon)
            or not 0 <= self.dirichlet_epsilon <= 1
        ):
            raise ValueError("dirichlet_epsilon must be between zero and one")
        if not np.isfinite(self.c_puct) or self.c_puct <= 0:
            raise ValueError("c_puct must be finite and positive")
        if not np.isfinite(self.fpu_reduction) or self.fpu_reduction < 0:
            raise ValueError("fpu_reduction must be finite and nonnegative")
        if type(self.boundary_samples) is not int or self.boundary_samples < 1:
            raise ValueError("boundary_samples must be a positive integer")
        if (
            type(self.merge_symmetric_actions) is not bool
            or type(self.after_state_evaluate_all_children) is not bool
        ):
            raise ValueError("chance and symmetry options must be boolean")


DEFAULT_SEARCH_CONFIG = SearchConfig()


@dataclasses.dataclass(frozen=True, slots=True)
class SearchContext:
    """Shared tree bindings. Cached values are valid only for these instances."""

    config: SearchConfig = dataclasses.field(default_factory=SearchConfig)
    evaluator: Evaluator | None = None
    boundary_evaluator: BoundaryEvaluator | None = None
    rng: np.random.Generator = dataclasses.field(default_factory=np.random.default_rng)


def ucb_score(child: MCTSNode, parent: DecisionStateNode) -> float:
    config = parent.context.config
    value = (
        float(child.state_value[sj.get_player(parent.state)])
        if child.has_value_estimate
        else parent.model_value_for_current_player - config.fpu_reduction
    )
    return value + config.c_puct * parent.action_probability(child.action) * np.sqrt(
        parent.visit_count + 1
    ) / (1 + child.visit_count)


@dataclasses.dataclass(slots=True)
class DecisionStateNode:
    state: sj.Skyjo
    parent: DecisionStateNode | AfterStateNode | None
    action: sj.SkyjoAction | None
    context: SearchContext | None = None
    effective_merge_symmetric_actions: bool = False
    state_value_total: np.ndarray = dataclasses.field(init=False)
    model_prediction: Prediction | None = None
    children: dict[int, MCTSNode] = dataclasses.field(default_factory=dict)
    visit_count: int = 0
    is_expanded: bool = False
    is_retired: bool = False
    dirichlet_noise: np.ndarray = dataclasses.field(init=False)
    action_groups: symmetry.ActionGroups | None = None
    model_action_priors: np.ndarray | None = None
    action_priors: np.ndarray | None = None

    def __post_init__(self):
        if self.context is None:
            self.context = (
                self.parent.context if self.parent is not None else SearchContext()
            )
        self.state_value_total = np.zeros(self.state.players, dtype=np.float32)
        self.dirichlet_noise = np.zeros(sj.MASK_SIZE, dtype=np.float32)

    @property
    def has_value_estimate(self):
        return self.model_prediction is not None

    @property
    def model_value_for_current_player(self):
        assert self.model_prediction is not None
        return float(self.model_prediction.value[sj.get_player(self.state)])

    @property
    def state_value(self):
        if self.visit_count:
            return self.state_value_total / self.visit_count
        if self.model_prediction is not None:
            return self.model_prediction.value
        return np.zeros(self.state.players, dtype=np.float32)

    def expand(self, model_prediction: Prediction):
        assert not self.is_expanded
        self.model_prediction = model_prediction
        self.action_groups = symmetry.ActionGroups.from_state(
            self.state, merge=self.effective_merge_symmetric_actions
        )
        self.model_action_priors = self.action_groups.aggregate(model_prediction.policy)
        self.action_priors = self.model_action_priors
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
        scores = [(ucb_score(child, self), child) for child in self.children.values()]
        maximum = max(score for score, _ in scores)
        candidates = [child for score, child in scores if abs(maximum - score) < 1e-5]
        return (
            candidates[int(self.context.rng.integers(len(candidates)))]
            if len(candidates) > 1
            else candidates[0]
        )

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
        self.action_priors = self.model_action_priors
        if epsilon:
            actions = np.flatnonzero(sj.actions(self.state))
            self.dirichlet_noise[actions] = self.context.rng.dirichlet(
                np.full(len(actions), 10 / len(actions))
            )
            self.action_priors = (
                1 - epsilon
            ) * self.model_action_priors + epsilon * self.action_groups.aggregate(
                self.dirichlet_noise
            )

    def validate_search_extension(
        self, state, iterations, config, evaluator, boundary_evaluator, rng
    ):
        """Reject incompatible cache reuse before consuming randomness or mutating."""
        if self.is_retired:
            raise ValueError("reused root is retired after subtree promotion")
        if state is not self.state and any(
            not np.array_equal(
                getattr(state, field.name), getattr(self.state, field.name)
            )
            for field in dataclasses.fields(sj.Skyjo)
        ):
            raise ValueError("reused root state does not match")
        if self.context.config != config:
            raise ValueError("reused root search configuration does not match")
        if self.context.evaluator is not evaluator:
            raise ValueError("reused root evaluator does not match")
        if (
            boundary_evaluator is not None
            and self.context.boundary_evaluator is not boundary_evaluator
        ):
            raise ValueError("reused root boundary evaluator does not match")
        if rng is not None and self.context.rng is not rng:
            raise ValueError("reused root random generator does not match")
        if (
            self.effective_merge_symmetric_actions
            and not symmetry.safe_to_merge_actions(state, self.visit_count + iterations)
        ):
            raise ValueError("reused pooled root exceeds its safe search budget")

    def promote_to_root(self):
        """Retire ancestors whose cached statistics will no longer be updated."""
        ancestor = self.parent
        while ancestor is not None:
            if isinstance(ancestor, DecisionStateNode):
                ancestor.is_retired = True
            ancestor = ancestor.parent
        self.parent = None


@dataclasses.dataclass(slots=True)
class AfterStateNode:
    state: sj.Skyjo
    action: sj.SkyjoAction
    parent: DecisionStateNode
    state_value_total: np.ndarray = dataclasses.field(init=False)
    children: dict[int, DecisionStateNode] = dataclasses.field(default_factory=dict)
    child_weights: dict[int, float] = dataclasses.field(default_factory=dict)
    child_weight_total: float = 0.0
    visit_count: int = 0
    is_expanded: bool = False
    all_children_discovered: bool = False

    def __post_init__(self):
        self.state_value_total = np.zeros(self.state.players, dtype=np.float32)

    @property
    def context(self):
        return self.parent.context

    @property
    def has_value_estimate(self):
        return self.is_expanded

    @property
    def state_value(self):
        if not self.is_expanded:
            return np.zeros(self.state.players, dtype=np.float32)
        if self.all_children_discovered or not self.visit_count:
            return self.state_value_total
        return self.state_value_total / self.visit_count

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
        if self.all_children_discovered:
            for key, child in self.children.items():
                self.state_value_total += (
                    self.child_weights[key]
                    * child.state_value
                    / self.child_weight_total
                )
        self.is_expanded = True

    def select_child(self, update_child_weights=True):
        key = self._realize_outcome()
        if update_child_weights:
            self.child_weights[key] = self.child_weights.get(key, 0) + 1
            self.child_weight_total += 1
        return self.children[key]

    def update_exact_child(self, child, previous_child_value):
        assert self.all_children_discovered
        self.state_value_total += (
            (child.state_value - previous_child_value)
            * self.child_weights[sj.hash_skyjo(child.state)]
            / self.child_weight_total
        )


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
        """Update the pooled estimate, returning only this visit's fresh mean."""
        first = self.deterministic_completed_state
        if first is not None:
            if sj.get_game_over(first) and self.has_value_estimate:
                return self.state_value
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
        return batch_total / len(completed)


MCTSNode = DecisionStateNode | AfterStateNode | RoundBoundaryNode


def find_leaf(root: MCTSNode, update_after_state_child_weights=False):
    path = [root]
    node = root
    while node.is_expanded and not isinstance(node, RoundBoundaryNode):
        node = node.select_child(update_child_weights=update_after_state_child_weights)
        path.append(node)
    return path


def backpropagate(search_path: list[MCTSNode], value: np.ndarray):
    """Back up one fresh return, substituting expectations at exact chance nodes."""
    effective_value = value
    previous_node = previous_old_value = None
    for node in reversed(search_path):
        old_value = node.state_value.copy()
        if isinstance(node, DecisionStateNode):
            node.state_value_total += effective_value
        elif isinstance(node, AfterStateNode) and previous_node is not None:
            if node.all_children_discovered:
                node.update_exact_child(previous_node, previous_old_value)
                effective_value = node.state_value.copy()
            else:
                node.state_value_total += effective_value
        node.visit_count += 1
        previous_node, previous_old_value = node, old_value


def run_mcts(
    game_state: sj.Skyjo,
    evaluator: Evaluator,
    iterations: int,
    *,
    config: SearchConfig = DEFAULT_SEARCH_CONFIG,
    boundary_evaluator: BoundaryEvaluator | None = None,
    rng: np.random.Generator | None = None,
    root_node: DecisionStateNode | None = None,
) -> DecisionStateNode:
    """Search one round, using fixed-seat values throughout the tree.

    Reuse requires the same configuration and evaluator objects. Omitting rng
    or boundary_evaluator on reuse keeps the original tree's binding.
    Promoting a subtree retires its ancestors; their cached values become stale.
    """
    if type(iterations) is not int or iterations < 0:
        raise ValueError("iterations must be a nonnegative integer")
    if root_node is None:
        context = SearchContext(
            config,
            evaluator,
            boundary_evaluator
            if boundary_evaluator is not None
            else NextDealEvaluator(evaluator),
            rng if rng is not None else np.random.default_rng(),
        )
        root_node = DecisionStateNode(
            game_state,
            None,
            None,
            context,
            effective_merge_symmetric_actions=config.merge_symmetric_actions
            and symmetry.safe_to_merge_actions(game_state, iterations),
        )
        root_node.expand(evaluator.evaluate([game_state])[0])
    else:
        if not isinstance(root_node, DecisionStateNode):
            raise TypeError("root_node must be a DecisionStateNode")
        root_node.validate_search_extension(
            game_state, iterations, config, evaluator, boundary_evaluator, rng
        )
        root_node.promote_to_root()
    root_node.refresh_root_noise()
    for _ in range(iterations):
        path = find_leaf(
            root_node,
            update_after_state_child_weights=not config.after_state_evaluate_all_children,
        )
        leaf = path[-1]
        if isinstance(leaf, RoundBoundaryNode):
            value = leaf.evaluate()
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
            if not leaf.all_children_discovered:
                # The first sampled outcome is part of this traversal, too.
                leaf = children[0]
                path.append(leaf)
            value = leaf.state_value.copy()
        else:
            leaf.expand(evaluator.evaluate([leaf.state])[0])
            value = leaf.state_value.copy()
        backpropagate(path, value)
    return root_node


def visualize_children(node: MCTSNode):
    if isinstance(node, RoundBoundaryNode):
        raise TypeError("Round boundaries have no children")
    for action, child in node.children.items():
        print(action, child.visit_count, child.state_value)
