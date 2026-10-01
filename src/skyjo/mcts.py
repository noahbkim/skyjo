"""Monte Carlo Tree Search Implementation for Skyjo."""

from __future__ import annotations

import dataclasses
import typing

import numpy as np
import torch

from . import config
from . import predictor
from . import game as sj
from . import skynet

# MARK: Config


@dataclasses.dataclass(slots=True)
class MCTSConfig(config.Config):
    iterations: int
    dirichlet_epsilon: float
    after_state_evaluate_all_children: bool
    c_puct: float = 1.5
    fpu_reduction: float = 0.0
    score_utility_weight: float = 0.0


# MARK: NODE SCORING


def ucb_score(
    child: MCTSNode,
    parent: DecisionStateNode,
) -> float:
    assert isinstance(parent, DecisionStateNode), (
        f"Parent must be DecisionStateNode, got {type(parent)}"
    )
    action_probability = parent.action_probability(child.action)
    if child.has_value_estimate:
        value = skynet.state_value_for_player(
            child.state_value, sj.get_player(parent.state)
        )
    else:
        value = parent.model_value_for_current_player - parent.fpu_reduction
    exploration = (
        parent.c_puct
        * action_probability
        * np.sqrt(parent.visit_count + 1)
        / (1 + child.visit_count)
    )
    return value + exploration - getattr(child, "virtual_loss", 0.0)


# MARK: NODES


@dataclasses.dataclass(slots=True)
class DecisionStateNode:
    state: sj.Skyjo
    parent: DecisionStateNode | AfterStateNode | None
    action: sj.SkyjoAction | None  # previous action
    state_value_total: skynet.StateValue | None = None  # initialized in __post_init__
    model_prediction: skynet.SkyNetPrediction | None = None
    children: dict[sj.SkyjoAction, MCTSNode] = dataclasses.field(default_factory=dict)
    visit_count: int = 0
    virtual_loss_total: float = 0.0
    is_expanded: bool = False
    are_children_discovered: bool = False
    dirichlet_noise: np.ndarray[tuple[int], np.float32] | None = None
    dirichlet_epsilon: float = 0.0
    c_puct: float = 1.5
    fpu_reduction: float = 0.0
    score_utility_weight: float = 0.0

    def __post_init__(self):
        # need to initialize here because we don't know the player count until after we have the state
        self.state_value_total = np.zeros(
            sj.get_player_count(self.state), dtype=np.float32
        )
        self.dirichlet_noise = np.zeros(sj.MASK_SIZE, dtype=np.float32)

    def __str__(self) -> str:
        return (
            f"DecisionStateNode\n"
            f"{sj.visualize_state(self.state)}\n"
            f"Visit Count: {self.visit_count}\n"
            f"State Value: {self.state_value}\n"
            f"Is Expanded: {self.is_expanded}\n"
            f"Model Prediction: {self.model_prediction}\n"
            f"Children visit counts: {self.policy_targets() * sum(child.visit_count for child in self.children.values())}\n"
        )

    @property
    def total_count(self) -> int:
        return self.visit_count

    @property
    def child_count(self) -> int:
        return self.visit_count

    @property
    def virtual_loss(self) -> float:
        return self.virtual_loss_total

    @property
    def has_value_estimate(self) -> bool:
        return self.model_prediction is not None

    @property
    def model_value_for_current_player(self) -> float:
        assert self.model_prediction is not None, "expected an expanded decision node"
        return self.model_prediction.search_value(self.score_utility_weight)[0].item()

    @property
    def state_value(self) -> skynet.StateValue:
        if self.visit_count == 0:
            if self.model_prediction is not None:
                return skynet.to_state_value(
                    self.model_prediction.search_value(self.score_utility_weight),
                    sj.get_player(self.state),
                )
            return np.zeros(sj.get_player_count(self.state), dtype=np.float32)
        return self.state_value_total / self.visit_count

    def _select_highest_ucb_child(self) -> MCTSNode:
        child_ucbs = [
            (ucb_score(child, self), child) for child in self.children.values()
        ]
        max_ucb = max(child_ucbs, key=lambda x: x[0])[0]
        candidates = [item[1] for item in child_ucbs if abs(max_ucb - item[0]) < 1e-5]
        if len(candidates) == 1:
            return candidates[0]
        return np.random.choice(candidates)

    def highest_visit_child(self) -> MCTSNode:
        return max(self.children.values(), key=lambda x: x.visit_count)

    def expand(
        self,
        model_prediction: skynet.SkyNetPrediction,
    ) -> None:
        """Expand node by evaluating state with model"""
        assert not self.is_expanded, "Node already expanded"
        assert self.model_prediction is None, "Model prediction already set"
        self.model_prediction = model_prediction
        self.is_expanded = True
        for action in sj.get_actions(self.state):
            self.children[action] = self.create_child_node(
                action
            )

    def select_child(self, **kwargs) -> MCTSNode:
        return self._select_highest_ucb_child()

    def create_child_node(
        self,
        action: sj.SkyjoAction,
    ) -> MCTSNode:
        if sj.get_round_about_to_end(self.state):
            return RoundBoundaryNode(
                pre_terminal_state=self.state,
                parent=self,
                action=action,
            )

        if sj.is_action_random(action, self.state):
            return AfterStateNode(state=self.state, action=action, parent=self)

        next_state = sj.apply_action(self.state, action)
        return DecisionStateNode(
            state=next_state,
            parent=self,
            action=action,
            c_puct=self.c_puct,
            fpu_reduction=self.fpu_reduction,
            score_utility_weight=self.score_utility_weight,
        )

    def policy_targets(
        self, temperature: float = 1.0
    ) -> np.ndarray[tuple[int], np.float32]:
        visit_counts = np.zeros((sj.MASK_SIZE,), dtype=np.float32)
        for action, child in self.children.items():
            visit_counts[action] = child.visit_count

        if temperature == 0:
            visit_probabilities = np.zeros(visit_counts.shape, dtype=np.float32)
            visit_probabilities[visit_counts.argmax().item()] = 1
            return visit_probabilities
        visit_probabilities = visit_counts ** (1 / temperature)
        visit_probabilities = visit_probabilities / visit_probabilities.sum()
        return visit_probabilities

    def action_probability(self, action) -> float:
        assert self.model_prediction is not None, (
            "Model prediction must be set before calling"
        )
        return (
            self.model_prediction.policy_output[action].item()
            * (1 - self.dirichlet_epsilon)
            + self.dirichlet_epsilon * self.dirichlet_noise[action]
        )


@dataclasses.dataclass(slots=True)
class AfterStateNode:
    state: sj.Skyjo
    action: sj.SkyjoAction
    parent: DecisionStateNode
    state_value_total: skynet.StateValue | None = None
    children: dict[int, DecisionStateNode | RoundBoundaryNode] = dataclasses.field(
        default_factory=dict
    )
    child_weights: dict[int, float] = dataclasses.field(default_factory=dict)
    child_weight_total: float = 0.0
    visit_count: int = 0
    virtual_loss_total: float = 0.0
    is_expanded: bool = False
    all_children_discovered: bool = False

    def __post_init__(self):
        self.state_value_total = np.zeros(
            sj.get_player_count(self.state), dtype=np.float32
        )

    def __str__(self) -> str:
        return (
            f"AfterStateNode\n"
            f"{sj.visualize_state(self.state)}\n"
            f"Action: {sj.get_action_name(self.action)}\n"
            f"Visit Count: {self.visit_count}\n"
            f"Value: {self.state_value}\n"
            f"Is Expanded: {self.is_expanded}\n"
            f"Children: {len(self.children)}\n"
        )

    @property
    def child_count(self) -> int:
        return self.visit_count

    @property
    def virtual_loss(self) -> float:
        return self.virtual_loss_total

    @property
    def has_value_estimate(self) -> bool:
        return self.is_expanded

    @property
    def state_value(self) -> skynet.StateValue:
        if not self.is_expanded:
            return np.zeros(sj.get_player_count(self.state), dtype=np.float32)
        # If all children discovered and exact probabilities were accounted for
        # we can just return the exact weighted total state value
        if (
            self.all_children_discovered
            or self.visit_count == 0
            or sj.get_round_about_to_end(self.state)
        ):
            return self.state_value_total
        return (
            self.state_value_total / self.visit_count
        )  # visit_count == realized_count_total

    def _create_child(self, state: sj.Skyjo) -> MCTSNode:
        assert not sj.get_round_over(state), (
            "Create a round boundary node explicitly instead"
        )
        return DecisionStateNode(
            state=state,
            parent=self,
            action=self.action,
            c_puct=self.parent.c_puct,
            fpu_reduction=self.parent.fpu_reduction,
            score_utility_weight=self.parent.score_utility_weight,
        )

    def realize_outcomes(self, n) -> None:
        for _ in range(n):
            _ = self._realize_outcome()

    def _realize_outcome(self) -> sj.Skyjo:
        outcome_state = sj.apply_action(self.state, self.action)
        assert not sj.get_round_over(outcome_state), (
            "Create a round boundary node explicitly instead"
        )
        outcome_state_hash = sj.hash_skyjo(outcome_state)
        if outcome_state_hash not in self.children:
            self.children[outcome_state_hash] = self._create_child(outcome_state)
        return outcome_state

    def _expand_single_child(self) -> None:
        # Realize a single next child state. Child weights are now observered
        # frequencies of the child states.
        next_state = self._realize_outcome()
        next_state_hash = sj.hash_skyjo(next_state)
        self.child_weights[next_state_hash] = 1
        self.child_weight_total += 1

    def _expand_all_possible_children(self) -> None:
        self.all_children_discovered = True
        # Realize all possible child states. Child weights are now exactly the
        # probabilities of those child states computed from the deck card counts.
        cards_remaining = np.sum(sj.get_deck(self.state))
        for card, card_count in enumerate(sj.get_deck(self.state)):
            if card_count > 0:
                next_state = sj.apply_action(
                    sj.preordain(self.state, card), self.action
                )
                child = self._create_child(next_state)
                self.children[sj.hash_skyjo(next_state)] = child
                # Child weight is the probability of that card being next
                self.child_weights[sj.hash_skyjo(next_state)] = (
                    card_count / cards_remaining
                )
        self.child_weight_total = 1.0

    def _compute_state_value_from_children(self) -> skynet.StateValue:
        state_value = np.zeros(sj.get_player_count(self.state), dtype=np.float32)
        for key_hash, child in self.children.items():
            state_value += (
                child.state_value
                * self.child_weights[key_hash]
                / self.child_weight_total
            )
        return state_value

    def highest_visit_child(self) -> MCTSNode:
        return max(self.children.values(), key=lambda x: x.visit_count)

    def discover(
        self,
        discover_all_children: bool = False,
    ):
        """Expands node after all initial children values are ready"""
        assert len(self.children) == 0, "Children not empty"
        if discover_all_children:
            self.all_children_discovered = True
            self._expand_all_possible_children()
        else:
            self._expand_single_child()

    def expand(self):
        assert not self.is_expanded, "Node already expanded"
        for child_hash, child in self.children.items():
            self.state_value_total += (
                self.child_weights[child_hash]
                * child.state_value
                / self.child_weight_total
            )
        self.is_expanded = True

    def select_child(
        self, update_child_weights: bool = True
    ) -> DecisionStateNode | RoundBoundaryNode:
        """Realize next state by applying action. Returns node in game tree that represents realized next state."""
        realized_next_state = self._realize_outcome()
        next_state_hash = sj.hash_skyjo(realized_next_state)
        if update_child_weights:
            self.child_weights[next_state_hash] = (
                self.child_weights.get(next_state_hash, 0) + 1
            )
            self.child_weight_total += 1
        return self.children[next_state_hash]

    def update_exact_child(
        self,
        child: DecisionStateNode,
        previous_child_value: skynet.StateValue,
    ) -> None:
        """Update an exact chance expectation after one child value changes."""
        assert isinstance(child, DecisionStateNode), (
            f"Child must be a DecisionStateNode, got {type(child)} instead",
        )
        assert self.all_children_discovered, "expected an exact chance node"
        self.state_value_total += (
            (child.state_value - previous_child_value)
            * self.child_weights[sj.hash_skyjo(child.state)]
            / self.child_weight_total
        )


@dataclasses.dataclass(slots=True)
class RoundBoundaryNode:
    """A round boundary with one cached sample; never expanded into another round."""

    pre_terminal_state: sj.Skyjo
    parent: AfterStateNode | DecisionStateNode
    action: sj.SkyjoAction
    visit_count: int = 0
    virtual_loss: float = 0.0
    is_expanded: bool = False
    next_round_state: sj.Skyjo | None = None
    value: skynet.StateValue | None = None

    @property
    def child_count(self) -> int:
        return self.visit_count

    @property
    def has_value_estimate(self) -> bool:
        return self.value is not None

    @property
    def state(self) -> sj.Skyjo:
        return self.pre_terminal_state

    @property
    def state_value(self) -> skynet.StateValue:
        if self.value is None:
            return np.zeros(sj.get_player_count(self.state), dtype=np.float32)
        return self.value

    def prepare(self) -> sj.Skyjo | None:
        """Sample the boundary once, returning a playable state if inference is needed."""
        if self.value is not None:
            return None
        if self.next_round_state is None:
            completed = sj.apply_action(self.state, self.action)
            if sj.get_game_over(completed):
                self.value = skynet.skyjo_to_game_state_value(completed)
                return None
            self.next_round_state = sj.start_next_round(completed)
        return self.next_round_state

    def set_prediction(self, prediction: skynet.SkyNetPrediction) -> None:
        assert self.next_round_state is not None and self.value is None
        self.value = skynet.to_state_value(
            prediction.value_output, sj.get_player(self.next_round_state)
        ).copy()


# MARK: MCTS Algorithm


def find_leaf(
    root: MCTSNode,
    update_after_state_child_weights: bool = False,
):
    search_path = [root]
    node = root
    while node.is_expanded:
        node = node.select_child(update_child_weights=update_after_state_child_weights)
        search_path.append(node)
    return search_path


def backpropagate(search_path: list[MCTSNode], value: skynet.StateValue) -> None:
    """Back up one traversal return using edge visits consistently."""
    effective_value = value
    previous_node: MCTSNode | None = None
    previous_node_old_value: skynet.StateValue | None = None
    for node in reversed(search_path):
        original_state_value = node.state_value.copy()
        if isinstance(node, DecisionStateNode):
            node.state_value_total += effective_value
        elif isinstance(node, AfterStateNode) and previous_node is not None:
            assert isinstance(previous_node, DecisionStateNode)
            assert previous_node_old_value is not None
            if node.all_children_discovered:
                node.update_exact_child(previous_node, previous_node_old_value)
                effective_value = node.state_value.copy()
            else:
                node.state_value_total += effective_value
        node.visit_count += 1
        previous_node = node
        previous_node_old_value = original_state_value


def run_mcts(
    game_state: sj.Skyjo,
    predictor_client: predictor.AbstractPredictorClient,
    iterations: int,
    *,
    dirichlet_epsilon: float = 0.0,
    after_state_evaluate_all_children: bool = False,
    c_puct: float = 1.5,
    fpu_reduction: float = 0.0,
    score_utility_weight: float = 0.0,
    root_node: MCTSNode | None = None,
) -> MCTSNode:
    """Search within a round, using one cached game-value sample at its boundary.

    A continuing round bootstraps from the model's value of the next deal.
    Ordinary in-round chance nodes retain their existing sampling behavior.
    """
    if c_puct <= 0:
        raise ValueError("c_puct must be positive")
    if fpu_reduction < 0:
        raise ValueError("fpu_reduction cannot be negative")
    if score_utility_weight != 0:
        raise ValueError("Full-game search requires score_utility_weight=0")

    # Get model prediction for root state
    if root_node is None:
        _ = predictor_client.put(game_state)
        predictor_client.send()
        _, prediction = predictor_client.get()
        root_node = DecisionStateNode(
            state=game_state,
            parent=None,
            action=None,
            c_puct=c_puct,
            fpu_reduction=fpu_reduction,
            score_utility_weight=score_utility_weight,
        )
        root_node.expand(model_prediction=prediction)

    else:
        if not isinstance(root_node, DecisionStateNode):
            raise TypeError("root_node must be a DecisionStateNode")
        if (
            root_node.c_puct != c_puct
            or root_node.fpu_reduction != fpu_reduction
            or root_node.score_utility_weight != score_utility_weight
        ):
            raise ValueError("reused root scoring configuration does not match")

    # Root noise is local to this search invocation.
    root_node.dirichlet_noise.fill(0)
    root_node.dirichlet_epsilon = dirichlet_epsilon
    if dirichlet_epsilon > 0:
        valid_action_count = sj.actions(root_node.state).sum().item()
        dirichlet_noise = np.random.dirichlet(
            np.ones(valid_action_count) * 10 / valid_action_count,
        )
        root_node.dirichlet_noise[sj.get_actions(root_node.state)] = dirichlet_noise

    search_depths = []
    for _ in range(iterations):
        search_path = find_leaf(
            root_node,
            update_after_state_child_weights=not after_state_evaluate_all_children,
        )
        search_depths.append(len(search_path))
        leaf = search_path[-1]

        if isinstance(leaf, RoundBoundaryNode):
            next_round = leaf.prepare()
            if next_round is not None:
                prediction_id = predictor_client.put(next_round)
                predictor_client.send()
                returned_id, prediction = predictor_client.get()
                assert prediction_id == returned_id
                leaf.set_prediction(prediction)
            backup_value = leaf.state_value.copy()

        # AFTER STATE LEAF
        # We want to pre-expand the afterstate and either realize all potential
        # outcomes or just roll a single next state based on parameter
        #
        # We also need to queue all the children decision state for model prediction
        elif isinstance(leaf, AfterStateNode):
            leaf.discover(discover_all_children=after_state_evaluate_all_children)

            afterstate_prediction_ids = {}
            # Add realized outcome children to prediction
            for hash_, child in leaf.children.items():
                prediction_id = predictor_client.put(child.state)
                afterstate_prediction_ids[prediction_id] = hash_

            if afterstate_prediction_ids:
                predictor_client.send()

            for prediction_id, prediction in predictor_client.get_all():
                child_hash = afterstate_prediction_ids[prediction_id]
                child = leaf.children[child_hash]
                child.expand(model_prediction=prediction)
                del afterstate_prediction_ids[prediction_id]

            leaf.expand()
            backup_value = leaf.state_value.copy()

        # DECISION STATE LEAF
        # We want to queue the decision state for model prediction. Also
        # we want to pre-expand the decision state, so that parallel threads
        # can go deeper and queue a child state for model prediction.
        elif isinstance(leaf, DecisionStateNode):
            prediction_id = predictor_client.put(leaf.state)
            predictor_client.send()
            returned_prediction_id, prediction = predictor_client.get()
            assert prediction_id == returned_prediction_id, (
                f"Returned prediction id: {returned_prediction_id} "
                f"does NOT match given prediction id: {prediction_id}"
            )
            leaf.expand(model_prediction=prediction)
            backup_value = leaf.state_value.copy()
        else:
            backup_value = leaf.state_value.copy()
        backpropagate(search_path, backup_value)
    # print(sum(search_depths) / len(search_depths))
    return root_node


# MARK: TYPES

MCTSNode: typing.TypeAlias = DecisionStateNode | AfterStateNode | RoundBoundaryNode


if __name__ == "__main__":
    import explain

    np.random.seed(42)
    torch.manual_seed(42)
    players = 2
    model = skynet.SimpleSkyNet(
        spatial_input_shape=(players, sj.ROW_COUNT, sj.COLUMN_COUNT, sj.FINGER_SIZE),
        non_spatial_input_shape=skynet.get_non_spatial_input_shape(players),
        value_output_shape=(players,),
        policy_output_shape=(sj.MASK_SIZE,),
        hidden_layers=[64, 64],
        device=torch.device("cpu"),
    )
    predictor_client = predictor.NaivePredictorClient(model)
    winning_state = explain.create_almost_surely_winning_position()
    root_node = run_mcts(
        sj.apply_action(sj.apply_action(winning_state, sj.MASK_TAKE), sj.MASK_SIZE - 1),
        predictor_client,
        iterations=1600,
    )
    print(root_node)

# MARK: Debugging


def visualize_children(node: MCTSNode):
    assert not isinstance(node, RoundBoundaryNode), (
        "Terminal state nodes have no children"
    )
    if isinstance(node, DecisionStateNode):
        for action, child in node.children.items():
            print(sj.get_action_name(action))
            print(ucb_score(child, node))
            print(child)
    elif isinstance(node, AfterStateNode):
        for hash_state, child in node.children.items():
            print(child)
    else:
        raise ValueError(f"Unknown node type: {type(node)}")
