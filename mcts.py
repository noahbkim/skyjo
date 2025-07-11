"""Monte Carlo Tree Search Implementation for Skyjo."""

from __future__ import annotations

import dataclasses
import typing

import numpy as np
import torch

import config
import predictor
import skyjo as sj
import skynet

# MARK: Config


@dataclasses.dataclass(slots=True)
class MCTSConfig(config.Config):
    iterations: int
    dirichlet_epsilon: float
    after_state_evaluate_all_children: bool
    terminal_state_initial_rollouts: int
    forced_playout_k: float | None = None


# MARK: NODE SCORING


def ucb_score(
    child: MCTSNode,
    parent: DecisionStateNode,
    forced_playout_k: float | None = None,
) -> float:
    assert isinstance(parent, DecisionStateNode), (
        f"Parent must be DecisionStateNode, got {type(parent)}"
    )
    action_probability = parent.action_probability(child.action)
    if (
        forced_playout_k is not None
        and child.visit_count > 0
        and child.visit_count
        < np.sqrt(forced_playout_k * parent.visit_count * action_probability)
    ):
        return 1000
    return skynet.state_value_for_player(
        child.state_value, sj.get_player(parent.state)
    ) + action_probability * np.sqrt(parent.total_count) / (1 + child.child_count)


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
    is_expanded: bool = False
    are_children_discovered: bool = False
    dirichlet_noise: np.ndarray[tuple[int], np.float32] | None = None
    dirichlet_epsilon: float = 0.0
    forced_playout_k: float | None = None

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
            f"Forced Playout K: {self.forced_playout_k}\n"
            f"Children visit counts: {self.policy_targets() * sum(child.visit_count for child in self.children.values())}\n"
        )

    @property
    def total_count(self) -> int:
        # TODO: optimize
        if sj.get_game_about_to_end(self.state):
            return sum(child.outcome_count for child in self.children.values())
        return self.visit_count

    @property
    def child_count(self) -> int:
        return self.visit_count

    @property
    def state_value(self) -> skynet.StateValue:
        if self.visit_count == 0:
            if self.model_prediction is not None:
                return skynet.to_state_value(
                    self.model_prediction.value_output,
                    sj.get_player(self.state),
                )
            return np.ones(
                sj.get_player_count(self.state), dtype=np.float32
            ) / sj.get_player_count(self.state)
        return self.state_value_total / self.visit_count

    def _select_highest_ucb_child(self) -> MCTSNode:
        child_ucbs = [
            (ucb_score(child, self, self.forced_playout_k), child)
            for child in self.children.values()
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
        terminal_state_rollouts: int = 1,
    ) -> None:
        """Expand node by evaluating state with model"""
        assert not self.is_expanded, "Node already expanded"
        assert self.model_prediction is None, "Model prediction already set"
        self.model_prediction = model_prediction
        self.is_expanded = True
        for action in sj.get_actions(self.state):
            self.children[action] = self.create_child_node(
                action, terminal_state_initial_rollouts=terminal_state_rollouts
            )

    def select_child(self, **kwargs) -> MCTSNode:
        highest_ucb_child = self._select_highest_ucb_child()
        if (
            isinstance(highest_ucb_child, TerminalStateNode)
            and highest_ucb_child.is_random
        ):
            highest_ucb_child.realize_outcome()
        return highest_ucb_child

    def create_child_node(
        self,
        action: sj.SkyjoAction,
        terminal_state_initial_rollouts: int = 1,
    ) -> MCTSNode:
        if sj.get_game_about_to_end(self.state):
            return TerminalStateNode(
                pre_terminal_state=self.state,
                parent=self,
                action=action,
                is_random=sj.is_action_random(action, self.state),
                initial_rollouts=terminal_state_initial_rollouts,
            )

        if sj.is_action_random(action, self.state):
            return AfterStateNode(state=self.state, action=action, parent=self)

        next_state = sj.apply_action(self.state, action)
        return DecisionStateNode(
            state=next_state,
            parent=self,
            action=action,
        )

    def policy_targets(
        self, temperature: float = 1.0, forced_playout_k: float | None = None
    ) -> np.ndarray[tuple[int], np.float32]:
        visit_counts = np.zeros((sj.MASK_SIZE,), dtype=np.float32)
        for action, child in self.children.items():
            visit_counts[action] = child.visit_count

        if forced_playout_k is not None:
            most_visited_child = self.highest_visit_child()
            most_visited_child_ucb = ucb_score(most_visited_child, self)
            for action, child in self.children.items():
                child_value = skynet.state_value_for_player(
                    child.state_value, sj.get_player(self.state)
                )
                if action == most_visited_child.action or child.visit_count == 0:
                    continue
                forced_visits = np.floor(
                    (
                        self.action_probability(action) * np.sqrt(self.visit_count)
                        + child_value * (1 + child.visit_count)
                        - most_visited_child_ucb * (1 + child.visit_count)
                    )
                    / (child_value - most_visited_child_ucb)
                )
                visit_counts[action] -= max(0, forced_visits)

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
    children: dict[sj.Skyjo, DecisionStateNode | TerminalStateNode] = dataclasses.field(
        default_factory=dict
    )
    child_weights: dict[sj.Skyjo, int] = dataclasses.field(default_factory=dict)
    child_weight_total: int = 0
    visit_count: int = 0
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
    def state_value(self) -> skynet.StateValue:
        if not self.is_expanded:
            return np.ones(
                sj.get_player_count(self.state), dtype=np.float32
            ) / sj.get_player_count(self.state)
        # If all children discovered and exact probabilities were accounted for
        # we can just return the exact weighted total state value
        if (
            self.all_children_discovered
            or self.visit_count == 0
            or sj.get_game_about_to_end(self.state)
        ):
            return self.state_value_total
        return (
            self.state_value_total / self.visit_count
        )  # visit_count == realized_count_total

    @property
    def weighted_model_prediction_value(self) -> skynet.StateValue:
        assert all(
            child.model_prediction is not None for child in self.children.values()
        ), "All children must have a model prediction"
        return sum(
            child.model_prediction.value_output
            * self.child_weights[hash_]
            / self.child_weight_total
            for hash_, child in self.children.items()
        )

    def _create_child(self, state: sj.Skyjo) -> MCTSNode:
        assert not sj.get_game_over(state), (
            "Create terminal state node explicitly instead"
        )
        return DecisionStateNode(
            state=state,
            parent=self,
            action=self.action,
        )

    def realize_outcomes(self, n) -> None:
        for _ in range(n):
            _ = self._realize_outcome()

    def _realize_outcome(self) -> sj.Skyjo:
        outcome_state = sj.apply_action(self.state, self.action)
        assert not sj.get_game_over(outcome_state), (
            "Create terminal state node explicitly instead"
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
    ) -> DecisionStateNode | TerminalStateNode:
        """Realize next state by applying action. Returns node in game tree that represents realized next state."""
        realized_next_state = self._realize_outcome()
        next_state_hash = sj.hash_skyjo(realized_next_state)
        if update_child_weights:
            self.child_weights[next_state_hash] = (
                self.child_weights.get(next_state_hash, 0) + 1
            )
            self.child_weight_total += 1
        return self.children[next_state_hash]

    def update(
        self,
        child: DecisionStateNode,
        value: skynet.StateValue,
        prev_value: skynet.StateValue,
    ) -> None:
        """Used during backpropogation of a state value"""
        assert isinstance(child, DecisionStateNode), (
            f"Child must be a DecisionStateNode, got {type(child)} instead",
        )
        # If all children are discovered, we can update by just the change
        # in the state value of the previous (child) node since the weights
        # are constant
        if self.all_children_discovered:
            self.state_value_total += (
                (child.state_value - prev_value)
                * self.child_weights[sj.hash_skyjo(child.state)]
                / self.child_weight_total
            )

        # Otherwise we are just weighting by the occurences so we can just
        # explicitly update the state value total
        else:
            self.state_value_total += value


@dataclasses.dataclass(slots=True)
class TerminalStateNode:
    pre_terminal_state: sj.Skyjo
    parent: AfterStateNode | DecisionStateNode
    action: sj.SkyjoAction
    is_random: bool
    initial_rollouts: int = 1
    outcome_count: int = 0
    visit_count: int = 0
    virtual_loss: float = 0.0
    is_expanded: bool = False
    are_children_discovered: bool = False
    outcome_total: skynet.StateValue | None = None

    def __str__(self) -> str:
        return (
            f"TerminalStateNode\n"
            f"{sj.visualize_state(self.pre_terminal_state)}\n"
            f"action: {sj.get_action_name(self.action)}\n"
            f"state value: {self.state_value}\n"
            f"outcome count: {self.outcome_count}\n"
            f"is random: {self.is_random}\n"
        )

    def __post_init__(self):
        self.outcome_total = np.zeros(
            sj.get_player_count(self.pre_terminal_state), dtype=np.float32
        )
        self.realize_outcomes(self.initial_rollouts)

    @property
    def child_count(self) -> int:
        return self.outcome_count

    @property
    def state(self) -> sj.Skyjo:
        return self.pre_terminal_state

    @property
    def state_value(self) -> skynet.StateValue:
        return self.outcome_total / self.outcome_count

    def realize_outcome(self) -> None:
        terminal_state = sj.apply_action(self.pre_terminal_state, self.action)

        self.outcome_total += skynet.skyjo_to_state_value(terminal_state)
        self.outcome_count += 1

    def realize_outcomes(self, n: int) -> None:
        if self.is_random:
            for _ in range(n):
                self.realize_outcome()
        else:
            self.realize_outcome()

    def expand(self):
        raise ValueError("Terminal nodes should not need to be expanded")


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


def backpropagate(search_path: list[MCTSNode], value: skynet.StateValue):
    prev_node = None
    prev_node_prev_state_value = None
    for node in reversed(search_path):
        original_state_value = node.state_value
        if isinstance(node, DecisionStateNode):
            node.state_value_total += value
        elif isinstance(node, AfterStateNode) and prev_node is not None:
            node.update(prev_node, value, prev_node_prev_state_value)
        node.visit_count += 1
        prev_node = node
        prev_node_prev_state_value = original_state_value


def run_mcts(
    game_state: sj.Skyjo,
    predictor_client: predictor.PredictorClient,
    iterations: int,
    dirichlet_epsilon: float = 0.0,
    after_state_evaluate_all_children: bool = False,
    terminal_state_initial_rollouts: int = 1,
    forced_playout_k: float | None = None,
    root_node: MCTSNode | None = None,
) -> MCTSNode:
    """Runs a batched MCTS using virtual loss evaluation.

    This is not truly parallel an actually is just a single process and thread.
    However, it delays the model inference until there are max_parall_evaluation
    leaves pending.

    Description of algorithm:

    Starting at the root node, we inject a dirichlet noise with
    alpha = 10 / # of  valid actions.

    For every iteration we select a leaf node. A leaf node is just a node that
    we haven't discovered the children of. Generally, we call prexpand on the
    node to discover the children. This is a little different than the standard
    MCTS which considers a node expanded after a model prediction is ready and
    the children are discovered. This was done due to the batched nature of the
    evaluation. While the virtual loss does discourage exploring the same exact
    path to the leaf. This will guarantee that each iteration will reach a "new"
    leaf even if we don't have much information on the value of that path since
    the model prediction of the nodes along the path might not be ready yet.

    Once we have a leaf node, we can pre-expand the node and if necessary add it
    to the queue for model prediction. If we have after state realizations we
    will add all possible child states to the queue as well. Otherwise, we just
    realize a single child state and add it to the queue. Note, that each time
    that the after state is selected during the search, we will re-realize a
    child state.

    Once the pending leaf prediction queue is at the limit, we send them off
    for network inference throught the predictor client. Once the results are
    ready we process them.
    """
    # Get model prediction for root state
    if root_node is None:
        _ = predictor_client.put(game_state)
        predictor_client.send()
        _, prediction = predictor_client.get()
        root_node = DecisionStateNode(
            state=game_state,
            parent=None,
            action=None,
        )
        root_node.expand(
            model_prediction=prediction,
            terminal_state_rollouts=terminal_state_initial_rollouts,
        )

    # Inject dirichlet noise into the root node
    if dirichlet_epsilon > 0:
        valid_action_count = sj.actions(root_node.state).sum().item()
        dirichlet_noise = np.random.dirichlet(
            np.ones(valid_action_count) * 10 / valid_action_count,
        )
        root_node.dirichlet_noise[sj.get_actions(root_node.state)] = dirichlet_noise
        root_node.dirichlet_epsilon = dirichlet_epsilon

    if forced_playout_k is not None:
        root_node.forced_playout_k = forced_playout_k

    search_depths = []
    for _ in range(iterations):
        search_path = find_leaf(
            root_node,
            update_after_state_child_weights=not after_state_evaluate_all_children,
        )
        search_depths.append(len(search_path))
        leaf = search_path[-1]

        # if isinstance(leaf, TerminalStateNode):
        #     leaf.realize_outcomes(terminal_state_initial_rollouts)

        # AFTER STATE LEAF
        # We want to pre-expand the afterstate and either realize all potential
        # outcomes or just roll a single next state based on parameter
        #
        # We also need to queue all the children decision state for model prediction
        if isinstance(leaf, AfterStateNode):
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
                # Treate these as being expanded and visited
                child.expand(model_prediction=prediction)
                child.visit_count += 1
                child.state_value_total += skynet.to_state_value(
                    prediction.value_output, sj.get_player(child.state)
                )
                del afterstate_prediction_ids[prediction_id]

            leaf.expand()

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
            leaf.expand(
                model_prediction=prediction,
                terminal_state_rollouts=terminal_state_initial_rollouts,
            )
        backpropagate(search_path, leaf.state_value)
    # print(sum(search_depths) / len(search_depths))
    return root_node


# MARK: TYPES

MCTSNode: typing.TypeAlias = DecisionStateNode | AfterStateNode | TerminalStateNode


if __name__ == "__main__":
    import explain

    np.random.seed(42)
    torch.manual_seed(42)
    players = 2
    model = skynet.SimpleSkyNet(
        spatial_input_shape=(players, sj.ROW_COUNT, sj.COLUMN_COUNT, sj.FINGER_SIZE),
        non_spatial_input_shape=(sj.GAME_SIZE,),
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
        max_parallel_threads=100,
    )
    print(root_node)

# MARK: Debugging


def visualize_children(node: MCTSNode):
    assert not isinstance(node, TerminalStateNode), (
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
