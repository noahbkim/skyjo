"""Tree-Parallel MCTS implementation for Skyjo.

This implementation 'batches' model evaluation and implements a virtual loss
to encourage exploration. Discussion of parallel algorithm can be found in
https://dke.maastrichtuniversity.nl/m.winands/documents/multithreadedMCTS2.pdf
"""

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
class BatchedMCTSConfig(config.Config):
    iterations: int
    dirichlet_epsilon: float
    after_state_evaluate_all_children: bool
    terminal_state_initial_rollouts: int
    batched_leaf_count: int
    virtual_loss: float
    forced_playout_k: float | None


# MARK: NODE SCORING


# def ucb_score(child: MCTSNode, parent: DecisionStateNode) -> float:
#     assert isinstance(parent, DecisionStateNode), (
#         f"Parent must be DecisionStateNode, got {type(parent)}"
#     )
#     total_visits = parent.visit_count
#     child_visits = child.visit_count
#     if isinstance(child, AfterStateNode):
#         action = child.action
#     elif isinstance(child, TerminalStateNode):
#         action = child.action
#         if not child.is_random:
#             return skynet.state_value_for_player(
#                 child.state_value, sj.get_player(parent.state)
#             )
#         child_visits = child.outcome_count
#         total_visits = sum([child.outcome_count for child in parent.children.values()])
#     else:
#         action = child.previous_action

#     if parent.model_prediction is not None and not isinstance(child, TerminalStateNode):
#         action_probability = (
#             parent.model_prediction.policy_output[action].item()
#             * (1 - parent.dirichlet_epsilon)
#             + parent.dirichlet_epsilon * parent.dirichlet_noise[action]
#         )
#     else:
#         action_probability = 1 / np.sum(sj.actions(parent.state)).item()

#     return (
#         skynet.state_value_for_player(child.state_value, sj.get_player(parent.state))
#         + action_probability * np.sqrt(total_visits) / (1 + child_visits)
#         - child.virtual_loss
#     )


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
    action: sj.SkyjoAction | None
    state_value_total: skynet.StateValue | None = None  # initialized in __post_init__
    model_prediction: skynet.SkyNetPrediction | None = None
    children: dict[sj.SkyjoAction, MCTSNode] = dataclasses.field(default_factory=dict)
    visit_count: int = 0
    is_expanded: bool = False
    are_children_discovered: bool = False
    virtual_loss_total: float = 0.0
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
    def virtual_loss(self) -> float:
        if self.visit_count == 0:
            return self.virtual_loss_total
        return self.virtual_loss_total / self.visit_count

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

    def expand(self, model_prediction: skynet.SkyNetPrediction) -> None:
        """Expand node by evaluating state with model"""
        assert self.model_prediction is None, "Model prediction already set"
        self.model_prediction = model_prediction
        self.is_expanded = True

    def preexpand(self, terminal_rollouts: int = 1):
        assert not self.are_children_discovered, (
            "Children must not already be discovered before preexpanding"
        )
        self.are_children_discovered = True
        for action in sj.get_actions(self.state):
            self.children[action] = self.create_child_node(action, terminal_rollouts)

    def highest_visit_child(self) -> MCTSNode:
        return max(self.children.values(), key=lambda x: x.visit_count)

    def select_child(
        self,
        **kwargs,
    ) -> MCTSNode:
        highest_ucb_child = self._select_highest_ucb_child()
        if (
            isinstance(highest_ucb_child, TerminalStateNode)
            and highest_ucb_child.is_random
        ):
            highest_ucb_child.realize_outcome()
        return highest_ucb_child

    def create_child_node(
        self, action: sj.SkyjoAction, terminal_rollouts: int = 1
    ) -> MCTSNode:
        if sj.get_game_about_to_end(self.state):
            return TerminalStateNode(
                pre_terminal_state=self.state,
                parent=self,
                action=action,
                is_random=sj.is_action_random(action, self.state),
                initial_outcome_realizations=terminal_rollouts,
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
                visit_counts[action] -= min(max(0, forced_visits), visit_counts[action])

        if temperature == 0:
            visit_probabilities = np.zeros(visit_counts.shape, dtype=np.float32)
            visit_probabilities[visit_counts.argmax().item()] = 1
            return visit_probabilities
        visit_probabilities = visit_counts ** (1 / temperature)
        visit_probabilities = visit_probabilities / visit_probabilities.sum()
        return visit_probabilities

    def action_probability(self, action) -> float:
        if self.model_prediction is None:
            return 1 / np.sum(sj.actions(self.state))
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
    are_children_discovered: bool = False
    all_children_discovered: bool = False
    virtual_loss_total: float = 0.0

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
        if self.visit_count == 0:
            return self.virtual_loss_total
        return self.virtual_loss_total / self.visit_count

    @property
    def state_value(self) -> skynet.StateValue:
        if not self.is_expanded:
            return np.ones(
                sj.get_player_count(self.state), dtype=np.float32
            ) / sj.get_player_count(self.state)
        # If all children discovered and exact probabilities were accounted for
        # we can just return the exact weighted total state value
        if self.all_children_discovered or self.visit_count == 0:
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

    def _preexpand_single_child(self) -> None:
        # Realize a single next child state. Child weights are now observered
        # frequencies of the child states.
        next_state = self._realize_outcome()
        next_state_hash = sj.hash_skyjo(next_state)
        self.child_weights[next_state_hash] = 1
        self.child_weight_total += 1

    def _preexpand_all_possible_children(self) -> None:
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

    def preexpand(self, discover_all_children: bool = False):
        assert not self.are_children_discovered, "Children already discovered"
        assert not sj.get_game_about_to_end(self.state), (
            "Game should not be about to end from an afterstate node"
        )
        self.are_children_discovered = True

        if discover_all_children:
            self._preexpand_all_possible_children()
        else:
            self._preexpand_single_child()

    def _compute_state_value_from_children(self) -> skynet.StateValue:
        state_value = np.zeros(sj.get_player_count(self.state), dtype=np.float32)
        for key_hash, child in self.children.items():
            state_value += (
                child.state_value
                * self.child_weights[key_hash]
                / self.child_weight_total
            )
        return state_value

    def expand(
        self,
    ):
        """Expands node after all initial children values are ready"""
        assert not self.is_expanded, "Node already expanded"
        self.is_expanded = True
        assert all(child.are_children_discovered for child in self.children.values()), (
            "All children must be discovered before expanding"
        )
        assert not self.all_children_discovered or np.allclose(
            self.state_value, self._compute_state_value_from_children()
        ), (
            "State value total should match weighted state value of child outcomes\n"
            f"state value: {self.state_value}\n"
            f"weighted state value of child outcomes: {self._compute_state_value_from_children()}\n"
        )

    def highest_visit_child(self) -> MCTSNode:
        return max(self.children.values(), key=lambda x: x.visit_count)

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
            # if int(sj.hash_skyjo(self.state)) == -4569980319794928185:
            #     print(
            #         "update ", child.state_value, prev_value, child.visit_count, value
            #     )
            self.state_value_total += (
                child.state_value - prev_value
            ) * self.child_weights[sj.hash_skyjo(child.state)]
            if (
                self.state_value_total.min() < 0
                and int(sj.hash_skyjo(self.state)) == -4569980319794928185
            ):
                print(
                    self.state_value_total,
                    child.state_value,
                    prev_value,
                    self.child_weights[sj.hash_skyjo(child.state)],
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
    initial_outcome_realizations: int
    outcome_count: int = 0
    visit_count: int = 0
    virtual_loss: float = 0.0
    is_expanded: bool = False
    are_children_discovered: bool = False
    outcome_total: skynet.StateValue | None = None

    def __post_init__(self):
        self.outcome_total = np.zeros(
            sj.get_player_count(self.pre_terminal_state), dtype=np.float32
        )
        self.realize_outcomes(self.initial_outcome_realizations)

    def __str__(self) -> str:
        return (
            f"TerminalStateNode\n"
            f"{sj.visualize_state(self.pre_terminal_state)}\n"
            f"action: {sj.get_action_name(self.action)}\n"
            f"state value: {self.state_value}\n"
            f"outcome count: {self.outcome_count}\n"
            f"is random: {self.is_random}\n"
        )

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
    virtual_loss: float = 0.5,
    update_after_state_child_weights: bool = False,
):
    search_path = [root]
    node = root
    while node.are_children_discovered:
        node = node.select_child(update_child_weights=update_after_state_child_weights)
        if not isinstance(node, TerminalStateNode):
            node.virtual_loss_total += virtual_loss
        search_path.append(node)
    return search_path


def backpropagate(search_path: list[MCTSNode], value: skynet.StateValue, virtual_loss):
    prev_node = search_path[-1]
    prev_node_prev_state_value = search_path[-1].state_value
    for i, node in enumerate(reversed(search_path)):
        original_state_value = node.state_value
        if isinstance(node, DecisionStateNode):
            node.state_value_total += value
            node.virtual_loss_total -= virtual_loss
        elif isinstance(node, AfterStateNode):
            node.virtual_loss_total -= virtual_loss
            if i > 0:
                # if int(sj.hash_skyjo(node.state)) == -4569980319794928185:
                #     print("backprop")
                #     print(search_path[-1])
                node.update(prev_node, value, prev_node_prev_state_value)
            elif node.all_children_discovered:
                for child in node.children.values():
                    child.state_value_total += skynet.to_state_value(
                        child.model_prediction.value_output,
                        sj.get_player(child.state),
                    )
                    child.visit_count += 1
                    # if int(sj.hash_skyjo(node.state)) == -4569980319794928185:
                    #     print(child.state_value)
        node.visit_count += 1
        prev_node = node
        prev_node_prev_state_value = original_state_value


def _handle_prediction_results(
    prediction_id: predictor.PredictionId,
    prediction: skynet.SkyNetPrediction,
    pending_decision_state_search_paths: dict[predictor.PredictionId, list[MCTSNode]],
    pending_after_state_search_paths: dict[predictor.PredictionId, list[MCTSNode]],
    pending_after_state_prediction_ids: dict[
        predictor.PredictionId, set[predictor.PredictionId]
    ],
    prediction_id_to_after_state_prediction_id: dict[
        predictor.PredictionId, predictor.PredictionId
    ],
    virtual_loss: float,
):
    """Decision States:
    We add the model prediction to the node and update the state value total
    with the value of that model prediction.

    After state children:
    We add the model prediction to the corresponding decision state child. If
    all the children of the after state have had their prediction processed, we
    can compute the state value of the after state as a functino of all those
    child state values. We then backpropagate that value up the search path.

    Terminal states:
    """
    # Decision state prediction
    if prediction_id in pending_decision_state_search_paths:
        search_path = pending_decision_state_search_paths[prediction_id]
        leaf = search_path[-1]
        leaf.expand(prediction)
        backpropagate(
            search_path,
            skynet.to_state_value(
                leaf.model_prediction.value_output,
                sj.get_player(leaf.state),
            ),
            virtual_loss,
        )
        del pending_decision_state_search_paths[prediction_id]
        return True
    # After state child decision state predictions
    elif prediction_id in pending_after_state_search_paths:
        search_path = pending_after_state_search_paths[prediction_id]
        after_state_leaf, decision_state_child = (
            search_path[-2],
            search_path[-1],
        )
        model_state_value = skynet.to_state_value(
            prediction.value_output, sj.get_player(decision_state_child.state)
        )
        decision_state_child.expand(prediction)
        # if int(sj.hash_skyjo(after_state_leaf.state)) == -4569980319794928185:
        #     print("handle")
        after_state_leaf.update(decision_state_child, model_state_value, 0)
        after_state_prediction_id = prediction_id_to_after_state_prediction_id[
            prediction_id
        ]
        pending_after_state_prediction_ids[after_state_prediction_id].remove(
            prediction_id
        )
        del prediction_id_to_after_state_prediction_id[prediction_id]
        del pending_after_state_search_paths[prediction_id]
        # If all after state realized children have been processed, backpropagate
        if len(pending_after_state_prediction_ids[after_state_prediction_id]) == 0:
            after_state_leaf.expand()
            backpropagate(
                search_path[:-1],  # don't update decision node child
                after_state_leaf.state_value,
                virtual_loss,
            )
            del pending_after_state_prediction_ids[after_state_prediction_id]
            return True
        return False
    else:
        raise ValueError(
            f"prediction id {prediction_id} not found in pending predictions"
        )


def run_mcts(
    game_state: sj.Skyjo,
    predictor_client: predictor.PredictorClient,
    iterations: int,
    dirichlet_epsilon: float = 0.0,
    after_state_evaluate_all_children: bool = False,
    terminal_state_initial_rollouts: int = 1,
    batched_leaf_count: int = 1,
    virtual_loss: float = 0.5,
    forced_playout_k: float | None = None,
    root_node: MCTSNode | None = None,
) -> MCTSNode:
    """Runs a batched MCTS using virtual loss evaluation.

    This is not truly parallel an actually is just a single process and thread.
    However, it delays the model inference until there are batched_leaf_count
    leaves pending model inference.

    Description of algorithm:

    Starting at the root node, we inject a dirichlet noise with
    alpha = 10 / # of  valid actions.

    We also allow optional forced playout with constant k from KataGo:
    n_forced = sqrt(k * child_action_probability * parent_visit_count)

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
        root_node.preexpand(terminal_rollouts=terminal_state_initial_rollouts)
        root_node.expand(model_prediction=prediction)

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
    # Holds map of prediction id to values needed to update and backpropagate
    # once the prediction is ready
    pending_decision_state_search_paths = {}
    pending_after_state_search_paths = {}
    # Maps an after state prediction id to all the children prediction ids.
    pending_after_state_prediction_ids = {}
    prediction_id_to_after_state_prediction_id = {}
    after_state_prediction_count = 0
    pending_leaf_count = 0
    search_depths = []
    for _ in range(iterations):
        search_path = find_leaf(
            root_node,
            virtual_loss=virtual_loss,
            update_after_state_child_weights=not after_state_evaluate_all_children,
        )
        search_depths.append(len(search_path))
        leaf = search_path[-1]

        # TERMINAL STATE LEAF
        # We can backpropagate immediately since we don't need to wait on a model
        # prediction to get the value of the state
        if isinstance(leaf, TerminalStateNode):
            # Process all pending predictions first
            while pending_leaf_count > 0:
                predictor_client.send()
                for prediction_id, prediction in predictor_client.get_all():
                    if _handle_prediction_results(
                        prediction_id,
                        prediction,
                        pending_decision_state_search_paths,
                        pending_after_state_search_paths,
                        pending_after_state_prediction_ids,
                        prediction_id_to_after_state_prediction_id,
                        virtual_loss,
                    ):
                        pending_leaf_count -= 1
            backpropagate(search_path, leaf.state_value, virtual_loss)
            continue

        # AFTER STATE LEAF
        # We want to pre-expand the afterstate and either realize all potential
        # outcomes or just roll a single next state based on parameter
        #
        # We also need to queue all the children decision state for model prediction
        elif isinstance(leaf, AfterStateNode):
            leaf.preexpand(
                discover_all_children=after_state_evaluate_all_children,
            )
            after_state_prediction_id = after_state_prediction_count
            after_state_prediction_count += 1
            pending_after_state_prediction_ids[after_state_prediction_id] = set()
            pending_leaf_count += 1

            # Add realized outcome children to prediction
            for child in leaf.children.values():
                if isinstance(child, DecisionStateNode):
                    child.preexpand(terminal_rollouts=terminal_state_initial_rollouts)
                    prediction_id = predictor_client.put(child.state)
                    pending_after_state_search_paths[prediction_id] = search_path + [
                        child
                    ]
                    pending_after_state_prediction_ids[after_state_prediction_id].add(
                        prediction_id
                    )
                    prediction_id_to_after_state_prediction_id[prediction_id] = (
                        after_state_prediction_id
                    )
                elif isinstance(child, TerminalStateNode):
                    pass
                else:
                    raise ValueError(
                        f"Decision or Terminal node expected to follow After, got {type(child)}"
                    )
            # Edge case where all children of after state are terminal nodes
            # so no need to wait on any predictions
            if len(pending_after_state_prediction_ids[after_state_prediction_id]) == 0:
                del pending_after_state_prediction_ids[after_state_prediction_id]
                leaf.expand()
                backpropagate(
                    search_path,
                    leaf.state_value,
                    virtual_loss,
                )
                pending_leaf_count -= 1

        # DECISION STATE LEAF
        # We want to queue the decision state for model prediction. Also
        # we want to pre-expand the decision state, so that parallel threads
        # can go deeper and queue a child state for model prediction.
        elif isinstance(leaf, DecisionStateNode):
            leaf.preexpand(terminal_rollouts=terminal_state_initial_rollouts)
            prediction_id = predictor_client.put(leaf.state)
            pending_decision_state_search_paths[prediction_id] = search_path
            pending_leaf_count += 1

        else:
            raise ValueError(f"Unknown node type: {type(leaf)}")

        # Process predictions until we have at most max_parallel_threads pending leaf nodes
        while pending_leaf_count >= batched_leaf_count:
            predictor_client.send()
            for prediction_id, prediction in predictor_client.get_all():
                if _handle_prediction_results(
                    prediction_id,
                    prediction,
                    pending_decision_state_search_paths,
                    pending_after_state_search_paths,
                    pending_after_state_prediction_ids,
                    prediction_id_to_after_state_prediction_id,
                    virtual_loss,
                ):
                    pending_leaf_count -= 1
    while pending_leaf_count > 0:
        if len(predictor_client.current_inputs) > 0:
            predictor_client.send()
        for prediction_id, prediction in predictor_client.get_all():
            if _handle_prediction_results(
                prediction_id,
                prediction,
                pending_decision_state_search_paths,
                pending_after_state_search_paths,
                pending_after_state_prediction_ids,
                prediction_id_to_after_state_prediction_id,
                virtual_loss,
            ):
                pending_leaf_count -= 1
    # print(f"Mean search depth: {sum(search_depths) / len(search_depths)}")
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
