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

import predictor
import skyjo as sj
import skynet

# MARK: NODE SCORING


def ucb_score(child: MCTSNode, parent: DecisionStateNode) -> float:
    assert isinstance(parent, DecisionStateNode), (
        f"Parent must be DecisionStateNode, got {type(parent)}"
    )
    if isinstance(child, TerminalStateNode):
        return skynet.state_value_for_player(child.outcome, sj.get_player(parent.state))
    if isinstance(child, AfterStateNode):
        action = child.action
    else:
        action = child.previous_action

    return (
        skynet.state_value_for_player(child.state_value, sj.get_player(parent.state))
        + parent.model_prediction.policy_output[action].item()
        * np.sqrt(parent.visit_count)
        / (1 + child.visit_count)
        - child.virtual_loss
    )


# MARK: NODES


@dataclasses.dataclass(slots=True)
class DecisionStateNode:
    state: sj.Skyjo
    parent: DecisionStateNode | AfterStateNode | None
    previous_action: sj.SkyjoAction | None
    state_value_total: skynet.StateValue | None = None  # initialized in __post_init__
    model_prediction: skynet.SkyNetPrediction | None = None
    children: dict[sj.SkyjoAction, MCTSNode] = dataclasses.field(default_factory=dict)
    visit_count: int = 0
    is_expanded: bool = False
    virtual_loss: float = 0.0

    def __post_init__(self):
        # need to initialize here because we don't know the player count until after we have the state
        self.state_value_total = np.zeros(sj.get_player_count(self.state))

    def __str__(self) -> str:
        return (
            f"DecisionStateNode\n"
            f"{sj.visualize_state(self.state)}\n"
            f"Visit Count: {self.visit_count}\n"
            f"State Value: {self.state_value}\n"
            f"Is Expanded: {self.is_expanded}\n"
            f"Children: {len(self.children)}\n"
        )

    @property
    def state_value(self) -> skynet.StateValue:
        if self.visit_count == 0:
            if self.model_prediction is not None:
                return skynet.to_state_value(
                    self.model_prediction.value_output,
                    sj.get_player(self.state),
                )

            return np.ones(sj.get_player_count(self.state)) / sj.get_player_count(
                self.state
            )
        return self.state_value_total / self.visit_count

    def _select_highest_ucb_child(self) -> MCTSNode:
        return max(
            self.children.values(),
            key=lambda child: ucb_score(child, self),
        )

    def expand(self, model_prediction: skynet.SkyNetPrediction | None = None) -> None:
        """Expand node by evaluating state with model"""
        assert self.model_prediction is not None or model_prediction is not None, (
            "Model prediction must be provided if model_prediction is not already set"
        )
        if self.model_prediction is None:
            self.model_prediction = model_prediction
        self.model_prediction.mask_and_renormalize(sj.actions(self.state))
        for action in sj.get_actions(self.state):
            self.children[action] = self.create_child_node(action)
        self.is_expanded = True

    def select_child(
        self,
    ) -> MCTSNode:
        highest_ucb_child = self._select_highest_ucb_child()
        return highest_ucb_child

    def create_child_node(self, action: sj.SkyjoAction) -> MCTSNode:
        if sj.is_action_random(action, self.state):
            return AfterStateNode(state=self.state, action=action, parent=self)
        next_state = sj.apply_action(self.state, action)
        if sj.get_game_over(next_state):
            return TerminalStateNode(
                state=next_state, parent=self, previous_action=action
            )
        return DecisionStateNode(
            state=next_state,
            parent=self,
            previous_action=action,
        )

    def sample_child_visit_probabilities(
        self, temperature: float = 1.0
    ) -> np.ndarray[tuple[int], np.float32]:
        """Sample visit probabilities for children nodes."""
        visit_counts = np.zeros((sj.MASK_SIZE,))
        for action, child in self.children.items():
            visit_counts[action] = child.visit_count
        if temperature == 0:
            visit_probabilities = np.zeros(visit_counts.shape)
            visit_probabilities[visit_counts.argmax().item()] = 1
            return visit_probabilities
        visit_probabilities = visit_counts ** (1 / temperature)
        visit_probabilities = visit_probabilities / visit_probabilities.sum()
        return visit_probabilities


@dataclasses.dataclass(slots=True)
class AfterStateNode:
    state: sj.Skyjo
    action: sj.SkyjoAction
    parent: DecisionStateNode
    state_value_total: skynet.StateValue | None = None
    children: dict[sj.Skyjo, DecisionStateNode | TerminalStateNode] = dataclasses.field(
        default_factory=dict
    )
    realized_counts: dict[sj.Skyjo, int] = dataclasses.field(default_factory=dict)
    visit_count: int = 0
    is_expanded: bool = False
    virtual_loss: float = 0.0

    def __post_init__(self):
        self.state_value_total = np.zeros(sj.get_player_count(self.state))

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
    def state_value(self) -> skynet.StateValue:
        if len(self.children) == 0:
            # TODO: change to use model predictions
            return np.ones(sj.get_player_count(self.state)) / sj.get_player_count(
                self.state
            )
        return sum(
            [
                self.children[state_hash].state_value * count
                for state_hash, count in self.realized_counts.items()
            ]
        ) / sum(self.realized_counts.values())

    def _create_child(self, state: sj.Skyjo) -> MCTSNode:
        if sj.get_game_over(state):
            return TerminalStateNode(
                state=state, parent=self, previous_action=self.action
            )
        return DecisionStateNode(
            state=state,
            parent=self,
            previous_action=self.action,
        )

    def realize_outcomes(self, n) -> None:
        for _ in range(n):
            _ = self._realize_outcome()

    def _realize_outcome(self) -> sj.Skyjo:
        outcome_state = sj.apply_action(self.state, self.action)
        outcome_state_hash = sj.hash_skyjo(outcome_state)
        if outcome_state_hash not in self.realized_counts:
            self.children[outcome_state_hash] = self._create_child(outcome_state)
        self.realized_counts[outcome_state_hash] = (
            self.realized_counts.get(outcome_state_hash, 0) + 1
        )
        return outcome_state

    def expand(self, initial_realizations: int = 10):
        # Placeholder for now we may want to simulate many random outcomes here
        self.is_expanded = True
        self.realize_outcomes(n=initial_realizations)

    def select_child(
        self,
    ) -> DecisionStateNode | TerminalStateNode:
        """Realize next state by applying action. Returns node in game tree that represents realized next state."""
        realized_next_state = self._realize_outcome()
        return self.children[sj.hash_skyjo(realized_next_state)]


@dataclasses.dataclass(slots=True)
class TerminalStateNode:
    state: sj.Skyjo
    parent: AfterStateNode | DecisionStateNode
    previous_action: sj.SkyjoAction
    visit_count: int = 0
    is_expanded: bool = False
    outcome: skynet.StateValue | None = None

    def __post_init__(self):
        self.outcome = skynet.skyjo_to_state_value(self.state)

    def __str__(self) -> str:
        return (
            f"TerminalStateNode\n"
            f"{sj.visualize_state(self.state)}\n"
            f"Outcome: {self.outcome}\n"
        )

    @property
    def state_value(self) -> skynet.StateValue:
        return self.outcome

    def expand(self):
        raise ValueError("Terminal nodes should not need to be expanded")


# MARK: MCTS Algorithm


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
    # Decision state prediction
    if prediction_id in pending_decision_state_search_paths:
        search_path = pending_decision_state_search_paths[prediction_id]
        leaf = search_path[-1]
        leaf.expand(model_prediction=prediction)
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
        decision_state_child.model_prediction = prediction
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
    afterstate_initial_realizations: int = 10,
    virtual_loss: float = 0.5,
    max_parallel_evaluations: int = 100,
) -> MCTSNode:
    # TODO: Modularize this function with helper functions
    _ = predictor_client.put(game_state)
    predictor_client.send()
    _, prediction = predictor_client.get()
    root_node = DecisionStateNode(
        state=game_state,
        parent=None,
        previous_action=None,
    )
    root_node.expand(model_prediction=prediction)
    # Holds map of prediction id to values needed to update and backpropagate
    # once the prediction is ready
    pending_decision_state_search_paths = {}
    pending_after_state_search_paths = {}
    # Maps an after state prediction id to all the children prediction ids.
    pending_after_state_prediction_ids = {}
    prediction_id_to_after_state_prediction_id = {}
    after_state_prediction_count = 0
    pending_leaf_count = 0
    for _ in range(iterations):
        search_path = find_leaf(
            root_node,
            virtual_loss=virtual_loss,
        )
        leaf = search_path[-1]
        if isinstance(leaf, TerminalStateNode):
            # TODO: think about whether leaf needs to be expanded...
            backpropagate(search_path, leaf.outcome, virtual_loss)

        elif isinstance(leaf, AfterStateNode):
            leaf.expand(initial_realizations=afterstate_initial_realizations)
            after_state_prediction_id = after_state_prediction_count
            after_state_prediction_count += 1
            pending_after_state_prediction_ids[after_state_prediction_id] = set()
            pending_leaf_count += 1
            # Add realized outcome children to prediction
            for child in leaf.children.values():
                if isinstance(child, DecisionStateNode):
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
                backpropagate(
                    search_path,
                    leaf.state_value,
                    virtual_loss,
                )
                pending_leaf_count -= 1
        # TODO: change this to evaluate all child node decision states too.
        elif isinstance(leaf, DecisionStateNode):
            if leaf.model_prediction is None:
                prediction_id = predictor_client.put(leaf.state)
                pending_decision_state_search_paths[prediction_id] = search_path
                pending_leaf_count += 1

            # If model prediction is already set, expand and backpropagate
            else:
                leaf.expand()
                backpropagate(
                    search_path,
                    leaf.state_value,
                    virtual_loss,
                )
        else:
            raise ValueError(f"Unknown node type: {type(leaf)}")

        # Process predictions until we have at most max_parallel_threads pending leaf nodes
        while pending_leaf_count >= max_parallel_evaluations:
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
    return root_node


def find_leaf(
    root: MCTSNode,
    virtual_loss: float = 0.5,
):
    search_path = [root]
    node = root
    while node.is_expanded:
        node = node.select_child()
        search_path.append(node)
        if not isinstance(node, TerminalStateNode):
            node.virtual_loss += virtual_loss
    return search_path


def backpropagate(search_path: list[MCTSNode], value: skynet.StateValue, virtual_loss):
    # Update each node's visited count, state_value_total
    for node in reversed(search_path):
        if not isinstance(node, TerminalStateNode):
            node.state_value_total += value
            node.virtual_loss -= virtual_loss
        node.visit_count += 1


# MARK: TYPES

MCTSNode: typing.TypeAlias = DecisionStateNode | AfterStateNode | TerminalStateNode


if __name__ == "__main__":
    import explain

    np.random.seed(42)
    torch.manual_seed(42)
    players = 2
    model = skynet.SkyNet1D(
        spatial_input_shape=(players, sj.ROW_COUNT, sj.COLUMN_COUNT, sj.FINGER_SIZE),
        non_spatial_input_shape=(sj.GAME_SIZE,),
        value_output_shape=(players,),
        policy_output_shape=(sj.MASK_SIZE,),
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
