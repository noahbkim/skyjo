"""Monte Carlo Tree Search implementation for Reinforcement Learning."""

from __future__ import annotations

import dataclasses
import typing

import numpy as np
import torch

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

    return skynet.state_value_for_player(
        child.state_value, sj.get_player(parent.state)
    ) + parent.model_prediction.policy_output[action].item() * np.sqrt(
        parent.visit_count
    ) / (1 + child.visit_count)


# MARK: Nodes


@dataclasses.dataclass(slots=True)
class DecisionStateNode:
    state: sj.Skyjo
    parent: AfterStateNode | None
    previous_action: sj.SkyjoAction | None
    state_value_total: skynet.StateValue | None = None
    model_prediction: skynet.SkyNetPrediction | None = None
    children: dict[sj.SkyjoAction, MCTSNode] = dataclasses.field(default_factory=dict)
    visit_count: int = 0
    is_expanded: bool = False

    def __post_init__(self):
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
            # If model prediction is available, use it to initialize state value
            if self.model_prediction is not None:
                return skynet.to_state_value(
                    self.model_prediction.value_output, sj.get_player(self.state)
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

    def expand(self, model: skynet.SkyNet):
        """Expand node by evaluating state with model"""
        self.model_prediction = model.predict(self.state)
        self.model_prediction.policy_output = (
            skynet.mask_and_renormalize_policy_probabilities(
                self.model_prediction.policy_output,
                sj.actions(self.state),
            )
        )
        for action in sj.get_actions(self.state):
            self.children[action] = self.create_child_node(action, model)
        self.is_expanded = True

    def select_child(
        self,
    ) -> MCTSNode:
        return self._select_highest_ucb_child()

    def create_child_node(
        self, action: sj.SkyjoAction, model: skynet.SkyNet
    ) -> MCTSNode:
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
            model_prediction=model.predict(next_state),
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
            return np.ones(sj.get_player_count(self.state)) / sj.get_player_count(
                self.state
            )
        return sum(
            [
                self.children[state_hash].state_value * count
                for state_hash, count in self.realized_counts.items()
            ]
        ) / sum(self.realized_counts.values())

    def _realize_outcome(self, model: skynet.SkyNet) -> sj.Skyjo:
        outcome_state = sj.apply_action(self.state, self.action)
        outcome_state_hash = sj.hash_skyjo(outcome_state)
        if outcome_state_hash not in self.realized_counts:
            self.children[outcome_state_hash] = DecisionStateNode(
                state=outcome_state,
                parent=self,
                previous_action=self.action,
                model_prediction=model.predict(outcome_state),
            )
        self.realized_counts[outcome_state_hash] = (
            self.realized_counts.get(outcome_state_hash, 0) + 1
        )
        return outcome_state

    def realize_outcomes(self, n: int, model: skynet.SkyNet):
        for _ in range(n):
            _ = self._realize_outcome(model)

    def expand(self, model: skynet.SkyNet, initial_realizations: int = 10):
        # Placeholder for now we may want to simulate many random outcomes here
        self.is_expanded = True
        self.realize_outcomes(n=initial_realizations, model=model)

    def select_child(
        self, model: skynet.SkyNet
    ) -> DecisionStateNode | TerminalStateNode:
        """Realize next state by applying action. Returns node in game tree that represents realized next state."""
        realized_next_state = self._realize_outcome(model)
        if sj.get_game_over(realized_next_state):
            return TerminalStateNode(
                state=realized_next_state,
                parent=self,
                previous_action=self.action,
            )
        return self.children[sj.hash_skyjo(realized_next_state)]

    def realized_outcome_probability(self, realized_outcome_state: sj.Skyjo) -> float:
        realized_outcome_state_hash = sj.hash_skyjo(realized_outcome_state)

        return self.realized_counts.get(realized_outcome_state_hash, 0) / sum(
            self.realized_counts.values()
        )


@dataclasses.dataclass(slots=True)
class TerminalStateNode:
    state: sj.Skyjo
    parent: AfterStateNode | DecisionStateNode
    previous_action: sj.SkyjoAction
    visit_count: int = 0
    is_expanded: bool = False

    def __str__(self) -> str:
        return (
            f"TerminalStateNode\n"
            f"{sj.visualize_state(self.state)}\n"
            f"State Value: {self.state_value}\n"
        )

    @property
    def state_value(self) -> skynet.StateValue:
        return skynet.skyjo_to_state_value(self.state)

    def expand(self):
        raise ValueError("Terminal nodes should not need to be expanded")


MCTSNode: typing.TypeAlias = DecisionStateNode | AfterStateNode | TerminalStateNode


# MARK: MCTS Algorithm
def run_mcts(
    game_state: sj.Skyjo,
    model: skynet.SkyNet,
    iterations: int,
    afterstate_initial_realizations: int = 10,
) -> MCTSNode:
    root_node = DecisionStateNode(
        state=game_state,
        parent=None,
        previous_action=None,
        model_prediction=model.predict(game_state),
    )

    for _ in range(iterations):
        search(
            root_node,
            model,
            afterstate_initial_realizations=afterstate_initial_realizations,
        )

    assert root_node.visit_count == iterations, (
        f"Root node visit count should be equal to number of iterations, "
        f"but visit count: {root_node.visit_count} and iterations: {iterations}"
    )

    return root_node


def search(
    node: MCTSNode,
    model: skynet.SkyNet,
    afterstate_initial_realizations: int = 10,
):
    search_path = [node]
    while node.is_expanded:
        if isinstance(node, AfterStateNode):
            node = node.select_child(model)
        else:
            node = node.select_child()
        search_path.append(node)
    leaf = node

    if isinstance(leaf, AfterStateNode):
        # realize an outcome and propogate value back up tree
        leaf.expand(model, initial_realizations=afterstate_initial_realizations)
        value = leaf.state_value
    elif isinstance(leaf, TerminalStateNode):
        value = leaf.state_value
        backpropagate(search_path, value)
        return
    else:
        leaf.expand(model)
        value = skynet.to_state_value(
            leaf.model_prediction.value_output, sj.get_player(leaf.state)
        )

    backpropagate(search_path, value)


def backpropagate(search_path: list[MCTSNode], value: skynet.StateValue):
    # Update each node's visited cound and average value
    # to be value of leaf node from that player's perspective
    for node in reversed(search_path):
        # No need to update state value for terminal nodes
        if not isinstance(node, TerminalStateNode):
            node.state_value_total += value
        node.visit_count += 1


if __name__ == "__main__":
    import explain

    np.random.seed(42)
    torch.manual_seed(42)
    players = 2
    game_state = sj.new(players=players)
    players = game_state[3]
    model = skynet.SimpleSkyNet(
        spatial_input_shape=(players, sj.ROW_COUNT, sj.COLUMN_COUNT, sj.FINGER_SIZE),
        non_spatial_input_shape=(sj.GAME_SIZE,),
        value_output_shape=(players,),
        policy_output_shape=(sj.MASK_SIZE,),
        hidden_layers=[64, 64],
        device=torch.device("cpu"),
    )
    # game_state = sj.start_round(game_state)
    game_state = explain.create_almost_surely_winning_position()
    training_data = []
    # Play round
    countdown = game_state[6]

    # simulate game
    while countdown != 0:
        root_node = run_mcts(
            game_state, model, iterations=1600, afterstate_initial_realizations=100
        )
        mcts_probs = root_node.sample_child_visit_probabilities(temperature=1.0)
        print(root_node)
        print(mcts_probs)
        training_data.append((game_state, mcts_probs))
        choice = np.random.choice(sj.MASK_SIZE, p=mcts_probs)
        assert sj.actions(game_state)[choice]
        game_state = sj.apply_action(game_state, choice)
        print(f"action: {sj.get_action_name(choice)}")
        print(f"{sj.visualize_state(game_state)}")
        countdown = game_state[6]
    winner = sj.get_fixed_perspective_winner(game_state)
    print(winner)
    print(sj.get_round_scores(game_state, 0))
