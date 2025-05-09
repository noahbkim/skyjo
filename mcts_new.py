from __future__ import annotations

import dataclasses
import enum

import einops
import numpy as np
import torch

import skyjo as sj
import skynet


# MARK: Enums
class MCTSSelectionMethod(enum.Enum):
    UCB = "UCB"  # AlphaZero UCB
    # Lots of different research ideas to explore here
    # with other selection methods


# MARK: Dataclasses
@dataclasses.dataclass(slots=True)
class DecisionStateNode:
    state: sj.Skyjo
    parent: AfterStateNode | None
    prev_action: sj.SkyjoAction | None
    model_output: skynet.SkyNetOutput | None = None
    children: dict[sj.SkyjoAction, MCTSNode] = dataclasses.field(default_factory=dict)
    num_visits: int = 0
    average_visit_value: float = 1  # initialize to 1 to force initial exploration
    is_expanded: bool = False

    def _select_highest_ucb_child(self) -> MCTSNode:
        return max(
            self.children.values(),
            key=lambda child: ucb_score(child, self),
        )

    def expand(self, model: skynet.SkyNet):
        """Expand node by evaluating state with model"""
        self.model_output = model.predict(self.state)
        self.model_output.policy_output = (
            self.model_output.policy_output.mask_invalid_and_renormalize(
                einops.rearrange(sj.actions(self.state), "a -> 1 a")
            )
        )
        for action in sj.actions_list(self.state):
            self.children[action] = self.create_child_node(action, model)
        self.is_expanded = True

    def select_child(
        self,
        model: skynet.SkyNet,
        method: MCTSSelectionMethod = MCTSSelectionMethod.UCB,
    ) -> MCTSNode:
        match method:
            case MCTSSelectionMethod.UCB:
                return self._select_highest_ucb_child()
            case _:
                raise ValueError(f"Unknown method: {method}")

    def create_child_node(
        self, action: sj.SkyjoAction, model: skynet.SkyNet
    ) -> MCTSNode:
        if sj.is_action_random(action, self.state):
            return AfterStateNode(state=self.state, action=action, parent=self)
        next_state = sj.apply_action(self.state, action)
        if sj.get_game_over(next_state):
            return TerminalStateNode(state=next_state, parent=self, prev_action=action)
        return DecisionStateNode(
            state=next_state,
            parent=self,
            prev_action=action,
            model_output=model.predict(next_state),
        )

    def sample_child_visit_probabilities(
        self, temperature: float = 1.0
    ) -> np.ndarray[tuple[int], np.float32]:
        """Sample visit probabilities for children nodes."""
        visit_counts = np.zeros((sj.MASK_SIZE,))
        for action, child in self.children.items():
            visit_counts[action] = child.num_visits
        visit_probabilities = visit_counts ** (1 / temperature)
        visit_probabilities = visit_probabilities / visit_probabilities.sum()
        return visit_probabilities


@dataclasses.dataclass(slots=True)
class AfterStateNode:
    state: sj.Skyjo
    action: sj.SkyjoAction
    parent: DecisionStateNode
    children: dict[sj.Skyjo, DecisionStateNode | TerminalStateNode] = dataclasses.field(
        default_factory=dict
    )
    realized_counts: dict[sj.Skyjo, int] = dataclasses.field(default_factory=dict)
    num_visits: int = 0
    average_visit_value: float = 1
    is_expanded: bool = False

    def _realize_outcome(self) -> sj.Skyjo:
        outcome_state = sj.apply_action(self.state, self.action)
        outcome_state_hash = sj.hash_skyjo(outcome_state)
        self.realized_counts[outcome_state_hash] = (
            self.realized_counts.get(outcome_state_hash, 0) + 1
        )
        return outcome_state

    def expand(self, model: skynet.SkyNet):
        # Placeholder for now we may want to simulate many random outcomes here
        self.is_expanded = True

    def select_child(
        self, model: skynet.SkyNet
    ) -> DecisionStateNode | TerminalStateNode:
        """Realize next state by applying action. Returns node in game tree that represents realized next state."""
        realized_next_state = self._realize_outcome()
        if sj.get_game_over(realized_next_state):
            return TerminalStateNode(
                state=realized_next_state,
                parent=self,
                prev_action=self.action,
            )
        realized_next_state_hash = sj.hash_skyjo(realized_next_state)
        if realized_next_state_hash not in self.children:
            self.children[realized_next_state_hash] = DecisionStateNode(
                state=realized_next_state,
                parent=self,
                prev_action=self.action,
                model_output=model.predict(realized_next_state),
            )

        return self.children[realized_next_state_hash]

    def realized_outcome_probability(self, realized_outcome_state: sj.Skyjo) -> float:
        realized_outcome_state_hash = sj.hash_skyjo(realized_outcome_state)

        return self.realized_counts.get(realized_outcome_state_hash, 0) / sum(
            self.realized_counts.values()
        )


@dataclasses.dataclass(slots=True)
class TerminalStateNode:
    state: sj.Skyjo
    parent: AfterStateNode | DecisionStateNode
    prev_action: sj.SkyjoAction
    num_visits: int = 0
    average_visit_value: float = 1
    is_expanded: bool = False

    @property
    def outcome(self) -> skynet.StateValue:
        return skynet.skyjo_to_state_value(self.state)

    def expand(self, model: skynet.SkyNet):
        raise ValueError("Terminal nodes should not need to be expanded")


MCTSNode = DecisionStateNode | AfterStateNode | TerminalStateNode


# MARK: Node Scoring
def ucb_score(child: MCTSNode, parent: MCTSNode) -> float:
    if isinstance(child, TerminalStateNode):
        return skynet.state_value_for_player(child.outcome, sj.get_player(parent.state))
    elif isinstance(parent, DecisionStateNode):
        if isinstance(child, AfterStateNode):
            action = child.action
        else:
            action = child.prev_action
        return (
            child.average_visit_value
            + parent.model_output.policy_output.get_action_probability(action)
            * np.sqrt(parent.num_visits)
            / (1 + child.num_visits)
        )

    # Child node must be DecisionStateNode since
    # After -> Decision | Terminal and Terminal returns earlier
    elif isinstance(parent, AfterStateNode):
        # For unvisited nodes, return 1 to force exploration
        if child.num_visits == 0:
            return 1
        # Probability weighted average of value of realized decision nodes
        # as evaluated by model from current player's perspective
        return sum(
            [
                # empirical probability of realized state
                child.realized_outcome_probability(realized_decision_node.state)
                * ucb_score(realized_decision_node, child)
                for realized_decision_node in child.children.values()
            ]
        )
    else:
        raise ValueError(f"Unknown node type: {type(child)}")


# MARK: MCTS Algorithm
def run_mcts(
    game_state: sj.Skyjo,
    model: skynet.SkyNet,
    iterations: int,
) -> MCTSNode:
    root_node = DecisionStateNode(
        state=game_state,
        parent=None,
        prev_action=None,
        model_output=model.predict(game_state),
    )

    for _ in range(iterations):
        search(root_node, model)

    return root_node


def search(node: MCTSNode, model: skynet.SkyNet):
    search_path = [node]
    while node.is_expanded:
        node = node.select_child(model)
        search_path.append(node)
    leaf = node

    if isinstance(leaf, AfterStateNode):
        # realize an outcome and propogate value back up tree
        leaf.expand(model)
        leaf = leaf.select_child(model)
    if isinstance(leaf, TerminalStateNode):
        value = leaf.outcome
        backpropagate(search_path, value)
        return
    leaf.expand(model)
    value = leaf.model_output.value_output.state_value()

    backpropagate(search_path, value)


def backpropagate(search_path: list[MCTSNode], value: skynet.StateValue):
    # Update each node's visited cound and average value
    # to be value of leaf node from that player's perspective
    for node in search_path:
        node.average_visit_value = (
            node.average_visit_value * (node.num_visits)
            + skynet.state_value_for_player(value, sj.get_player(node.state))
        ) / (node.num_visits + 1)
        node.num_visits += 1


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    players = 2
    game_state = sj.new(players=players)
    players = game_state[3]
    model = skynet.SkyNet1D(
        spatial_input_shape=(players, sj.ROW_COUNT, sj.COLUMN_COUNT, sj.FINGER_SIZE),
        non_spatial_input_shape=(sj.GAME_SIZE,),
        value_output_shape=(players,),
        policy_output_shape=(sj.MASK_SIZE,),
    )
    game_state = sj.start_round(game_state)
    training_data = []
    # Play round
    countdown = game_state[6]

    # simulate game
    while countdown != 0:
        root_node = run_mcts(game_state, model, iterations=100)
        mcts_probs = root_node.sample_child_visit_probabilities(temperature=1.0)
        training_data.append((game_state, mcts_probs))
        choice = np.random.choice(sj.MASK_SIZE, p=mcts_probs)
        assert sj.actions(game_state)[choice]

        game_state = sj.apply_action(game_state, choice)
        countdown = game_state[6]
    winner = sj.get_fixed_perspective_winner(game_state)
    print(winner)
    print(sj.get_round_scores(game_state, 0))
