from __future__ import annotations

import dataclasses
import enum

import numpy as np

import skyjo as sj
import skynet


# MARK: Enums
class MCTSSelectionMethod(enum.Enum):
    UCB = "UCB"  # AlphaZero UCB
    # Lots of different research ideas to explore here with other
    # selection methods


# MARK: Dataclasses
@dataclasses.dataclass(slots=True)
class DecisionStateNode:
    state: sj.Skyjo
    parent: AfterStateNode | None
    children: dict[sj.SkyjoAction, MCTSNode] = dataclasses.field(default_factory=dict)
    prev_action: sj.SkyjoAction | None
    model_output: skynet.SkynetOutput
    num_visits: int = 0
    average_visit_value: float = 1  # initialize to 1 to force initial exploration

    @property
    def is_expanded(self) -> bool:
        return len(self.children) > 0

    def _select_highest_ucb_child(self) -> MCTSNode:
        return max(
            self.children.values(),
            key=lambda child: ucb_score(child, self),
        )

    def expand(self, model: skynet.Skynet):
        """Expand node by evaluating state with model"""
        self.model_output = model.predict(self.state)
        for action in sj.get_valid_actions(self.state):
            self.children[action] = self.create_child_node(action)

    def select_child(
        self, method: MCTSSelectionMethod = MCTSSelectionMethod.UCB
    ) -> MCTSNode:
        match method:
            case MCTSSelectionMethod.UCB:
                return self._select_highest_ucb_child()
            case _:
                raise ValueError(f"Unknown method: {method}")

    def create_child_node(self, action: sj.SkyjoAction) -> MCTSNode:
        if sj.is_action_random(action, self.state):
            return AfterStateNode(state=self.state, action=action, parent=self)
        next_state = sj.apply_action(self.state, action)
        if sj.is_game_over(next_state):
            return TerminalStateNode(state=next_state, parent=self)
        return DecisionStateNode(state=next_state, action=action, parent=self)


@dataclasses.dataclass(slots=True)
class AfterStateNode:
    state: sj.Skyjo
    action: sj.SkyjoAction
    parent: DecisionStateNode
    children: dict[sj.Skyjo, DecisionStateNode | TerminalStateNode] = dataclasses.field(
        default_factory=dict
    )
    num_visits: int = 0
    average_visit_value: float = 1
    is_expanded: bool = False

    def expand(self, model: skynet.Skynet, rng: np.random.RandomState):
        # Placeholder for now we may want to simulate many random outcomes here
        self.is_expanded = True

    def select_child(self) -> DecisionStateNode | TerminalStateNode:
        """Realize next state by applying action. Returns node in game tree that represents realized next state."""
        realized_next_state = sj.apply_action(self.state, self.action)
        if realized_next_state.game_has_ended:
            return TerminalStateNode(
                state=realized_next_state,
                parent=self,
            )

        if realized_next_state not in self.children:
            self.children[realized_next_state] = DecisionStateNode(
                state=realized_next_state,
                parent=self,
                prev_action=self.action,
            )

        return self.children[realized_next_state]


@dataclasses.dataclass(slots=True)
class TerminalStateNode:
    state: sj.Skyjo
    parent: AfterStateNode | DecisionStateNode
    num_visits: int = 0
    average_visit_value: float = 1

    @property
    def outcome(self) -> sj.SkyjoOutcome:
        return sj.SkyjoOutcome.from_state(self.state)


MCTSNode = DecisionStateNode | AfterStateNode | TerminalStateNode


# MARK: Node Scoring
def ucb_score(child: MCTSNode, parent: MCTSNode) -> float:
    if isinstance(child, TerminalStateNode):
        return child.outcome.value_for_player(sj.get_player(parent.state))
    elif isinstance(child, DecisionStateNode):
        return (
            child.average_visit_value
            + parent.model_output.policy_output.get_action_probability(
                child.prev_action
            )
            * np.sqrt(parent.num_visits)
            / (1 + child.num_visits)
        )
    elif isinstance(child, AfterStateNode):
        # Probability weighted average of value of realized decision nodes
        # as evaluated by model from current player's perspective
        return sum(
            [
                # empirical probability of realized state
                realized_decision_node.num_visits
                / child.children[realized_decision_node.state].num_visits
                * realized_decision_node.model_output.value_output.value_for_player(
                    sj.get_player(parent.state)
                )
                for _, realized_decision_node in child.children.values()
            ]
        )
    else:
        raise ValueError(f"Unknown node type: {type(child)}")


# MARK: MCTS Algorithm
def run_mcts(
    game_state: sj.Skyjo,
    model: skynet.Skynet,
    iterations: int,
) -> MCTSNode:
    root_node = DecisionStateNode(state=game_state, parent=None, prev_action=None)

    for _ in range(iterations):
        search(root_node, model)

    return root_node


def search(node: MCTSNode, model: skynet.Skynet):
    search_path = [node]
    while node.is_expanded:
        node = node.select_child()
        search_path.append(node)
    leaf = node
    if isinstance(leaf, TerminalStateNode):
        value = leaf.outcome
    else:
        leaf.expand(model)
        value = leaf.model_output.value_output.game_state_value

    backpropagate(search_path, value)


def backpropagate(search_path: list[MCTSNode], value: sj.SkyjoOutcome):
    # Update each node's visited cound and average value
    # to be value of leaf node from that player's perspective
    for node in search_path:
        node.average_visit_value = (
            node.average_visit_value * (node.num_visits)
            + value.value_for_player(sj.get_player(node.state))
        ) / (node.num_visits + 1)
        node.num_visits += 1
