from __future__ import annotations

import abc
import dataclasses
import typing

import numpy as np

import abstract
import skyjo_immutable as sj
import skynet


class AbstractGameTreeNode(abc.ABC):
    """Abstract base class for nodes in the game tree"""

    @property
    @abc.abstractmethod
    def is_expanded(self) -> bool:
        pass

    @property
    @abc.abstractmethod
    def parent(self) -> typing.Self | None:
        pass

    @property
    @abc.abstractmethod
    def children(self) -> typing.Iterable[typing.Self]:
        pass

    @property
    @abc.abstractmethod
    def game_state(self) -> abstract.AbstractImmutableGameState:
        pass


@dataclasses.dataclass(slots=True)
class TerminalStateNode(AbstractGameTreeNode):
    outcome: abstract.AbstractGameOutcome
    game_state: abstract.AbstractImmutableGameState
    parent: MCTSNode | None
    is_expanded: bool = False

    @property
    def children(self) -> typing.Iterable[typing.Self]:
        return []


@dataclasses.dataclass(slots=True)
class AfterStateNode(AbstractGameTreeNode):
    action: abstract.AbstractGameAction
    game_state: abstract.AbstractImmutableGameState
    parent: MCTSNode | None
    is_expanded: bool = False
    average_visit_value: float = 0.0
    number_of_visits: int = 0

    @property
    def children(self) -> typing.Iterable[typing.Self]:
        assert self.is_expanded, "Must expand node before accessing children"
        return self.children_map.values()

    def __hash__(self) -> int:
        return hash((hash(self.game_state), hash(self.action)))

    def create_child(
        self, game_state: abstract.AbstractImmutableGameState
    ) -> typing.Self:
        if game_state.game_has_ended:
            return TerminalStateNode(
                game_state=game_state,
                outcome=game_state.game_outcome(),
                parent=self,
            )
        return DecisionStateNode(
            game_state=game_state,
            parent=self,
        )

    def expand(self, model: abstract.AbstractModel):
        self.model_output = model.afterstate_predict(self.game_state, self.action)
        self.children_map = {}
        self.is_expanded = True

    def sample_probabilities(self) -> dict[MCTSNode, float]:
        total_visit_count = sum(child.number_of_visits for child in self.children)
        return {
            child: child.number_of_visits / total_visit_count for child in self.children
        }

    def actual_probabilities(self) -> dict[MCTSNode, float]:
        """Optionally, implement an exact probability distribution for the children"""
        raise NotImplementedError

    def select_child(self, c_puct: float):
        realized_next_state = self.game_state.apply_action(self.action)
        if realized_next_state in self.children_map:
            return self.children_map[realized_next_state]
        child = self.create_child(realized_next_state)
        self.children_map[realized_next_state] = child
        return child


@dataclasses.dataclass(slots=True)
class DecisionStateNode(AbstractGameTreeNode):
    game_state: abstract.AbstractImmutableGameState
    parent: MCTSNode | None
    action: abstract.AbstractGameAction | None = None
    is_expanded: bool = False
    average_visit_value: float = 0.0
    number_of_visits: int = 0

    @property
    def children(self) -> typing.Iterable[typing.Self]:
        assert self.is_expanded, "Must expand node before accessing children"
        return self.children_map.values()

    def __hash__(self) -> int:
        return hash(self.game_state)

    def create_child(
        self,
        game_state: abstract.AbstractImmutableGameState,
        action: abstract.AbstractGameAction,
    ) -> typing.Self:
        if game_state.involves_chance(action):
            return AfterStateNode(
                game_state=game_state,
                action=action,
                parent=self,
            )
        next_game_state = game_state.apply_action(action)
        if next_game_state.game_has_ended:
            return TerminalStateNode(
                game_state=next_game_state,
                outcome=next_game_state.game_outcome(),
                parent=self,
            )
        return DecisionStateNode(
            game_state=next_game_state,
            action=action,
            parent=self,
        )

    def expand(self, model: abstract.AbstractModel):
        self.model_output = model.predict(self.game_state)
        # may need to normalize model output here (e.g. masking invalid actions)

        # create child nodes
        self.children_map = {
            action: self.create_child(self.game_state, action)
            for action in self.game_state.valid_actions
        }
        self.is_expanded = True

    def select_child(self, c_puct: float) -> float:
        """Selects a child node based on the child node with the highest Upper Confidence Bound (UCB)"""
        best_action, best_child = max(
            self.children_map.items(),
            key=lambda action_child: action_child[1].average_visit_value
            + c_puct
            * self.model_output.policy_output.get_action_probability(action_child[0])
            * np.sqrt(self.number_of_visits)
            / (1 + action_child[1].number_of_visits),
        )
        return best_child

    def sample_child_probabilities(self, temperature: float) -> dict[MCTSNode, float]:
        if temperature == 0:
            max_visit_count = max(child.number_of_visits for child in self.children)
            visit_counts = {
                child: 1 if child.number_of_visits == max_visit_count else 0
                for child in self.children
            }
        else:
            visit_counts = {
                child: child.number_of_visits ** (1 / temperature)
                for child in self.children
            }
        total_temp_scaled_visit_count = sum(visit_counts.values())
        return {
            child: visit_count / total_temp_scaled_visit_count
            for child, visit_count in visit_counts.items()
        }

    def sample_action_probabilities(
        self, temperature: float
    ) -> dict[abstract.AbstractGameAction, float]:
        child_probabilities = self.sample_child_probabilities(temperature)
        return {
            child.action: probability
            for child, probability in child_probabilities.items()
        }


MCTSNode = DecisionStateNode | AfterStateNode | TerminalStateNode


class Tree:
    """Class for running and storing Monte Carlo Tree Search information for a game"""

    def __init__(self, c_puct: float):
        self.c_puct = c_puct

    def run(
        self,
        game_state: abstract.AbstractImmutableGameState,
        model: abstract.AbstractModel,
        iterations: int,
    ) -> float:
        root_node = DecisionStateNode(
            game_state=game_state,
            parent=None,
        )
        root_node.expand(model)
        for _ in range(iterations):
            self.search(root_node, model)

        return root_node

    def search(self, node: MCTSNode, model: abstract.AbstractModel):
        search_path = [node]
        while node.is_expanded:
            node = node.select_child(self.c_puct)
            search_path.append(node)
        if node.game_state.game_has_ended:
            value = node.outcome
        else:
            node.expand(model)
            value = node.model_output.value_output.game_state_value
        self.update_node_values(search_path, value)

    def update_node_values(
        self, search_path: list[MCTSNode], value: abstract.AbstractGameStateValue
    ):
        for node in reversed(search_path):
            node.number_of_visits += 1
            node.average_visit_value = (
                node.number_of_visits * node.average_visit_value
                + value.value_from_perspective_of(node.game_state.curr_player)
            ) / (node.number_of_visits + 1)


if __name__ == "__main__":
    tree = Tree(c_puct=1.0)
    game_state = sj.ImmutableState(
        num_players=2,
        player_scores=np.array([0, 0], dtype=np.int16),
        remaining_card_counts=sj.CardCounts.create_initial_deck_counts(),
        valid_actions=[sj.SkyjoAction(sj.SkyjoActionType.START_ROUND)],
    ).apply_action(sj.SkyjoAction(sj.SkyjoActionType.START_ROUND))
    model = skynet.SkyNet(
        spatial_input_channels=sj.CARD_TYPES,
        non_spatial_features=game_state.non_spatial_numpy().shape[0],
        num_players=game_state.num_players,
    )
    tree.run(game_state, model, iterations=1600)
