import dataclasses
import logging
import typing

import numpy as np
import torch

import skyjo.abstract as abstract
import skyjo.skyjo_immutable as sj
import skyjo.skynet as skynet


@dataclasses.dataclass(slots=True)
class MCTSNode:
    game_state: abstract.AbstractImmutableGameState
    parent: typing.Self | None
    children: dict[abstract.AbstractGameAction, typing.Self]
    number_of_visits: int
    average_visit_value: float
    is_expanded: bool = False
    normalized_policy: skynet.PolicyOutput | None = None
    model_value: skynet.ValueOutput | None = None

    def expand(self, model: abstract.AbstractModel):
        assert not self.is_expanded, "Node already expanded"
        with torch.no_grad():
            model_output = model.predict(self.game_state.numpy())
        valid_actions_mask = self.game_state.create_valid_actions_mask().reshape(
            -1, *sj.ACTION_SHAPE
        )
        self.normalized_policy = (
            model_output.policy_output.mask_invalid_and_renormalize(valid_actions_mask)
        )
        self.model_value = model_output.value_output

        for action in self.game_state.valid_actions:
            self.children[action] = MCTSNode(
                game_state=self.game_state.next_state(action),
                parent=self,
                children={},
                number_of_visits=0,
                average_visit_value=0,
            )
        self.is_expanded = True

    def select_highest_ucb_child(self, c_puct: float) -> typing.Self:
        max_ucb = -float("inf")
        best_child = None
        for action, child in self.children.items():
            child_ucb = (
                child.average_visit_value
                + c_puct
                * self.normalized_policy.get_action_probability(action)
                * np.sqrt(self.number_of_visits)
                / (1 + child.number_of_visits)
            )
            if child_ucb > max_ucb:
                max_ucb = child_ucb
                best_child = child
        return best_child

    def action_probabilities(
        self, temperature: float
    ) -> dict[abstract.AbstractGameAction, float]:
        actions = [action for action, _ in self.children.items()]
        counts = np.array(
            [child.number_of_visits for _, child in self.children.items()]
        )
        if temperature == 0:
            prob_mass = np.where(counts == np.max(counts), 1, 0)
        else:
            prob_mass = counts ** (1 / temperature)
        prob_mass = prob_mass / np.sum(prob_mass)
        return {action: prob_mass for action, prob_mass in zip(actions, prob_mass)}


class MCTS:
    """Class for running and storing Monte Carlo Tree Search information for a game"""

    def __init__(self, c_puct: float = 1.0):
        self.c_puct = c_puct

    def run(
        self,
        game_state: abstract.AbstractImmutableGameState,
        model: abstract.AbstractModel,
        iterations: int,
    ) -> float:
        root_node = MCTSNode(
            game_state=game_state,
            parent=None,
            children={},
            number_of_visits=0,
            average_visit_value=0,
        )
        root_node.expand(model)
        logging.debug(
            f"Starting search from state:\n{root_node.game_state.display_str()}"
        )
        for _ in range(iterations):
            self.search(root_node, model, self.c_puct)

        return root_node

    def search(self, node, model, c_puct: float):
        search_path = [node]
        while node.is_expanded:
            node = node.select_highest_ucb_child(c_puct)
            search_path.append(node)
        if node.game_state.game_ended():
            logging.debug(f"Found terminal node:\n{node.game_state.display_str()}")
            value = sj.GameStateValue.from_winning_player(
                node.game_state.winning_player,
                node.game_state.curr_player,
                node.game_state.num_players,
            )
        else:
            logging.debug(
                f"Found leaf node to expand:\n{node.game_state.display_str()}"
            )
            node.expand(model)
            value = sj.GameStateValue.from_numpy(
                node.model_value.values, node.game_state.curr_player
            )
        logging.debug(f"Updating searchpath node values: {value}")
        self.update_node_values(search_path, value)

    def update_node_values(self, search_path, value: sj.GameStateValue):
        for node in reversed(search_path):
            node.number_of_visits += 1
            node.average_visit_value = (
                node.number_of_visits * node.average_visit_value
                + value.player_value(node.game_state.curr_player)
            ) / (node.number_of_visits + 1)
