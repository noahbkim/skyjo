"""
Module for Skyjo players.
"""

import abc

import numpy as np
import torch

import mcts
import parallel_mcts
import predictor
import skyjo as sj
import skynet


class AbstractPlayer(abc.ABC):
    """Abstract base class for all players.

    Each implementation must implement the `get_action_probabilities` method.
    This method returns the probability distribution from which to sample the
    next action."""

    def _action_to_action_probabilities(
        self, action: sj.SkyjoAction, game_state: sj.Skyjo
    ) -> np.ndarray[tuple[int], np.float32]:
        action_probabilities = np.zeros(sj.MASK_SIZE, dtype=np.float32)
        action_probabilities[action] = 1.0
        assert sj.actions(game_state)[action]
        return action_probabilities

    def _actions_to_action_probabilities(
        self, actions: list[sj.SkyjoAction], game_state: sj.Skyjo
    ) -> np.ndarray[tuple[int], np.float32]:
        action_probabilities = np.zeros(sj.MASK_SIZE, dtype=np.float32)
        for action in actions:
            action_probabilities[action] = 1.0
            assert sj.actions(game_state)[action]
        action_probabilities /= sum(action_probabilities)
        return action_probabilities

    def get_action(self, game_state: sj.Skyjo) -> np.ndarray[tuple[int], np.float32]:
        action = np.random.choice(
            sj.MASK_SIZE, p=self.get_action_probabilities(game_state)
        )
        assert sj.actions(game_state)[action]
        return action

    @abc.abstractmethod
    def get_action_probabilities(
        self, game_state: sj.Skyjo
    ) -> np.ndarray[tuple[int], np.float32]:
        raise NotImplementedError("Not implemented")


class NaiveQuickFinishPlayer(AbstractPlayer):
    """A player that plays an action to finish the game as quickly as possible."""

    def get_action_probabilities(
        self, game_state: sj.Skyjo
    ) -> np.ndarray[tuple[int], np.float32]:
        return self._action_to_action_probabilities(
            sj.quick_finish_action(game_state), game_state
        )


class RandomPlayer(AbstractPlayer):
    """A player that plays a random valid action."""

    def get_action_probabilities(
        self, game_state: sj.Skyjo
    ) -> np.ndarray[tuple[int], np.float32]:
        return self._action_to_action_probabilities(
            sj.random_valid_action(game_state), game_state
        )


class GreedyExpectedValuePlayer(AbstractPlayer):
    """A player that plays the action with the best greedy expected value.

    Computes the expected value of a remaining random card in the deck. Uses
    this value to determine the 'expected value' of each valid action based on
    how it would lower the board point total. Note this does not account for
    clears and simply considers the raw point values of each card.

    For example, for a replace action on a hidden slot the expected value would
    be top card - average remaining card value. For flip, the value is 0 since
    there is no change in the expected point value of that slot."""

    def get_action_probabilities(
        self, game_state: sj.Skyjo
    ) -> np.ndarray[tuple[int], np.float32]:
        return self._actions_to_action_probabilities(
            self._highest_expected_value_actions(game_state), game_state
        )

    def _highest_expected_value_actions(self, game_state: sj.Skyjo) -> sj.SkyjoAction:
        game = game_state[0]
        deck = game_state[2]
        if game[sj.GAME_ACTION + sj.ACTION_FLIP_SECOND]:
            return [sj.MASK_FLIP_SECOND_BELOW]
        if game[sj.GAME_ACTION + sj.ACTION_DRAW_OR_TAKE]:
            # unknown_expected_value = sum(
            #     [(idx - 2) * count for idx, count in enumerate(deck.astype(np.int32))]
            # ) / sum(deck)
            unknown_expected_value = 5
            if self._expected_draw_value(
                game_state, unknown_expected_value
            ) > self._expected_take_value(game_state, unknown_expected_value):
                return [sj.MASK_TAKE]
            else:
                return [sj.MASK_DRAW]
        if game[sj.GAME_ACTION + sj.ACTION_FLIP_OR_REPLACE]:
            unknown_expected_value = sum(
                [(idx - 2) * count for idx, count in enumerate(deck.astype(np.int32))]
            ) / sum(deck)
            unknown_expected_value = 5
            replace_expected_values = []
            flip_coords = []
            for index in range(sj.FINGER_COUNT):
                row, column = divmod(index, sj.COLUMN_COUNT)
                replace_expected_values.append(
                    (
                        (row, column),
                        self._expected_replace_value(
                            game_state, sj.MASK_REPLACE + index, unknown_expected_value
                        ),
                    )
                )
                if sj.get_finger(game_state, row, column, 0) == sj.FINGER_HIDDEN:
                    flip_coords.append((row, column))
            sorted_replace_expected_values = sorted(
                replace_expected_values, key=lambda x: x[1]
            )
            if sorted_replace_expected_values[0][1] >= self._expected_flip_value():
                return [
                    sj.MASK_FLIP + flip_coord[0] * sj.COLUMN_COUNT + flip_coord[1]
                    for flip_coord in flip_coords
                ]
            else:
                best_ev = sorted_replace_expected_values[0][1]
                best_coords = []
                for coord, ev in sorted_replace_expected_values:
                    if abs(ev - best_ev) < 1e-6:
                        best_coords.append(coord)
                    else:
                        break
                return [
                    sj.MASK_REPLACE + coord[0] * sj.COLUMN_COUNT + coord[1]
                    for coord in best_coords
                ]
        if game[sj.GAME_ACTION + sj.ACTION_REPLACE]:
            unknown_expected_value = sum(
                [(idx - 2) * count for idx, count in enumerate(deck.astype(np.int32))]
            ) / sum(deck)
            unknown_expected_value = 5
            replace_expected_values = []
            for index in range(sj.FINGER_COUNT):
                row, column = divmod(index, sj.COLUMN_COUNT)
                replace_expected_values.append(
                    (
                        (row, column),
                        self._expected_replace_value(
                            game_state, sj.MASK_REPLACE + index, unknown_expected_value
                        ),
                    )
                )
            sorted_replace_expected_values = sorted(
                replace_expected_values, key=lambda x: x[1]
            )
            best_ev = sorted_replace_expected_values[0][1]
            best_coords = []
            for coord, ev in sorted_replace_expected_values:
                if abs(ev - best_ev) < 1e-6:
                    best_coords.append(coord)
                else:
                    break
            return [
                sj.MASK_REPLACE + coord[0] * sj.COLUMN_COUNT + coord[1]
                for coord in best_coords
            ]
        raise ValueError("No valid action specified")

    def _highest_curr_card_value(self, game_state: sj.Skyjo) -> float:
        cards = []
        for idx in range(sj.FINGER_COUNT):
            row, column = divmod(idx, sj.COLUMN_COUNT)
            if sj.get_finger(game_state, row, column, 0) < sj.CARD_SIZE:
                cards.append(sj.get_finger(game_state, row, column, 0) - 2)

        if len(cards) == 0:
            return 0
        return max(cards)

    def _expected_draw_value(
        self,
        game_state: sj.Skyjo,
        unknown_expected_value: float,
    ) -> float:
        highest_curr_card_value = self._highest_curr_card_value(game_state)
        # small incentive to draw a card over placing for when placing from discard wouldn't change value at all
        return min(unknown_expected_value - highest_curr_card_value, -0.01)

    def _expected_take_value(
        self,
        game_state: sj.Skyjo,
        unknown_expected_value: float,
    ) -> float:
        highest_curr_card_value = self._highest_curr_card_value(game_state)
        return min(
            (sj.get_top(game_state) - 2) - highest_curr_card_value,
            (sj.get_top(game_state) - 2) - unknown_expected_value,
        )

    def _expected_flip_value(
        self,
    ) -> float:
        return 0

    def _expected_replace_value(
        self,
        game_state: sj.Skyjo,
        action: sj.SkyjoAction,
        unknown_expected_value: float,
    ) -> float:
        row, col = divmod(action - sj.MASK_REPLACE, sj.COLUMN_COUNT)
        curr_card = sj.get_finger(game_state, row, col, 0)
        if curr_card == sj.FINGER_HIDDEN:
            return (sj.get_top(game_state) - 2) - unknown_expected_value
        if curr_card == sj.FINGER_CLEARED:
            return float("inf")
        return sj.get_top(game_state) - (curr_card)


class SimpleModelPlayer(AbstractPlayer):
    """Runs a simple (non-parallel) MCTS with specified model and parameters."""

    def __init__(
        self,
        model: skynet.SkyNet,
        temperature: float = 0.0,
        mcts_iterations: int = 10,
        afterstate_initial_realizations: int = 10,
    ):
        self.model = model
        self.temperature = temperature
        self.mcts_iterations = mcts_iterations
        self.afterstate_initial_realizations = afterstate_initial_realizations

    def get_action_probabilities(
        self, game_state: sj.Skyjo
    ) -> np.ndarray[tuple[int], np.float32]:
        self.model.eval()
        with torch.no_grad():
            node = mcts.run_mcts(
                game_state,
                self.model,
                self.mcts_iterations,
                self.afterstate_initial_realizations,
            )
        return node.sample_child_visit_probabilities(self.temperature)


class ModelPlayer(AbstractPlayer):
    """Runs a parallel MCTS with specified model and parameters."""

    def __init__(
        self,
        predictor_client: predictor.PredictorClient,
        action_softmax_temperature: float,
        mcts_iterations: int,
        batched_leaf_count: int,
        virtual_loss: float,
        dirichlet_epsilon: float,
        after_state_evaluate_all_children: bool,
        terminal_state_rollouts: int,
    ):
        self.predictor_client = predictor_client
        self.action_softmax_temperature = action_softmax_temperature
        self.mcts_iterations = mcts_iterations
        self.batched_leaf_count = batched_leaf_count
        self.virtual_loss = virtual_loss
        self.dirichlet_epsilon = dirichlet_epsilon
        self.after_state_evaluate_all_children = after_state_evaluate_all_children
        self.terminal_state_rollouts = terminal_state_rollouts

    @classmethod
    def from_mcts_config(
        cls,
        predictor_client: predictor.PredictorClient,
        action_softmax_temperature: float,
        mcts_config: parallel_mcts.Config,
    ):
        return cls(
            predictor_client,
            action_softmax_temperature,
            mcts_config.iterations,
            mcts_config.batched_leaf_count,
            mcts_config.virtual_loss,
            mcts_config.dirichlet_epsilon,
            mcts_config.after_state_evaluate_all_children,
            mcts_config.terminal_state_rollouts,
        )

    def get_action_probabilities(
        self, game_state: sj.Skyjo
    ) -> np.ndarray[tuple[int], np.float32]:
        root = parallel_mcts.run_mcts(
            game_state,
            self.predictor_client,
            self.mcts_iterations,
            self.batched_leaf_count,
            self.virtual_loss,
            self.dirichlet_epsilon,
            self.after_state_evaluate_all_children,
            self.terminal_state_rollouts,
        )
        # print(root)
        return root.sample_child_visit_probabilities(self.action_softmax_temperature)


class PureModelPlayer(AbstractPlayer):
    """Uses only the model to predict the action probabilities."""

    def __init__(self, model: skynet.SkyNet, temperature: float = 1.0):
        self.model = model
        self.temperature = temperature

    def get_action_probabilities(
        self, game_state: sj.Skyjo
    ) -> np.ndarray[tuple[int], np.float32]:
        self.model.eval()
        with torch.no_grad():
            prediction = self.model.predict(game_state)
        if self.temperature == 0:
            action_probabilities = np.zeros(prediction.policy_output.shape)
            action_probabilities[prediction.policy_output.argmax().item()] = 1
            return action_probabilities
        action_probabilities = prediction.policy_output ** (1 / self.temperature)
        action_probabilities = action_probabilities / action_probabilities.sum()
        return action_probabilities
