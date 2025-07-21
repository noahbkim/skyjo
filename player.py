"""
Module for Skyjo players.
"""

import abc
import dataclasses

import numpy as np
import torch

import config
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


class CappedModelPlayer(AbstractPlayer):
    """Player that uses a capped MCTS with specified model and parameters."""

    def __init__(
        self,
        predictor_client: predictor.AbstractPredictorClient,
        action_softmax_temperature: float,
        full_search_rate: float,
        fast_mcts_iterations: int,
        full_mcts_iterations: int,
        fast_mcts_dirichlet_epsilon: float,
        full_mcts_dirichlet_epsilon: float,
        fast_mcts_after_state_evaluate_all_children: bool,
        fast_mcts_terminal_state_rollouts: int,
        full_mcts_after_state_evaluate_all_children: bool,
        full_mcts_terminal_state_rollouts: int,
        fast_mcts_forced_playout_k: float | None,
        full_mcts_forced_playout_k: float | None,
    ):
        self.predictor_client = predictor_client
        self.action_softmax_temperature = action_softmax_temperature
        self.full_search_rate = full_search_rate
        self.fast_mcts_iterations = fast_mcts_iterations
        self.full_mcts_iterations = full_mcts_iterations
        self.fast_mcts_dirichlet_epsilon = fast_mcts_dirichlet_epsilon
        self.full_mcts_dirichlet_epsilon = full_mcts_dirichlet_epsilon
        self.fast_mcts_after_state_evaluate_all_children = (
            fast_mcts_after_state_evaluate_all_children
        )
        self.full_mcts_after_state_evaluate_all_children = (
            full_mcts_after_state_evaluate_all_children
        )
        self.fast_mcts_terminal_state_rollouts = fast_mcts_terminal_state_rollouts
        self.full_mcts_terminal_state_rollouts = full_mcts_terminal_state_rollouts
        self.fast_mcts_forced_playout_k = fast_mcts_forced_playout_k
        self.full_mcts_forced_playout_k = full_mcts_forced_playout_k

    def get_action_probabilities(
        self, game_state: sj.Skyjo
    ) -> np.ndarray[tuple[int], np.float32]:
        if np.random.random() < self.full_search_rate:
            root = mcts.run_mcts(
                game_state,
                self.predictor_client,
                self.full_mcts_iterations,
                self.full_mcts_dirichlet_epsilon,
                self.full_mcts_after_state_evaluate_all_children,
                self.full_mcts_terminal_state_rollouts,
                self.full_mcts_forced_playout_k,
            )
            return root.policy_targets(
                self.action_softmax_temperature, self.full_mcts_forced_playout_k
            )
        else:
            root = mcts.run_mcts(
                game_state,
                self.predictor_client,
                self.fast_mcts_iterations,
                self.fast_mcts_dirichlet_epsilon,
                self.fast_mcts_after_state_evaluate_all_children,
                self.fast_mcts_terminal_state_rollouts,
                self.fast_mcts_forced_playout_k,
            )
            return root.policy_targets(
                self.action_softmax_temperature, self.fast_mcts_forced_playout_k
            )


@dataclasses.dataclass(slots=True)
class ModelPlayerConfig(config.Config):
    action_softmax_temperature: float
    mcts_iterations: int
    mcts_dirichlet_epsilon: float
    mcts_after_state_evaluate_all_children: bool
    mcts_terminal_state_initial_rollouts: int
    mcts_forced_playout_k: float | None = None


class ModelPlayer(AbstractPlayer):
    """Player that uses MCTS with specified model and parameters."""

    def __init__(
        self,
        predictor_client: predictor.AbstractPredictorClient,
        action_softmax_temperature: float,
        mcts_iterations: int,
        mcts_dirichlet_epsilon: float,
        mcts_after_state_evaluate_all_children: bool,
        mcts_terminal_state_initial_rollouts: int,
        mcts_forced_playout_k: float | None = None,
    ):
        self.predictor_client = predictor_client
        self.action_softmax_temperature = action_softmax_temperature
        self.mcts_iterations = mcts_iterations
        self.mcts_dirichlet_epsilon = mcts_dirichlet_epsilon
        self.mcts_after_state_evaluate_all_children = (
            mcts_after_state_evaluate_all_children
        )
        self.mcts_terminal_state_initial_rollouts = mcts_terminal_state_initial_rollouts
        self.mcts_forced_playout_k = mcts_forced_playout_k

    def get_action_probabilities(
        self, game_state: sj.Skyjo
    ) -> np.ndarray[tuple[int], np.float32]:
        node = self.run_mcts(game_state)
        return node.policy_targets(
            self.action_softmax_temperature, self.mcts_forced_playout_k
        )

    def run_mcts(
        self,
        game_state: sj.Skyjo,
        root_node: mcts.MCTSNode | None = None,
    ) -> mcts.MCTSNode:
        return mcts.run_mcts(
            game_state,
            self.predictor_client,
            self.mcts_iterations,
            self.mcts_dirichlet_epsilon,
            self.mcts_after_state_evaluate_all_children,
            self.mcts_terminal_state_initial_rollouts,
            self.mcts_forced_playout_k,
            root_node,
        )


@dataclasses.dataclass(slots=True)
class BatchedModelPlayerConfig(config.Config):
    action_softmax_temperature: float
    mcts_iterations: int
    mcts_dirichlet_epsilon: float
    mcts_after_state_evaluate_all_children: bool
    mcts_terminal_state_initial_rollouts: int
    mcts_batched_leaf_count: int
    mcts_virtual_loss: float
    mcts_forced_playout_k: float | None


class BatchedModelPlayer(AbstractPlayer):
    """Runs a parallel MCTS with specified model and parameters."""

    def __init__(
        self,
        predictor_client: predictor.AbstractPredictorClient,
        action_softmax_temperature: float,
        mcts_iterations: int,
        mcts_dirichlet_epsilon: float,
        mcts_after_state_evaluate_all_children: bool,
        mcts_terminal_state_initial_rollouts: int,
        mcts_batched_leaf_count: int,
        mcts_virtual_loss: float,
        mcts_forced_playout_k: float | None,
    ):
        self.predictor_client = predictor_client
        self.action_softmax_temperature = action_softmax_temperature
        self.mcts_iterations = mcts_iterations
        self.mcts_dirichlet_epsilon = mcts_dirichlet_epsilon
        self.mcts_after_state_evaluate_all_children = (
            mcts_after_state_evaluate_all_children
        )
        self.mcts_terminal_state_initial_rollouts = mcts_terminal_state_initial_rollouts
        self.mcts_batched_leaf_count = mcts_batched_leaf_count
        self.mcts_virtual_loss = mcts_virtual_loss
        self.mcts_forced_playout_k = mcts_forced_playout_k

    def get_action_probabilities(
        self,
        game_state: sj.Skyjo,
    ) -> np.ndarray[tuple[int], np.float32]:
        node = self.run_mcts(game_state)
        return node.policy_targets(
            self.action_softmax_temperature, self.mcts_forced_playout_k
        )

    def run_mcts(
        self,
        game_state: sj.Skyjo,
        root_node: mcts.MCTSNode | None = None,
    ) -> mcts.MCTSNode:
        return parallel_mcts.run_mcts(
            game_state,
            self.predictor_client,
            self.mcts_iterations,
            self.mcts_dirichlet_epsilon,
            self.mcts_after_state_evaluate_all_children,
            self.mcts_terminal_state_initial_rollouts,
            self.mcts_batched_leaf_count,
            self.mcts_virtual_loss,
            self.mcts_forced_playout_k,
            root_node,
        )


class PureModelPolicyPlayer(AbstractPlayer):
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


class PureModelValuePlayer(AbstractPlayer):
    """Uses only the model to predict the value of the game."""

    def __init__(self, model: skynet.SkyNet, terminal_state_rollouts: int = 10):
        self.model = model
        self.terminal_state_rollouts = terminal_state_rollouts

    def get_action_probabilities(
        self, game_state: sj.Skyjo
    ) -> np.ndarray[tuple[int], np.float32]:
        self.model.eval()
        action_values = sj.actions(game_state) * 1e-7
        for action in sj.get_actions(game_state):
            if sj.get_game_about_to_end(game_state):
                for _ in range(self.terminal_state_rollouts):
                    next_state = sj.apply_action(game_state, action)
                    winner = sj.get_winner(next_state)
                    if winner == 0:
                        action_values[action] += 1 / self.terminal_state_rollouts

            elif sj.is_action_random(action, game_state):
                spatial_inputs = []
                non_spatial_inputs = []
                action_masks = []
                cards_remaining = np.sum(sj.get_deck(game_state))
                next_states = []
                for card, card_count in enumerate(sj.get_deck(game_state)):
                    if card_count > 0:
                        next_state = sj.apply_action(
                            sj.preordain(game_state, card), action
                        )
                        spatial_inputs.append(
                            skynet.get_spatial_state_numpy(next_state)
                        )
                        non_spatial_inputs.append(
                            skynet.get_non_spatial_state_numpy(next_state)
                        )
                        action_masks.append(sj.actions(next_state))
                        next_states.append(next_state)
                model_output = self.model.forward(
                    torch.tensor(
                        spatial_inputs, dtype=torch.float32, device=self.model.device
                    ),
                    torch.tensor(
                        non_spatial_inputs,
                        dtype=torch.float32,
                        device=self.model.device,
                    ),
                    torch.tensor(
                        action_masks, dtype=torch.float32, device=self.model.device
                    ),
                )
                idx = 0
                for card, card_count in enumerate(sj.get_deck(game_state)):
                    if card_count > 0:
                        model_value_prediction = model_output[0][idx]
                        if self.model.device != torch.device("cpu"):
                            value_output = model_value_prediction.cpu().detach().numpy()
                        else:
                            value_output = model_value_prediction.detach().numpy()
                        action_values[action] += (
                            skynet.to_state_value(
                                value_output,
                                sj.get_player(next_states[idx]),
                            )[sj.get_player(game_state)]
                            * card_count
                            / cards_remaining
                        )
                        idx += 1

            else:
                next_state = sj.apply_action(game_state, action)
                model_prediction = self.model.predict(next_state)
                action_values[action] = skynet.to_state_value(
                    model_prediction.value_output, sj.get_player(next_state)
                )[sj.get_player(game_state)]
        # print(action_values)
        # print(sj.get_action_name(np.argmax(action_values).item()))
        action_probabilities = np.zeros(sj.MASK_SIZE, dtype=np.float32)
        action_probabilities[np.argmax(action_values).item()] = 1
        return action_probabilities


class HumanPlayer(AbstractPlayer):
    """A player that allows the human to play the game."""

    def get_action_probabilities(
        self, game_state: sj.Skyjo
    ) -> np.ndarray[tuple[int], np.float32]:
        print(sj.visualize_state(game_state))
        game = sj.get_game(game_state)
        action_probabilities = np.zeros(sj.MASK_SIZE, dtype=np.float32)
        if game[sj.GAME_ACTION + sj.ACTION_FLIP_SECOND]:
            print("0: flip second card in same column")
            print("1: flip second card in different column")
            action = int(input("Enter action: "))
            assert action in (0, 1)
        elif game[sj.GAME_ACTION + sj.ACTION_DRAW_OR_TAKE]:
            print("0: Draw")
            print("1: Take")
            action = int(input("Enter action: "))
            assert action in (0, 1)
            action = sj.MASK_DRAW + action
        else:
            if game[sj.GAME_ACTION + sj.ACTION_FLIP_OR_REPLACE]:
                print("0: Flip")
                print("1: Replace")
                user_input = input("Enter action: ").strip().lower()
                assert int(user_input) in (0, 1)
                if user_input == "0":
                    action = sj.MASK_FLIP
                elif user_input == "1":
                    action = sj.MASK_REPLACE
            else:
                assert game[sj.GAME_ACTION + sj.ACTION_REPLACE]
                action = sj.MASK_REPLACE
            user_input = (
                input("Enter row and columns (comma separated): ").strip().lower()
            )
            assert len(user_input.split(",")) == 2
            row, column = map(int, user_input.split(","))
            assert 0 <= row < sj.ROW_COUNT
            assert 0 <= column < sj.COLUMN_COUNT
            action = action + row * sj.COLUMN_COUNT + column
        assert sj.actions(game_state)[action]
        print(sj.get_action_name(action))
        action_probabilities[action] = 1
        return action_probabilities
