import abc

import numpy as np
import torch

import mcts_new as mcts
import skyjo as sj
import skynet


class AbstractPlayer(abc.ABC):
    @abc.abstractmethod
    def get_action(self, game_state: sj.Skyjo) -> sj.SkyjoAction:
        pass


class NaiveQuickFinishPlayer(AbstractPlayer):
    def get_action(self, game_state: sj.Skyjo) -> sj.SkyjoAction:
        return sj.quick_finish_action(game_state)


class RandomPlayer(AbstractPlayer):
    def get_action(self, game_state: sj.Skyjo) -> sj.SkyjoAction:
        return sj.random_valid_action(game_state)


class GreedyExpectedValuePlayer(AbstractPlayer):
    def get_action(self, game_state: sj.Skyjo) -> sj.SkyjoAction:
        game = game_state[0]
        deck = game_state[2]
        if game[sj.GAME_ACTION + sj.ACTION_FLIP_SECOND]:
            return sj.MASK_FLIP_SECOND_BELOW
        if game[sj.GAME_ACTION + sj.ACTION_DRAW_OR_TAKE]:
            unknown_expected_value = sum(
                [(idx - 2) * count for idx, count in enumerate(deck.astype(np.int32))]
            ) / sum(deck)
            if self._expected_draw_value(
                game_state, unknown_expected_value
            ) > self._expected_take_value(game_state, unknown_expected_value):
                return sj.MASK_TAKE
            else:
                return sj.MASK_DRAW
        if game[sj.GAME_ACTION + sj.ACTION_FLIP_OR_REPLACE]:
            unknown_expected_value = sum(
                [(idx - 2) * count for idx, count in enumerate(deck.astype(np.int32))]
            ) / sum(deck)
            replace_expected_values = []
            flip_coord = None
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
                    flip_coord = (row, column)
            sorted_replace_expected_values = sorted(
                replace_expected_values, key=lambda x: x[1]
            )
            if sorted_replace_expected_values[0][1] >= self._expected_flip_value():
                return sj.MASK_FLIP + flip_coord[0] * sj.COLUMN_COUNT + flip_coord[1]
            else:
                return (
                    sj.MASK_REPLACE
                    + sorted_replace_expected_values[0][0][0] * sj.COLUMN_COUNT
                    + sorted_replace_expected_values[0][0][1]
                )
        if game[sj.GAME_ACTION + sj.ACTION_REPLACE]:
            unknown_expected_value = sum(
                [(idx - 2) * count for idx, count in enumerate(deck.astype(np.int32))]
            ) / sum(deck)
            replace_expected_values = []
            flip_coord = None
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
            return (
                sj.MASK_REPLACE
                + sorted_replace_expected_values[0][0][0] * sj.COLUMN_COUNT
                + sorted_replace_expected_values[0][0][1]
            )
        raise ValueError("No valid action specified")

    def _highest_curr_card_value(self, game_state: sj.Skyjo) -> float:
        cards = []
        for idx in range(sj.FINGER_COUNT):
            row, column = divmod(idx, sj.COLUMN_COUNT)
            if sj.get_finger(game_state, row, column, 0) < sj.CARD_SIZE:
                cards.append(sj.get_finger(game_state, row, column, 0) - 2)
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
            sj.get_top(game_state) - highest_curr_card_value,
            sj.get_top(game_state) - unknown_expected_value,
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
            return sj.get_top(game_state) - unknown_expected_value
        if curr_card == sj.FINGER_CLEARED:
            return float("inf")
        return sj.get_top(game_state) - (curr_card - 2)


class ModelPlayer(AbstractPlayer):
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

    def get_action(self, game_state: sj.Skyjo) -> sj.SkyjoAction:
        self.model.eval()
        with torch.no_grad():
            node = mcts.run_mcts(
                game_state,
                self.model,
                self.mcts_iterations,
                self.afterstate_initial_realizations,
            )
        action_probabilities = node.sample_child_visit_probabilities(self.temperature)
        return np.random.choice(sj.MASK_SIZE, p=action_probabilities)
