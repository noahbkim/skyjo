"""
Module for Skyjo players.
"""

import abc
import dataclasses

import numpy as np

from . import config
from . import mcts
from . import predictor
from . import game as sj


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

    def get_action(self, game_state: sj.Skyjo) -> sj.SkyjoAction:
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


class RandomPlayer(AbstractPlayer):
    """A player that plays a random valid action."""

    def get_action_probabilities(
        self, game_state: sj.Skyjo
    ) -> np.ndarray[tuple[int], np.float32]:
        return self._action_to_action_probabilities(
            sj.random_valid_action(game_state), game_state
        )


@dataclasses.dataclass(slots=True)
class ModelPlayerConfig(config.Config):
    action_softmax_temperature: float
    mcts_iterations: int
    mcts_dirichlet_epsilon: float
    mcts_after_state_evaluate_all_children: bool
    mcts_c_puct: float = 1.5
    mcts_fpu_reduction: float = 0.0


class ModelPlayer(AbstractPlayer):
    """Player that uses MCTS with specified model and parameters."""

    def __init__(
        self,
        inference: predictor.LocalPredictor,
        action_softmax_temperature: float,
        mcts_iterations: int,
        mcts_dirichlet_epsilon: float,
        mcts_after_state_evaluate_all_children: bool,
        mcts_c_puct: float = 1.5,
        mcts_fpu_reduction: float = 0.0,
    ):
        self.inference = inference
        self.action_softmax_temperature = action_softmax_temperature
        self.mcts_iterations = mcts_iterations
        self.mcts_dirichlet_epsilon = mcts_dirichlet_epsilon
        self.mcts_after_state_evaluate_all_children = (
            mcts_after_state_evaluate_all_children
        )
        self.mcts_c_puct = mcts_c_puct
        self.mcts_fpu_reduction = mcts_fpu_reduction

    def get_action_probabilities(
        self, game_state: sj.Skyjo
    ) -> np.ndarray[tuple[int], np.float32]:
        node = self.run_mcts(game_state)
        return node.policy_targets(self.action_softmax_temperature)

    def run_mcts(
        self,
        game_state: sj.Skyjo,
        root_node: mcts.MCTSNode | None = None,
    ) -> mcts.MCTSNode:
        return mcts.run_mcts(
            game_state,
            self.inference,
            self.mcts_iterations,
            dirichlet_epsilon=self.mcts_dirichlet_epsilon,
            after_state_evaluate_all_children=self.mcts_after_state_evaluate_all_children,
            c_puct=self.mcts_c_puct,
            fpu_reduction=self.mcts_fpu_reduction,
            root_node=root_node,
        )


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
