"""Small player contract and model-free interactive players."""

from typing import Protocol

import numpy as np

from skyjo.engine import game as sj


class Player(Protocol):
    def get_action_probabilities(self, game_state: sj.Skyjo) -> np.ndarray:
        """Return a normalized distribution supported on legal actions."""
        ...


class RandomPlayer:
    """Assign equal probability to every legal action."""

    def get_action_probabilities(self, game_state: sj.Skyjo) -> np.ndarray:
        mask = sj.actions(game_state).astype(np.float32)
        return mask / mask.sum()


class HumanPlayer:
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
