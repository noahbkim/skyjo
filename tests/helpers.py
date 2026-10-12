"""Small deterministic helpers for gameplay tests."""

import numpy as np

from skyjo.engine import game as sj


class NaiveQuickFinishPlayer:
    """A player that plays an action to finish the game as quickly as possible."""

    def get_action_probabilities(
        self, game_state: sj.Skyjo
    ) -> np.ndarray[tuple[int], np.float32]:
        probabilities = np.zeros(sj.MASK_SIZE, dtype=np.float32)
        probabilities[sj.quick_finish_action(game_state)] = 1
        return probabilities
