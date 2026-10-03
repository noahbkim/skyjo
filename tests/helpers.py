"""Small deterministic helpers for gameplay tests."""

import numpy as np

from skyjo import game as sj
from skyjo.player import AbstractPlayer


class NaiveQuickFinishPlayer(AbstractPlayer):
    """A player that plays an action to finish the game as quickly as possible."""

    def get_action_probabilities(
        self, game_state: sj.Skyjo
    ) -> np.ndarray[tuple[int], np.float32]:
        return self._action_to_action_probabilities(
            sj.quick_finish_action(game_state), game_state
        )
