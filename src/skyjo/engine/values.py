"""Outcome values in fixed seat order and explicit perspective conversions."""

import numpy as np

from . import game as sj

StateValue = np.ndarray


def skyjo_to_state_value(skyjo: sj.Skyjo) -> StateValue:
    """Get the outcome of the game from the fixed perspective."""
    players = skyjo.players
    outcome = np.zeros((players,), dtype=np.float32)
    outcome[sj.get_fixed_perspective_winner(skyjo)] = 1.0
    return outcome


def skyjo_to_game_state_value(skyjo: sj.Skyjo) -> StateValue:
    """Exact full-game outcome in fixed player order, sharing tied wins equally."""
    if not sj.get_game_over(skyjo):
        raise ValueError("Full-game outcomes require a completed game")
    scores = sj.get_fixed_perspective_game_scores(skyjo)
    winners = (scores == scores.min()).astype(np.float32)
    return winners / winners.sum()


def state_value_for_player(state_value: StateValue, player: int) -> float:
    """Get the value of the game for a given player."""
    state_value = state_value.squeeze()
    assert len(state_value.shape) == 1, "Expected a 1D state value"
    return state_value[player].item()


def to_state_value(
    value_output: np.ndarray[tuple[int], np.float32], curr_player: int
) -> StateValue:
    return np.roll(value_output, shift=curr_player)
