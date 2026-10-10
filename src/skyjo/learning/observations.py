"""Model observations in current-player order; feature order is a persisted contract."""

import numpy as np

from skyjo.engine import game as sj


def get_spatial_state_numpy(
    skyjo: sj.Skyjo,
) -> np.ndarray[tuple[int, int, int, int], np.float32]:
    return sj.get_table(skyjo).astype(np.float32)


def get_non_spatial_input_shape(players: int) -> tuple[int]:
    """Return the complete non-spatial observation shape for a player count."""
    if players < 1:
        raise ValueError("players must be positive")
    return (sj.GAME_SIZE + sj.CARD_SIZE + 2 + players,)


def get_non_spatial_state_numpy(
    skyjo: sj.Skyjo,
) -> np.ndarray[tuple[int], np.float32]:
    players = sj.get_player_count(skyjo)
    turn = sj.get_turn(skyjo)
    countdown = sj.get_countdown(skyjo)
    turns_since_reveal = turn - sj.get_last_revealed_turns(skyjo)
    observation = np.concatenate(
        (
            sj.get_game(skyjo),
            sj.get_deck(skyjo),
            np.array(
                [turn, -1 if countdown is None else countdown],
                dtype=np.int16,
            ),
            turns_since_reveal,
        )
    ).astype(np.float32)
    expected_shape = get_non_spatial_input_shape(players)
    if observation.shape != expected_shape:
        raise ValueError(
            f"non-spatial observation has shape {observation.shape}, "
            f"expected {expected_shape}"
        )
    return observation


def spatial_input_shape(players: int) -> tuple[int, int, int, int]:
    return (players, sj.ROW_COUNT, sj.COLUMN_COUNT, sj.FINGER_SIZE)


def action_mask_shape() -> tuple[int]:
    return (sj.MASK_SIZE,)


def action_mask(state: sj.Skyjo) -> np.ndarray:
    return sj.actions(state).astype(np.float32)
