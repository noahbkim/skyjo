from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import torch

from skyjo2 import DECK, HAND_COLUMNS, HAND_ROWS, Finger, Game, GameState

DTYPE = torch.float32

TURN_SIZE = 1
STATE_SIZE = len(GameState)
CARD_SIZE = len(DECK)
DRAWN_CARD_SIZE = CARD_SIZE
DRAW_PILE_SIZE = CARD_SIZE
DISCARDED_CARD_SIZE = CARD_SIZE
DISCARD_PILE_SIZE = CARD_SIZE
FINGER_SIZE = CARD_SIZE + 2  # hidden, cleared


def get_state_embedding(state: GameState, tensor: torch.Tensor) -> torch.Tensor:
    assert tensor.shape == (STATE_SIZE,)
    tensor[0] = state == GameState.DEAL_FIRST_CARDS
    tensor[1] = state == GameState.REVEAL_SECOND_CARD
    tensor[2] = state == GameState.DRAW_OR_REPLACE_WITH_DISCARD
    tensor[3] = state == GameState.DISCARD_DRAW_AND_REVEAL_OR_REPLACE_WITH_DRAW
    tensor[4] = state == GameState.REVEAL_HIDDEN_CARDS
    return tensor


def get_card_embedding(card_index: int | None, tensor: torch.Tensor) -> torch.Tensor:
    assert tensor.shape == (CARD_SIZE,)
    tensor[:] = False
    if card_index is not None:
        tensor[card_index] = True
    return tensor


def get_pile_embedding(
    pile: Sequence[int] | None,
    tensor: torch.Tensor,
) -> torch.Tensor:
    assert tensor.shape == (CARD_SIZE,)
    for i, card_count in enumerate(pile):
        tensor[i] = card_count
    return tensor


def get_finger_embedding(finger: Finger, tensor: torch.Tensor) -> torch.Tensor:
    assert tensor.shape == (FINGER_SIZE,)
    tensor[:] = False
    if finger.is_hidden:
        tensor[CARD_SIZE] = True
    elif finger.is_cleared:
        tensor[CARD_SIZE + 1] = True
    else:
        assert finger.card_index is not None
        tensor[finger.card_index] = True
    return tensor


def get_game_non_spatial_shape(game: Game) -> tuple[int, ...]:
    SCORES_SIZE = len(game.players)
    return (
        TURN_SIZE
        + STATE_SIZE
        + DRAWN_CARD_SIZE
        + DRAW_PILE_SIZE
        + DISCARDED_CARD_SIZE
        + DISCARD_PILE_SIZE
        + SCORES_SIZE,
    )


def get_game_non_spatial_embedding(
    game: Game,
    tensor: torch.Tensor | None = None,
) -> torch.Tensor:
    if tensor is None:
        tensor = torch.zeros(get_game_non_spatial_shape(game), dtype=DTYPE)
    else:
        assert tensor.shape == get_game_non_spatial_shape(game)

    i = 0

    tensor[i] = game.turn
    i += TURN_SIZE

    get_state_embedding(game.state, tensor[i : i + STATE_SIZE])
    i += STATE_SIZE

    get_card_embedding(game.drawn_card_index, tensor[i : i + DRAWN_CARD_SIZE])
    i += DRAWN_CARD_SIZE

    get_pile_embedding(game.draw_pile, tensor[i : i + DRAW_PILE_SIZE])
    i += DRAW_PILE_SIZE

    get_card_embedding(game.discarded_card_index, tensor[i : i + DISCARDED_CARD_SIZE])
    i += DISCARDED_CARD_SIZE

    get_pile_embedding(game.discard_pile, tensor[i : i + DISCARD_PILE_SIZE])
    i += DISCARD_PILE_SIZE

    for player in game.players:
        tensor[i] = player.score
        i += 1

    return tensor


def get_game_spatial_shape(game: Game) -> tuple[int, ...]:
    PLAYERS_SIZE = len(game.players)
    return (PLAYERS_SIZE, HAND_ROWS, HAND_COLUMNS, FINGER_SIZE)


def get_game_spatial_embedding(
    game: Game,
    tensor: torch.Tensor | None = None,
) -> torch.Tensor:
    if tensor is None:
        tensor = torch.zeros(get_game_spatial_shape(game), dtype=DTYPE)
    else:
        assert tensor.shape == get_game_spatial_shape(game)

    for i, player in enumerate(game.players):
        for j, finger in enumerate(player.hand):
            row, column = divmod(j, HAND_COLUMNS)
            get_finger_embedding(finger, tensor[i, row, column, :])

    return tensor


class GameShape(NamedTuple):
    non_spatial: tuple[int, ...]
    spatial: tuple[int, ...]


def get_game_shape(game: Game) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return GameShape(
        non_spatial=get_game_non_spatial_shape(game),
        spatial=get_game_spatial_shape(game),
    )


class GameEmbedding(NamedTuple):
    non_spatial: torch.Tensor
    spatial: torch.Tensor


def get_game_embedding(
    game: Game,
    non_spatial_tensor: torch.Tensor | None = None,
    spatial_tensor: torch.Tensor | None = None,
) -> GameEmbedding:
    """Embeds the game state as spatial and non-spatial tensors."""

    return GameEmbedding(
        non_spatial=get_game_non_spatial_embedding(game, tensor=non_spatial_tensor),
        spatial=get_game_spatial_embedding(game, tensor=spatial_tensor),
    )
