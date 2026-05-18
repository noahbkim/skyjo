from __future__ import annotations

from collections.abc import Sequence

import torch

import skyjo2 as sj

DTYPE = torch.float32

TURN_SIZE = 1
STATE_SIZE = len(sj.State)
CARD_SIZE = len(sj.DECK)
DRAWN_CARD_SIZE = CARD_SIZE
DRAW_PILE_SIZE = CARD_SIZE
DISCARDED_CARD_SIZE = CARD_SIZE
DISCARD_PILE_SIZE = CARD_SIZE
FINGER_SIZE = CARD_SIZE + 2  # hidden, cleared


def embed_state(state: sj.State, tensor: torch.Tensor) -> torch.Tensor:
    assert tensor.shape == (STATE_SIZE,)
    tensor[0] = state == sj.State.NULL
    tensor[1] = state == sj.State.REVEAL_SECOND_CARD
    tensor[2] = state == sj.State.DRAW_OR_REPLACE_WITH_DISCARD
    tensor[3] = state == sj.State.DISCARD_DRAW_AND_REVEAL_OR_REPLACE_WITH_DRAW
    return tensor


def embed_card_index(card_index: int | None, tensor: torch.Tensor) -> torch.Tensor:
    assert tensor.shape == (CARD_SIZE,)
    tensor[:] = False
    if card_index is not None:
        tensor[card_index] = True
    return tensor


def embed_pile(pile: Sequence[int] | None, tensor: torch.Tensor) -> torch.Tensor:
    assert tensor.shape == (CARD_SIZE,)
    for i, card_count in enumerate(pile):
        tensor[i] = card_count
    return tensor


def embed_finger(finger: sj.Finger, tensor: torch.Tensor) -> torch.Tensor:
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


def embed_game_non_spatial(
    game: sj.Game,
    tensor: torch.Tensor | None = None,
) -> torch.Tensor:
    SCORES_SIZE = len(game.players)
    SIZE = (
        TURN_SIZE
        + STATE_SIZE
        + DRAWN_CARD_SIZE
        + DRAW_PILE_SIZE
        + DISCARDED_CARD_SIZE
        + DISCARD_PILE_SIZE
        + SCORES_SIZE
    )

    if tensor is None:
        tensor = torch.zeros((SIZE,), dtype=DTYPE)

    assert tensor.shape == (SIZE,)

    i = 0

    tensor[i] = game.turn
    i += TURN_SIZE

    embed_state(game.state, tensor[i : i + STATE_SIZE])
    i += STATE_SIZE

    embed_card_index(game.drawn_card_index, tensor[i : i + DRAWN_CARD_SIZE])
    i += DRAWN_CARD_SIZE

    embed_pile(game.draw_pile, tensor[i : i + DRAW_PILE_SIZE])
    i += DRAW_PILE_SIZE

    embed_card_index(game.discarded_card_index, tensor[i : i + DISCARDED_CARD_SIZE])
    i += DISCARDED_CARD_SIZE

    embed_pile(game.discard_pile, tensor[i : i + DISCARD_PILE_SIZE])
    i += DISCARD_PILE_SIZE

    for player in game.players:
        tensor[i] = player.score
        i += 1

    return tensor


def embed_game_spatial(
    game: sj.Game,
    tensor: torch.Tensor | None = None,
) -> torch.Tensor:
    SHAPE = (len(game.players), sj.HAND_ROWS, sj.HAND_COLUMNS, FINGER_SIZE)

    if tensor is None:
        tensor = torch.zeros(SHAPE, dtype=DTYPE)

    assert tensor.shape == SHAPE

    for i, player in enumerate(game.players):
        for j, finger in enumerate(player.hand):
            row, column = divmod(j, sj.HAND_COLUMNS)
            embed_finger(finger, tensor[i, row, column, :])

    return tensor


def embed_game(
    game: sj.Game,
    non_spatial_tensor: torch.Tensor | None = None,
    spatial_tensor: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Embeds the game state as spatial and non-spatial tensors."""

    return (
        embed_game_non_spatial(game, tensor=non_spatial_tensor),
        embed_game_spatial(game, tensor=spatial_tensor),
    )
