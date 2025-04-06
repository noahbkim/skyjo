from __future__ import annotations

import math
import random
from typing import Protocol, TypeAlias

import numpy as np

CARD_N2 = 0
CARD_N1 = 1
CARD_0 = 2
CARD_P1 = 3
CARD_P2 = 4
CARD_P3 = 5
CARD_P4 = 6
CARD_P5 = 7
CARD_P6 = 8
CARD_P7 = 9
CARD_P8 = 10
CARD_P9 = 11
CARD_P10 = 12
CARD_P11 = 13
CARD_P12 = 14
CARD_SIZE = 15

CARD_COUNTS = [5, 10, 15, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
CARD_COUNT = sum(CARD_COUNTS)

# 0-14 are set with CARD_*
FINGER_HIDDEN = 15
FINGER_CLEARED = 16
FINGER_SIZE = 17  # 15 numbers, clear, hidden

PLAYER_COUNT = 8
ROW_COUNT = 3
COLUMN_COUNT = 4
FINGER_COUNT = ROW_COUNT * COLUMN_COUNT

ACTION_FLIP_SECOND = 0  # Revealing cards in hand
ACTION_DRAW_OR_TAKE = 1  # Start of turn, whether to draw or take discard
ACTION_FLIP_OR_REPLACE = 2  # After draw, whether to discard and flip or replace
ACTION_REPLACE = 3  # After take discard, which card to replace
ACTION_SIZE = 4

MASK_FLIP_SECOND_BELOW = 0
MASK_FLIP_SECOND_RIGHT = 1
MASK_DRAW = 2
MASK_TAKE = 3
MASK_FLIP = 4
MASK_REPLACE = MASK_FLIP + 12
MASK_SIZE = MASK_REPLACE + 12

GAME_TOP = 0
GAME_DISCARDS = GAME_TOP + CARD_SIZE
GAME_ACTION = GAME_DISCARDS + CARD_SIZE
GAME_SCORES = GAME_ACTION + ACTION_SIZE
GAME_SIZE = GAME_SCORES + PLAYER_COUNT


assert CARD_SIZE == CARD_P12 + 1
assert FINGER_COUNT == 12
assert GAME_DISCARDS == 15
assert GAME_ACTION == 30
assert GAME_SCORES == 34
assert GAME_SIZE == 42

Game: TypeAlias = np.ndarray[tuple[int], np.uint8]
"""Top discard/drawn card, count of discards, player scores."""

Table: TypeAlias = np.ndarray[tuple[int, int, int, int], np.uint8]
"""A tensor representing (player, row, column, card) tuples."""

Deck: TypeAlias = np.ndarray[tuple[int], np.uint8]
"""Cards remaining in the deck in lieu of a random seed."""

Skyjo: TypeAlias = tuple[Game, Table, Deck, int, int]
"""A lightweight representation of a Skyjo game."""


# MARK: Random


class Random(Protocol):
    """Minimal interface for retrieving random numbers."""

    def random(self) -> float:
        """Return a random `float` in [0, 1)."""


# MARK: Helpers


def _get_top(game: Game) -> int | None:
    """Get the current top card index."""

    for i in range(CARD_SIZE):
        if game[GAME_TOP + i]:
            return i

    return None


def _pop_top(game: Game) -> int | None:
    """Get the current top card index, clearing it."""

    for i in range(CARD_SIZE):
        if game[GAME_TOP + i]:
            game[GAME_TOP + i] = 0
            return i

    return None


def _swap_top(game: Game, card: int) -> int | None:
    """Switch out the discard without adding to the pile."""

    top = _pop_top(game)
    game[GAME_TOP + card] = 1
    return top


def _push_top(game: Game, card: int) -> None:
    """Draw a card or place a new discard on the pile."""

    # Get the current top and add it to permanent discards
    top = _pop_top(game)
    game[GAME_TOP + card] = 1
    game[GAME_DISCARDS + top] += 1


def _rotate_scores(game: Game, players: int) -> None:
    """Rotate the player scores once left."""

    swap = game[GAME_SCORES]
    for i in range(players - 1):
        game[GAME_SCORES + i] = game[GAME_SCORES + i + 1]
    game[GAME_SCORES + players - 1] = swap


def _rotate_table(table: Table, players: int) -> None:
    """Copy `table`, rotating all hands left by one."""

    swap = table[0].copy()
    for i in range(players - 1):
        table[i] = table[i + 1]
    table[players - 1] = swap


def _clear_columns(table: Table) -> None:
    """Clear any columns where all values match."""

    for i in range(COLUMN_COUNT):
        for j in range(CARD_SIZE):  # Not finger size, skip hidden and cleared
            if table[0, 0, i, j] and table[0, 1, i, j] and table[0, 2, i, j]:
                table[0, :, i, j] = 0
                table[0, :, i, FINGER_CLEARED] = 1


def _choose_card(deck: Deck, rng: Random) -> int:
    """Choose one of the remaining cards in `deck`."""

    choice = math.floor(rng.random() * np.sum(deck))
    for i in range(CARD_SIZE):
        remaining = CARD_COUNTS[i] - deck[GAME_DISCARDS + i]
        if remaining > choice:
            return i
        choice -= remaining

    assert False, "fuck you"


def _remove_card(deck: Deck, card: int) -> None:
    """Remove `card` from `deck`."""

    if deck[card] == 0:
        raise ValueError(f"{deck!r} has no card {card} remaining")

    deck[card] -= 1


# MARK: Construction


def new(players: int) -> Skyjo:
    """Generate a fresh game with no visible discard."""

    game = np.ndarray((GAME_SIZE,), dtype=np.uint8)
    game[GAME_ACTION + ACTION_FLIP_SECOND] = 1
    table_shape = (PLAYER_COUNT, ROW_COUNT, COLUMN_COUNT, FINGER_SIZE)
    table = np.ndarray(table_shape, dtype=np.uint8, order="C")
    table[:players, :, :, FINGER_HIDDEN] = 1
    deck = np.ndarray((CARD_SIZE,), dtype=np.uint8)
    deck[:] = CARD_COUNTS
    return game, table, deck, players, 0


# MARK: Convenience


def get_draw_count(skyjo: Skyjo) -> int:
    """Get the number of cards left in the deck."""

    return np.sum(skyjo[2])


def get_discard_count(skyjo: Skyjo) -> int:
    """Get the number of discarded cards excluding the visible discard."""

    game = skyjo[0]

    total = 0
    for i in range(CARD_SIZE):
        total += game[GAME_DISCARDS + i]

    return total


def get_top(skyjo: Skyjo) -> int | None:
    """Get the value of the currently-drawn card.

    Immediately after `draw` is called, this method will return the
    drawn card. For all other game states, this method will return the
    last discard. This method returns the card index, not its value,
    so a return of 0 corresponds to the -2 card.
    """

    card = _get_top(skyjo[0])
    return card if card is not None else None


def get_finger(skyjo: Skyjo, row: int, column: int) -> int:
    """Get the state of a given card in hand.

    This method returns the card index, not its value, so a return of
    0 corresponds to the -2 card. A return of 15 indicates the card is
    hidden and a return of 16 indicates the column has been cleared.
    """

    table = skyjo[1]

    for i in range(FINGER_SIZE):
        if table[0, row, column, i]:
            return i

    assert False, f"{skyjo!r} has no card at ({row}, {column})"


def get_is_visible(skyjo: Skyjo) -> bool:
    """Whether the current player's board is completely revealed."""

    table = skyjo[1]

    return any(table[0, :, :, FINGER_HIDDEN])


def get_score(skyjo: Skyjo, player: int = 0) -> int:
    """Get the score of the board. Throws if any cards are hidden."""

    table = skyjo[1]

    score = 0
    for row in range(ROW_COUNT):
        for column in range(COLUMN_COUNT):
            if table[player, row, column, FINGER_HIDDEN]:
                raise ValueError(f"{skyjo!r} player {player} still has hidden cards")
            for card in range(CARD_SIZE):  # Skip CLEARED, score zero
                if table[player, row, column, card]:
                    score += card - 2

    return score


# MARK: Actions


def deal(skyjo: Skyjo, card: int | None = None, rng: Random = random) -> Skyjo:
    """Flip the first discard. Must be called before gameplay."""

    game = skyjo[0]
    table = skyjo[1]
    deck = skyjo[2]
    players = skyjo[3]
    player = skyjo[4]

    assert _get_top(game) is None

    if card is None:
        card = _choose_card(deck, rng)

    new_game = game.copy()
    top = _swap_top(new_game, card)
    new_deck = deck.copy()
    _remove_card(new_deck, card)

    assert top is None

    return new_game, table, new_deck, players, player


def draw(skyjo: Skyjo, card: int | None = None, rng: Random = random) -> Skyjo:
    """Draw a random card from the deck, placing it in top position.

    This method constructs a copy of `skyjo` with a card randomly
    removed from its `deck` and placed at the top of its `game`.
    """

    game = skyjo[0]
    table = skyjo[1]
    deck = skyjo[2]
    players = skyjo[3]
    player = skyjo[4]

    # If no card was provided, randomly pick one.
    if card is None:
        card = _choose_card(deck, rng)

    new_game = game.copy()
    _push_top(new_game, card)
    new_deck = deck.copy()
    _remove_card(new_deck, card)

    return new_game, table, new_deck, players, player


def take(skyjo: Skyjo) -> Skyjo:
    """Take the last discard.

    Since the discard is already stored at the top of the `game`, this
    method is a no-op.
    """

    return skyjo


def flip(
    skyjo: Skyjo,
    row: int,
    column: int,
    card: int | None = None,
    rotate: bool = True,
    rng: Random = random,
) -> Skyjo:
    """Flip an unrevealed card, completing a turn.

    This method returns a copy of `skyjo` with the specified finger
    flipped to a random card drawn from the deck. The returned copy is
    also rotated to be centered on the next player.
    """

    game = skyjo[0]
    table = skyjo[1]
    deck = skyjo[2]
    players = skyjo[3]
    player = skyjo[4]

    # Make sure this is a valid move
    if not table[0, row, column, FINGER_HIDDEN]:
        raise ValueError(f"{skyjo!r} cannot flip visible ({row}, {column})")

    # If no card was provided, randomly pick one.
    if card is None:
        card = _choose_card(deck, rng)

    # No physical change to the top card, only semantic. What was the
    # draw card is now considered the discard. Rotate scores after.
    if rotate:
        new_game = game.copy()
        _rotate_scores(new_game, players)
    else:
        new_game = game

    # Copy the table, flipping our card and rotating hands.
    new_table = table.copy()
    new_table[0, row, column, FINGER_HIDDEN] = 0
    new_table[0, row, column, card] = 1
    _clear_columns(new_table)
    if rotate:
        _rotate_table(new_table, players)

    # Copy the deck, removing the card we chose.
    new_deck = deck.copy()
    _remove_card(new_deck, card)

    return new_game, new_table, new_deck, players, player + 1


def replace(
    skyjo: Skyjo,
    row: int,
    column: int,
    card: int | None = None,
    rng: Random = random,
) -> Skyjo:
    """Replace a card with the current draw, completing a turn.

    This method returns a copy of `skyjo` with the specified finger
    and draw card swapped. If the finger to replace is hidden, a card
    is randomly drawn from the deck and discarded to simulate revealing
    it. This random draw may be determined by setting `card`.
    """

    game = skyjo[0]
    table = skyjo[1]
    deck = skyjo[2]
    players = skyjo[3]
    player = skyjo[4]

    # If the finger is currently hidden, we need to draw, but only if
    # `card` is not specified.
    if table[0, row, column, FINGER_HIDDEN]:
        finger = FINGER_HIDDEN
        if card is None:
            card = _choose_card(deck, rng)

    # Otherwise, ensure no `card` was specified and determine which
    # card is currently at the given coordinates.
    else:
        if card is not None:
            raise ValueError("Draw specified when replacing revealed card!")
        for finger in range(CARD_SIZE):
            if table[0, row, column, finger]:
                break
        else:
            assert table[0, row, column, FINGER_CLEARED]
            raise ValueError(f"{skyjo!r} cannot replace cleared ({row}, {column})")

    # Replace the current discard with `card`
    new_game = game.copy()
    top = _swap_top(new_game, card)
    _rotate_scores(new_game, players)

    # Clear the current card in our hand and replace it with top, i.e.
    # the draw or last discard depending on our choice. Rotate after.
    new_table = table.copy()
    new_table[0, row, column, finger] = 0
    new_table[0, row, column, top] = 1
    _clear_columns(new_table)
    _rotate_table(new_table, players)

    # Remove the card from the deck for the next iteration.
    new_deck = deck.copy()
    _remove_card(new_deck, card)

    return new_game, new_table, new_deck, players, player + 1


# MARK: Learning


def actions(skyjo: Skyjo) -> np.ndarray[tuple[int], np.uint8]:
    """Generate a mask of possible actions from the current state.

    The mask is a flat, one-dimensional array where each index
    corresponds to a valid action:

      - [0, 2) are for the start of the game when players reveal the
        first two cards in their board. The simulation automatically
        flips the top left card, then the model is asked to flip
        either the card immediately below (index 0) or immediately
        to the right (index 1).
      - [2, 4) are for the start of each turn, during which players
        must decide to either draw a card (index 2) or take the last
        discard (index 3).
      - [4, 16) represent discarding the drawn card and flipping any
        currently-hidden one in hand. The row and column are computed
        by `divmod(index - 4, COLUMN_COUNT)`. This action is only
        allowed if the player drew a card in the previous decision.
      - [16, 28) represent replacing a card in hand with the one drawn
        or taken from the discard. The row and column are computed by
        `divmod(index - 16, ROW_COUNT)`.
    """

    game = skyjo[0]
    table = skyjo[1]

    mask = np.ndarray((MASK_SIZE,), dtype=np.uint8)

    if game[ACTION_FLIP_SECOND]:
        mask[MASK_FLIP_SECOND_BELOW] = 1
        mask[MASK_FLIP_SECOND_RIGHT] = 1
    elif game[ACTION_DRAW_OR_TAKE]:
        mask[MASK_DRAW] = 1
        mask[MASK_TAKE] = 1
    elif game[ACTION_FLIP_OR_REPLACE]:
        for i in range(FINGER_COUNT):
            row, column = divmod(i, COLUMN_COUNT)
            if table[0, row, column, FINGER_HIDDEN]:
                mask[MASK_FLIP + i] = 1  # You both flip or replace
                mask[MASK_REPLACE + i] = 1
            elif not table[0, row, column, FINGER_CLEARED]:
                mask[MASK_REPLACE + i] = 1  # You can only replace
    elif game[ACTION_REPLACE]:
        for i in range(FINGER_COUNT):
            row, column = divmod(i, COLUMN_COUNT)
            if not table[0, row, column, FINGER_CLEARED]:
                mask[MASK_REPLACE + i] = 1  # You can only replace

    return mask
