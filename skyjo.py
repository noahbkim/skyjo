from __future__ import annotations

import math
import random
from collections.abc import Callable
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
MASK_REPLACE = MASK_FLIP + FINGER_COUNT
MASK_SIZE = MASK_REPLACE + FINGER_COUNT

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

Skyjo: TypeAlias = tuple[Game, Table, Deck, int, int, int | None]
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
    assert top is not None, "Tried to push top with no discard"

    game[GAME_TOP + card] = 1
    game[GAME_DISCARDS + top] += 1


def _get_action(game: Game) -> int | None:
    """Get the current action index."""

    for i in range(ACTION_SIZE):
        if game[GAME_ACTION + i]:
            return i

    return None


def _replace_action(game: Game, action: int) -> None:
    """Replace the current action with a new one."""

    for i in range(ACTION_SIZE):
        if game[GAME_ACTION + i]:
            game[GAME_ACTION + i] = 0
            break

    game[GAME_ACTION + action] = 1


def _rotate_scores(game: Game, players: int) -> None:
    """Rotate the player scores once left."""

    swap = game[GAME_SCORES]
    game[GAME_SCORES : GAME_SCORES + players - 1] = game[
        GAME_SCORES + 1 : GAME_SCORES + players
    ]
    game[GAME_SCORES + players - 1] = swap


def _rotate_table(table: Table, players: int) -> None:
    """Copy `table`, rotating all hands left by one."""

    swap = table[0].copy()
    table[0 : players - 1] = table[1:players]
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
        remaining = deck[i]
        if deck[i] > choice:
            return i
        choice -= remaining

    assert False, "fuck you"


def _remove_card(deck: Deck, card: int) -> None:
    """Remove `card` from `deck`."""

    if deck[card] == 0:
        raise ValueError(f"{deck!r} has no card {card} remaining")

    deck[card] -= 1


# MARK: Construction


def new(*, players: int) -> Skyjo:
    """Generate a fresh game with no visible discard."""

    game = np.ndarray((GAME_SIZE,), dtype=np.uint8)
    game.fill(0)
    game[GAME_ACTION + ACTION_FLIP_SECOND] = 1

    table_shape = (PLAYER_COUNT, ROW_COUNT, COLUMN_COUNT, FINGER_SIZE)
    table = np.ndarray(table_shape, dtype=np.uint8, order="C")
    table.fill(0)
    table[:players, :, :, FINGER_HIDDEN] = 1

    deck = np.ndarray((CARD_SIZE,), dtype=np.uint8)
    deck.fill(0)
    deck[:] = CARD_COUNTS

    return game, table, deck, players, 0, None


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

    return _get_top(skyjo[0])


def get_action(skyjo: Skyjo) -> int | None:
    """Get the current action index."""

    return _get_action(skyjo[0])


def get_finger(skyjo: Skyjo, row: int, column: int, player: int = 0) -> int:
    """Get the state of a given card in hand.

    This method returns the card index, not its value, so a return of
    0 corresponds to the -2 card. A return of 15 indicates the card is
    hidden and a return of 16 indicates the column has been cleared.
    """

    table = skyjo[1]

    for i in range(FINGER_SIZE):
        if table[player, row, column, i]:
            return i

    assert False, f"{skyjo!r} has no card at ({row}, {column})"


def get_is_visible(skyjo: Skyjo, player: int = 0) -> bool:
    """Whether the current player's board is completely revealed."""

    table = skyjo[1]

    return table[player, :, :, FINGER_HIDDEN].any()


def get_score(skyjo: Skyjo, player: int = 0) -> int:
    """Get the score of visible cards on a player's board."""

    table = skyjo[1]

    score = 0
    for row in range(ROW_COUNT):
        for column in range(COLUMN_COUNT):
            for card in range(CARD_SIZE):  # Skip CLEARED, score zero
                if table[player, row, column, card]:
                    score += card - 2
                    break

    return score


def get_winner(skyjo: Skyjo) -> int:
    """Get the index of the winner relative to the current table."""

    players = skyjo[3]

    return min(range(players), key=lambda i: get_score(skyjo, player=i))


def get_turn(skyjo: Skyjo) -> int:
    """Get the index of the player taking a turn, zero for the first."""

    return skyjo[4]


# MARK: Validation


def validate(skyjo: Skyjo) -> bool:
    """Validate the consistency of a `Skyjo` state.

    Always returns `True` for use with `assert`. Validation errors are
    raised internally.
    """

    game = skyjo[0]
    table = skyjo[1]
    deck = skyjo[2]
    players = skyjo[3]

    # Game
    assert game.shape == (GAME_SIZE,)
    assert np.sum(game[GAME_ACTION : GAME_ACTION + ACTION_SIZE]) == 1
    if game[GAME_ACTION + ACTION_FLIP_SECOND]:
        assert np.sum(game[GAME_TOP : GAME_TOP + CARD_SIZE]) == 0
        assert np.sum(game[GAME_DISCARDS : GAME_DISCARDS + CARD_SIZE]) == 0
    else:
        assert np.sum(game[GAME_TOP : GAME_TOP + CARD_SIZE]) == 1

    # Table
    assert table.shape == (PLAYER_COUNT, ROW_COUNT, COLUMN_COUNT, FINGER_SIZE)
    fingers = np.zeros((FINGER_SIZE,), dtype=np.uint8)
    for i in range(players):
        for row in range(ROW_COUNT):
            for column in range(COLUMN_COUNT):
                assert np.sum(table[i, row, column]) == 1
                fingers += table[i, row, column]
    assert np.sum(table[players:]) == 0
    assert np.sum(fingers) == players * FINGER_COUNT, f"{np.sum(fingers)=}"

    # Deck
    assert deck.shape == (CARD_SIZE,)
    card_top = game[GAME_TOP : GAME_TOP + CARD_SIZE]
    cards_dealt = fingers[:CARD_SIZE]
    cards_discarded = game[GAME_DISCARDS : GAME_DISCARDS + CARD_SIZE]

    assert ((deck + card_top + cards_dealt + cards_discarded) == CARD_COUNTS).all()

    return True


# MARK: Actions


def randomize(skyjo: Skyjo, rng: Random = random) -> Skyjo:
    """Return a copy of the `skyjo` with a random card selected.

    This method prepares the game simulation for an action that incurs
    a random event, such as drawing or revealing a card.
    """

    card = skyjo[5]

    if card is not None:
        raise ValueError("A random card has already been selected")

    card = _choose_card(skyjo[2], rng)

    return skyjo[0], skyjo[1], skyjo[2], skyjo[3], skyjo[4], card


def begin(skyjo: Skyjo) -> Skyjo:
    """Start a round once initial cards are revealed.

    This method reveals the first discard and sets action to draw or
    take for the first player in the turn order.
    """

    game = skyjo[0]
    table = skyjo[1]
    deck = skyjo[2]
    players = skyjo[3]
    player = skyjo[4]
    card = skyjo[5]

    assert _get_top(game) is None
    assert _get_action(game) in {
        ACTION_FLIP_SECOND,
        ACTION_FLIP_OR_REPLACE,
        ACTION_REPLACE,
    }

    if card is None:
        raise ValueError("Expected a randomly-drawn card")

    new_game = game.copy()
    _swap_top(new_game, card)
    _replace_action(new_game, ACTION_DRAW_OR_TAKE)

    new_deck = deck.copy()
    _remove_card(new_deck, card)

    return new_game, table, new_deck, players, player, None


def draw(skyjo: Skyjo) -> Skyjo:
    """Draw a random card from the deck, placing it in top position.

    This method constructs a copy of `skyjo` with a card randomly
    removed from its `deck` and placed at the top of its `game`.
    """

    game = skyjo[0]
    table = skyjo[1]
    deck = skyjo[2]
    players = skyjo[3]
    player = skyjo[4]
    card = skyjo[5]

    if card is None:
        raise ValueError("Expected a randomly-drawn card")

    new_game = game.copy()
    _push_top(new_game, card)
    _replace_action(new_game, ACTION_FLIP_OR_REPLACE)
    new_deck = deck.copy()
    _remove_card(new_deck, card)

    return new_game, table, new_deck, players, player, None


def take(skyjo: Skyjo) -> Skyjo:
    """Take the last discard.

    Since the discard is already stored at the top of the `game`, this
    method only has to update the action.
    """

    game = skyjo[0]
    table = skyjo[1]
    deck = skyjo[2]
    players = skyjo[3]
    player = skyjo[4]
    card = skyjo[5]

    new_game = game.copy()
    _replace_action(new_game, ACTION_REPLACE)

    return new_game, table, deck, players, player, card


def flip(
    skyjo: Skyjo,
    row: int,
    column: int,
    rotate: bool = True,
    turn: bool = True,  # Whether to reset action to DRAW_OR_TAKE
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
    card = skyjo[5]

    # Make sure this is a valid move
    if not table[0, row, column, FINGER_HIDDEN]:
        raise ValueError(f"{skyjo!r} cannot flip visible ({row}, {column})")

    if card is None:
        raise ValueError("Expected a randomly-drawn card")

    # No physical change to the top card, only semantic. What was the
    # draw card is now considered the discard. Rotate scores after.
    new_game = game.copy()
    if rotate:
        _rotate_scores(new_game, players)
    if turn:
        _replace_action(new_game, ACTION_DRAW_OR_TAKE)

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

    return new_game, new_table, new_deck, players, player + 1, None


def replace(skyjo: Skyjo, row: int, column: int) -> Skyjo:
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
    card = skyjo[5]

    # If the finger is currently hidden, we need to draw, but only if
    # `card` is not specified.
    if table[0, row, column, FINGER_HIDDEN]:
        finger = FINGER_HIDDEN
        if card is None:
            raise ValueError("Expected a randomly-drawn card")

    # Otherwise, ensure no `card` was specified and determine which
    # card is currently at the given coordinates.
    else:
        if card is not None:
            raise ValueError("Unexpected randomly-drawn card")
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
    _replace_action(new_game, ACTION_DRAW_OR_TAKE)

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

    return new_game, new_table, new_deck, players, player + 1, None


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
    mask.fill(0)

    if game[GAME_ACTION + ACTION_FLIP_SECOND]:
        mask[MASK_FLIP_SECOND_BELOW] = 1
        mask[MASK_FLIP_SECOND_RIGHT] = 1
    elif game[GAME_ACTION + ACTION_DRAW_OR_TAKE]:
        mask[MASK_DRAW] = 1
        mask[MASK_TAKE] = 1
    elif game[GAME_ACTION + ACTION_FLIP_OR_REPLACE]:
        for i in range(FINGER_COUNT):
            row, column = divmod(i, COLUMN_COUNT)
            if table[0, row, column, FINGER_HIDDEN]:
                mask[MASK_FLIP + i] = 1  # You both flip or replace
                mask[MASK_REPLACE + i] = 1
            elif not table[0, row, column, FINGER_CLEARED]:
                mask[MASK_REPLACE + i] = 1  # You can only replace
    elif game[GAME_ACTION + ACTION_REPLACE]:
        for i in range(FINGER_COUNT):
            row, column = divmod(i, COLUMN_COUNT)
            if not table[0, row, column, FINGER_CLEARED]:
                mask[MASK_REPLACE + i] = 1  # You can only replace
    else:
        raise ValueError(f"No action specified by state {skyjo!r}")

    return mask


# MARK: Selfplay


def greedy(skyjo: Skyjo) -> int:
    """Always draw from the pile and replace the next hidden card."""

    game = skyjo[0]
    table = skyjo[1]

    if game[GAME_ACTION + ACTION_FLIP_SECOND]:
        return MASK_FLIP_SECOND_BELOW
    if game[GAME_ACTION + ACTION_DRAW_OR_TAKE]:
        return MASK_DRAW
    if game[GAME_ACTION + ACTION_FLIP_OR_REPLACE]:
        for index in range(FINGER_COUNT):
            row, column = divmod(index, COLUMN_COUNT)
            if table[0, row, column, FINGER_HIDDEN]:
                return MASK_REPLACE + index
        raise ValueError("No card to replace!")

    raise ValueError(f"Unexpected action {get_action(skyjo)}!")


def selfplay(
    model: Callable[[Skyjo], int],
    *,
    players: int = 4,
    rng: Random = random,
) -> None:
    """Self-play a model until a game finishes."""

    skyjo = new(players=players)
    assert validate(skyjo)

    for _ in range(players):
        skyjo = randomize(skyjo, rng=rng)
        assert validate(skyjo)

        skyjo = flip(skyjo, 0, 0, rotate=False, turn=False)
        assert validate(skyjo)

        mask = actions(skyjo)
        choice = model(skyjo)
        assert mask[choice], f"{mask=}, {choice=}"

        skyjo = randomize(skyjo, rng=rng)
        if choice == MASK_FLIP_SECOND_BELOW:
            skyjo = flip(skyjo, 1, 0, rotate=True, turn=False)
        elif choice == MASK_FLIP_SECOND_BELOW:
            skyjo = flip(skyjo, 0, 1, rotate=True, turn=False)
        else:
            raise ValueError(f"Invalid action {choice!r}")
        assert validate(skyjo)

    skyjo = randomize(skyjo)
    skyjo = begin(skyjo)  # Reveal discard and set action
    assert validate(skyjo)

    countdown = None

    while countdown != 0:
        mask = actions(skyjo)
        choice = model(skyjo)
        assert mask[choice]

        if choice == MASK_DRAW:
            skyjo = randomize(skyjo, rng=rng)
            assert validate(skyjo)
            skyjo = draw(skyjo)
            assert validate(skyjo)
        elif choice == MASK_TAKE:
            skyjo = take(skyjo)
            assert validate(skyjo)
        elif MASK_FLIP <= choice < MASK_FLIP + FINGER_COUNT:
            row, column = divmod(choice - MASK_FLIP, COLUMN_COUNT)
            skyjo = randomize(skyjo, rng=rng)
            skyjo = flip(skyjo, row, column)
            assert validate(skyjo)
        elif MASK_REPLACE <= choice < MASK_REPLACE + FINGER_COUNT:
            row, column = divmod(choice - MASK_REPLACE, COLUMN_COUNT)
            if skyjo[1][0, row, column, FINGER_HIDDEN]:
                skyjo = randomize(skyjo, rng=rng)
            skyjo = replace(skyjo, row, column)
            assert validate(skyjo)
        else:
            raise ValueError(f"Invalid action {choice!r}")

        # Decrement endgame counter if set.
        if countdown is not None:
            countdown -= 1

        # Rotated, also this only needs to happen in the latter two ifs
        # but whatever.
        if get_is_visible(skyjo, player=players - 1):
            countdown = (players - 1) * 2  # Each player gets two decisions

    for player in range(players):
        for row in range(ROW_COUNT):
            for column in range(COLUMN_COUNT):
                if get_finger(skyjo, row, column, player=player) == FINGER_HIDDEN:
                    skyjo = randomize(skyjo, rng=rng)
                    skyjo = flip(skyjo, row, column, rotate=False, turn=False)
                    assert validate(skyjo)

    winner = (get_winner(skyjo) + get_turn(skyjo)) % players
    print(winner)
