from __future__ import annotations

import math
import random
from collections.abc import Callable
from typing import Iterable, Protocol, TypeAlias

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

# Action mask index cutoffs
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
GAME_LAST_REVEALED_TURNS = GAME_SCORES + PLAYER_COUNT
GAME_SIZE = GAME_LAST_REVEALED_TURNS + PLAYER_COUNT

NO_PROGRESS_TURN_THRESHOLD = 20

assert CARD_SIZE == CARD_P12 + 1
assert FINGER_COUNT == 12
assert GAME_DISCARDS == 15
assert GAME_ACTION == 30
assert GAME_SCORES == 34
assert GAME_SIZE == 50

Game: TypeAlias = np.ndarray[tuple[int], np.int16]
"""Top discard/drawn card, count of discards, player scores, and last revealed turns."""

Table: TypeAlias = np.ndarray[tuple[int, int, int, int], np.int16]
"""A tensor representing (player, row, column, card) tuples."""

Deck: TypeAlias = np.ndarray[tuple[int], np.int16]
"""Cards remaining in the deck in lieu of a random seed."""

Skyjo: TypeAlias = tuple[
    Game,
    Table,
    Deck,
    int,
    int,
    int | None,
    int | None,
]
"""A lightweight representation of a Skyjo game.

Values:
    game: Game
        non-board game state values like discard pile, scores, etc.
    table: Table
        board of cards for each player
    deck: Deck
        remaining card counts does not include top card
    players: int
        number of players
    turn count: int
        current turn count
    card: int | None 
        for use in random outcome
    countdown: int 
        number of cards remaining in the deck
"""

SkyjoAction: TypeAlias = int
"""Integer representing an action in the Skyjo game."""


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
    assert card is not None, "Tried to swap top with no discard"

    top = _pop_top(game)
    game[GAME_TOP + card] = 1
    return top


def _push_top(game: Game, card: int) -> None:
    """Draw a card or place a new discard on the pile."""
    assert card is not None, "Tried to add a None card top of discard"
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


def _rotate_last_revealed_turns(game: Game, players: int) -> None:
    """Rotate the last revealed turns once left."""

    swap = game[GAME_LAST_REVEALED_TURNS]
    game[GAME_LAST_REVEALED_TURNS : GAME_LAST_REVEALED_TURNS + players - 1] = game[
        GAME_LAST_REVEALED_TURNS + 1 : GAME_LAST_REVEALED_TURNS + players
    ]
    game[GAME_LAST_REVEALED_TURNS + players - 1] = swap


def _rotate_table(table: Table, players: int) -> None:
    """Copy `table`, rotating all hands left by one."""

    swap = table[0].copy()
    table[0 : players - 1] = table[1:players]
    table[players - 1] = swap


def _rotate_skyjo(skyjo: Skyjo) -> Skyjo:
    """Copy `skyjo`, rotating all hands left by one."""

    game, table, deck, players, current_player, card, countdown = skyjo
    new_game = game.copy()
    _rotate_scores(new_game, players)
    _rotate_last_revealed_turns(new_game, players)
    new_table = table.copy()
    _rotate_table(new_table, players)
    return (
        new_game,
        new_table,
        deck,
        players,
        current_player,
        card,
        countdown,
    )


def _clear_columns(game: Game, table: Table, column: int | None = None) -> int | None:
    """Clear any columns where all values match."""

    if column is None:
        columns = range(COLUMN_COUNT)
    else:
        columns = [column]

    for i in columns:
        for j in range(CARD_SIZE):  # Not finger size, skip hidden and cleared
            if table[0, 0, i, j] and table[0, 1, i, j] and table[0, 2, i, j]:
                table[0, :, i, j] = 0
                table[0, :, i, FINGER_CLEARED] = 1
                # Add cleared cards to discard pile
                _push_top(game, j)
                _push_top(game, j)
                _push_top(game, j)


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
    assert card is not None, "Tried to remove None card"
    if deck[card] == 0:
        raise ValueError(f"{deck!r} has no card {card} remaining")

    deck[card] -= 1


def _update_last_revealed_turns(game: Game, turn: int) -> None:
    """Update the last revealed turns."""
    game[GAME_LAST_REVEALED_TURNS] = turn


def _update_countdown(
    countdown: int | None,
    table: Table,
    players: int,
    last_revealed_turn: int,
    current_turn: int,
) -> int | None:
    """Decrement if already set, otherwise check whether round is ending
    (i.e. all cards are revealed). If, so set countdown to
    (number of players - 1)* 2 since each other player gets two decisions.

    Should be called after an action has been applied.
    """

    if countdown is None and _player_table_is_visible(table, player=players - 1):
        return (players - 1) * 2
    if (
        countdown is None
        and (current_turn - last_revealed_turn) // players >= NO_PROGRESS_TURN_THRESHOLD
    ):
        return (players - 1) * 2
    return _decrement_countdown(countdown)


def _decrement_countdown(countdown: int | None) -> int | None:
    """Decrement the countdown."""
    if countdown is None:
        return None
    return countdown - 1


def _player_table_is_visible(table: Table, player: int) -> bool:
    """Whether the player's table is completely revealed."""
    return not table[player, :, :, FINGER_HIDDEN].any()


# MARK: Construction


def new(*, players: int) -> Skyjo:
    """Generate a fresh game with no visible discard."""

    game = np.ndarray((GAME_SIZE,), dtype=np.int16)
    game.fill(0)
    game[GAME_ACTION + ACTION_FLIP_SECOND] = 1

    table_shape = (PLAYER_COUNT, ROW_COUNT, COLUMN_COUNT, FINGER_SIZE)
    table = np.ndarray(table_shape, dtype=np.int16, order="C")
    table.fill(0)
    table[:players, :, :, FINGER_HIDDEN] = 1

    deck = np.ndarray((CARD_SIZE,), dtype=np.int16)
    deck.fill(0)
    deck[:] = CARD_COUNTS

    return game, table, deck, players, 0, None, None


# MARK: Convenience


def get_table(skyjo: Skyjo) -> Table:
    """Get the board portion of the game state."""
    return skyjo[1][: get_player_count(skyjo)]


def get_board(skyjo: Skyjo) -> Table:
    """Get the board portion of the game state."""
    return get_table(skyjo)


def get_spatial_input(skyjo: Skyjo) -> Table:
    """Get the board portion of the game state."""
    return skyjo[1][: get_player_count(skyjo)]


def get_game(skyjo: Skyjo) -> Game:
    """Get the game portion of the game state."""
    return skyjo[0]


def get_non_spatial_input(skyjo: Skyjo) -> Game:
    """Get the non-spatial (i.e. non-board) game state values."""
    return skyjo[0]


def get_deck(skyjo: Skyjo) -> Deck:
    """Get the deck portion of the game state."""
    return skyjo[2]


def get_countdown(skyjo: Skyjo) -> int | None:
    """Get the number of cards left in the deck."""
    return skyjo[6]


def get_turn_count(skyjo: Skyjo) -> int:
    """Get the current turn count."""
    return skyjo[4]


def get_draw_count(skyjo: Skyjo) -> int:
    """Get the number of cards left in the deck."""

    return np.sum(skyjo[2])


def get_last_revealed_turns(skyjo: Skyjo) -> np.ndarray[tuple[int], np.int8]:
    """Get the last turn a card was revealed."""
    return get_game(skyjo)[
        GAME_LAST_REVEALED_TURNS : GAME_LAST_REVEALED_TURNS + get_player_count(skyjo)
    ]


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


def get_facedown_count(skyjo: Skyjo, player: int = 0) -> int:
    """Get the number of face-down cards on a player's board."""

    table = skyjo[1]
    return np.sum(table[player, :, :, FINGER_HIDDEN]).item()


def get_is_visible(skyjo: Skyjo, player: int = 0) -> bool:
    """Whether the current player's board is completely revealed."""

    table = skyjo[1]

    return _player_table_is_visible(table, player)


def get_score(skyjo: Skyjo, player: int = 0) -> int:
    """Get the score of visible cards on a player's board."""

    table = skyjo[1].astype(np.int16)

    return (
        (
            # get all card index of card ignoring cleared and face-down
            np.argwhere(table[player, :, :, :CARD_SIZE] == 1)[:, 2]
            - 2  # since card index starts with -2
        )
        .sum()
        .item()
    )


def get_round_scores(
    skyjo: Skyjo, round_ending_player: int = 0
) -> np.ndarray[tuple[int], np.int16]:
    """Get the scores of all players for the current round.

        This method accounts for the round ending player's score being doubled,
    if they are not the lowest round score winner.

    Round ending player parameter is relative to current perspective."""
    players = skyjo[3]
    base_scores = np.array(
        [get_score(skyjo, player=i) for i in range(players)], dtype=np.int16
    )
    turn = get_turn(skyjo)
    for player in range(players):
        if player == round_ending_player:
            round_ender_score = base_scores[round_ending_player]
            if round_ender_score >= min(
                np.delete(base_scores, round_ending_player)
            ) or (
                (turn - get_last_revealed_turns(skyjo)[round_ending_player]) // players
                >= NO_PROGRESS_TURN_THRESHOLD
                + 1  # +1 because after last action turn is incremented again
            ):
                base_scores[round_ending_player] *= 2
        else:
            if (
                turn - get_last_revealed_turns(skyjo)[player]
            ) // players >= NO_PROGRESS_TURN_THRESHOLD:
                base_scores[player] *= 2
    return base_scores


def get_fixed_perspective_round_scores(
    skyjo: Skyjo,
) -> np.ndarray[tuple[int], np.int16]:
    """Get the scores of all players for the current round from a fixed perspective."""
    round_scores = get_round_scores(skyjo)
    return np.roll(round_scores, get_player(skyjo))


def get_winner(skyjo: Skyjo, round_ending_player: int = 0) -> int:
    """Get the index of the winner relative to the current table.
    Round ending player is relative to current perspective."""

    scores = get_round_scores(skyjo, round_ending_player)
    return np.argmin(scores).item()


def get_fixed_perspective_winner(skyjo: Skyjo) -> int:
    """Get the index of the winner not from perspective of current player."""

    players = skyjo[3]
    winner = (get_winner(skyjo) + get_turn(skyjo)) % players
    return winner


def get_turn(skyjo: Skyjo) -> int:
    """Get current turn count."""

    return skyjo[4]


def get_player(skyjo: Skyjo) -> int:
    """Get the index of the player taking a turn, zero for the first."""

    return skyjo[4] % skyjo[3]


def get_player_count(skyjo: Skyjo) -> int:
    """Get the number of players in the game."""
    return skyjo[3]


def get_game_over(skyjo: Skyjo) -> bool:
    """Whether the game is over.
    If there are any face-down cards game is not over"""
    # table = skyjo[1]
    # return not table[:, :, :, FINGER_HIDDEN].any()
    return skyjo[6] == 0


def hash_skyjo(skyjo: Skyjo) -> int:
    """Hash the `skyjo` state.

    NOTE: This is  tobytes() can return the same hash for arrays of different shape.
    However, this shouldn't be an issue for this specific game since the representation is
    fixed shape.
    """
    return hash(
        (
            skyjo[0].tobytes(),
            skyjo[1].tobytes(),
            skyjo[2].tobytes(),
            skyjo[3],
            skyjo[4],
            skyjo[5],
            skyjo[6],
        )
    )


# MARK: Debug


def get_action_name(action: SkyjoAction) -> str:
    if action == MASK_FLIP_SECOND_RIGHT:
        return "FLIP_SECOND_RIGHT"
    elif action == MASK_FLIP_SECOND_BELOW:
        return "FLIP_SECOND_BELOW"
    elif action == MASK_DRAW:
        return "DRAW"
    elif action == MASK_TAKE:
        return "TAKE"
    elif MASK_FLIP <= action < MASK_FLIP + FINGER_COUNT:
        row, column = divmod(action - MASK_FLIP, COLUMN_COUNT)
        return f"FLIP row: {row} column: {column}"
    elif MASK_REPLACE <= action < MASK_REPLACE + FINGER_COUNT:
        row, column = divmod(action - MASK_REPLACE, COLUMN_COUNT)
        return f"REPLACE row: {row} column: {column}"
    else:
        return f"UNKNOWN ({action})"


def visualize_table(table: Table, player: int):
    """Visualize the current state of the table."""
    player_table_str = "+--" * (COLUMN_COUNT) + "+\n"
    for row in range(ROW_COUNT):
        row_str = "|"
        for column in range(COLUMN_COUNT):
            cell_str = ""
            if table[player, row, column, FINGER_HIDDEN] == 1:
                cell_str = ""
            elif table[player, row, column, FINGER_CLEARED] == 1:
                cell_str = "X"
            else:
                cell_str = str(
                    np.argwhere(table[player, row, column, :CARD_SIZE] == 1).item() - 2
                )
            if len(cell_str) < 2:
                cell_str = " " * (2 - len(cell_str)) + cell_str
            row_str += f"{cell_str}|"
        player_table_str += row_str + "\n"
        player_table_str += "+--" * (COLUMN_COUNT) + "+\n"
    return player_table_str


def visualize_state(skyjo: Skyjo):
    """Visualize the current state of the game."""
    game = skyjo[0]
    table = skyjo[1]
    deck = skyjo[2]
    players = skyjo[3]

    curr_player = get_player(skyjo)
    state_str = f"current player: {curr_player} card: {get_top(skyjo) - 2 if get_top(skyjo) is not None else None}\n"
    state_str += f"turn: {get_turn(skyjo)} countdown: {skyjo[6]} last_revealed_turns: {get_last_revealed_turns(skyjo)}\n"
    state_str += f"Actions: {game[GAME_ACTION : GAME_ACTION + ACTION_SIZE]}\n\n"
    for i in range(players):
        state_str += f"Player {(curr_player + i) % players}\n"
        state_str += visualize_table(table, i)
    return state_str


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
    fingers = np.zeros((FINGER_SIZE,), dtype=np.int16)
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

    assert ((deck + card_top + cards_dealt + cards_discarded) == CARD_COUNTS).all(), (
        f"cards disappeared: {deck=}, {card_top=}, {cards_dealt=}, {cards_discarded=}"
    )

    # No card has been revealed for too long
    assert (
        get_turn(skyjo) - get_last_revealed_turns(skyjo)[0]
    ) // players <= NO_PROGRESS_TURN_THRESHOLD or get_countdown(skyjo) == 0, (
        f"A card has not been revealed for too long: {get_turn(skyjo)=}, {get_last_revealed_turns(skyjo)=}, {get_countdown(skyjo)=}"
        f"{(get_turn(skyjo) - get_last_revealed_turns(skyjo)[0]) // players=}"
    )

    return True


# MARK: Actions


def randomize(skyjo: Skyjo, rng: Random = random) -> Skyjo:
    """Return a copy of the `skyjo` with a random card selected.

    This method prepares the game simulation for an action that incurs
    a random event, such as drawing or revealing a card.
    """

    game = skyjo[0]
    deck = skyjo[2]
    card = skyjo[5]

    if card is not None:
        # raise ValueError("A random card has already been selected")
        return skyjo

    # Deck is empty, reset with discarded cards
    if not deck.any():
        deck = game[GAME_DISCARDS : GAME_DISCARDS + CARD_SIZE]
        game = game.copy()
        game[GAME_DISCARDS : GAME_DISCARDS + CARD_SIZE] = 0
    card = _choose_card(deck, rng)

    return game, skyjo[1], deck, skyjo[3], skyjo[4], card, skyjo[6]


def preordain(skyjo: Skyjo, card: int) -> Skyjo:
    """Sets next 'random' next card to specified card."""
    assert 0 <= card < CARD_SIZE, (
        f"Not a valid card. Valid cards are between [0, {CARD_SIZE}), but got {card}"
    )

    return (
        skyjo[0],
        skyjo[1],
        skyjo[2],
        skyjo[3],
        skyjo[4],
        card,
        skyjo[6],
    )


def begin(skyjo: Skyjo) -> Skyjo:
    """Start a round once initial cards are revealed.

    This method reveals the first discard and sets action to draw or
    take for the first player in the turn order.
    """

    game = skyjo[0]
    table = skyjo[1]
    deck = skyjo[2]
    players = skyjo[3]
    turn = skyjo[4]
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

    return new_game, table, new_deck, players, turn, None, None


def draw(skyjo: Skyjo) -> Skyjo:
    """Draw a random card from the deck, placing it in top position.

    This method constructs a copy of `skyjo` with a card randomly
    removed from its `deck` and placed at the top of its `game`.
    """

    game = skyjo[0]
    table = skyjo[1]
    deck = skyjo[2]
    players = skyjo[3]
    turn = skyjo[4]
    card = skyjo[5]
    countdown = skyjo[6]
    if card is None:
        raise ValueError("Expected a randomly-drawn card")

    new_game = game.copy()
    _push_top(new_game, card)
    _replace_action(new_game, ACTION_FLIP_OR_REPLACE)
    new_deck = deck.copy()
    _remove_card(new_deck, card)
    countdown = _decrement_countdown(countdown)
    return (
        new_game,
        table,
        new_deck,
        players,
        turn,
        None,
        countdown,
    )


def take(skyjo: Skyjo) -> Skyjo:
    """Take the last discard.

    Since the discard is already stored at the top of the `game`, this
    method only has to update the action.
    """

    game = skyjo[0]
    table = skyjo[1]
    deck = skyjo[2]
    players = skyjo[3]
    turn = skyjo[4]
    card = skyjo[5]
    countdown = skyjo[6]
    new_game = game.copy()
    _replace_action(new_game, ACTION_REPLACE)
    new_countdown = _decrement_countdown(countdown)

    return new_game, table, deck, players, turn, card, new_countdown


def flip(
    skyjo: Skyjo,
    row: int,
    column: int,
    rotate: bool = True,
    set_draw_or_take_action: bool = True,  # Whether to reset action to DRAW_OR_TAKE
    is_turn: bool = True,
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
    turn = skyjo[4]
    card = skyjo[5]
    countdown = skyjo[6]
    # Make sure this is a valid move
    if not table[0, row, column, FINGER_HIDDEN]:
        raise ValueError(f"{skyjo!r} cannot flip visible ({row}, {column})")

    if card is None:
        raise ValueError("Expected a randomly-drawn card")

    # No physical change to the top card, only semantic. What was the
    # draw card is now considered the discard. Rotate scores after.
    new_game = game.copy()
    if is_turn:
        _update_last_revealed_turns(new_game, turn)

    if rotate:
        _rotate_scores(new_game, players)
        _rotate_last_revealed_turns(new_game, players)

    if set_draw_or_take_action:
        _replace_action(new_game, ACTION_DRAW_OR_TAKE)

    # Copy the table, flipping our card and rotating hands.
    new_table = table.copy()
    new_table[0, row, column, FINGER_HIDDEN] = 0
    new_table[0, row, column, card] = 1
    _clear_columns(new_game, new_table, column)
    if rotate:
        _rotate_table(new_table, players)

    if is_turn:
        countdown = _update_countdown(countdown, new_table, players, turn, turn)
        turn = turn + 1

    # Copy the deck, removing the card we chose.
    new_deck = deck.copy()
    _remove_card(new_deck, card)

    return (new_game, new_table, new_deck, players, turn, None, countdown)


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
    turn = skyjo[4]
    card = skyjo[5]
    countdown = skyjo[6]
    # If the finger is currently hidden, we need to draw, but only if
    # `card` is not specified.
    new_game = game.copy()
    if table[0, row, column, FINGER_HIDDEN]:
        finger = FINGER_HIDDEN
        if card is None:
            raise ValueError("Expected a randomly-drawn card")
        last_revealed_turn = turn
        _update_last_revealed_turns(new_game, turn)
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
    last_revealed_turn = get_last_revealed_turns(skyjo)[0]

    # Replace the current discard with `card`
    if card is not None:
        top = _swap_top(new_game, card)
    else:
        top = _swap_top(new_game, finger)
    _rotate_scores(new_game, players)
    _rotate_last_revealed_turns(new_game, players)
    _replace_action(new_game, ACTION_DRAW_OR_TAKE)

    # Clear the current card in our hand and replace it with top, i.e.
    # the draw or last discard depending on our choice. Rotate after.
    new_table = table.copy()
    new_table[0, row, column, finger] = 0
    new_table[0, row, column, top] = 1
    _clear_columns(new_game, new_table, column)
    _rotate_table(new_table, players)

    # Remove the card from the deck for the next iteration.
    new_deck = deck.copy()
    if card is not None:
        _remove_card(new_deck, card)
    countdown = _update_countdown(
        countdown, new_table, players, last_revealed_turn, turn
    )

    return (new_game, new_table, new_deck, players, turn + 1, None, countdown)


# MARK: Learning


def actions(skyjo: Skyjo) -> np.ndarray[tuple[int], np.int16]:
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

    mask = np.ndarray((MASK_SIZE,), dtype=np.int16)
    mask.fill(0)
    game = skyjo[0]
    table = skyjo[1]
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
                mask[MASK_FLIP + i] = mask[MASK_REPLACE + i] = (
                    1  # You can both flip and replace
                )
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


def get_actions(skyjo: Skyjo) -> Iterable[SkyjoAction]:
    """List of all possible actions."""
    mask = actions(skyjo)
    return np.argwhere(mask == 1).squeeze()


def is_action_random(action: SkyjoAction, skyjo: Skyjo) -> bool:
    """Whether the action involves a random outcome."""
    table, players, countdown = (
        skyjo[1],
        skyjo[3],
        skyjo[6],
    )
    # last action before end round
    # end of round reveals of facedown cards which will reveal all
    if (
        countdown is not None
        and countdown == 1
        and np.any(table[:, :, :, FINGER_HIDDEN])
    ):
        return True
    if action in {MASK_FLIP_SECOND_BELOW, MASK_FLIP_SECOND_RIGHT}:
        return True
    if action == MASK_DRAW:
        return True
    if action == MASK_TAKE:
        return False
    # TAKE
    if MASK_FLIP <= action < MASK_FLIP + FINGER_COUNT:
        return True
    row, column = divmod(action - MASK_REPLACE, COLUMN_COUNT)
    # If replacing doesn't reveal a card and NO PROGRESS_TURN_THRESHOLD will be reached,
    # then game will end and thus there will be random outcomes from flipping.
    if not table[0, row, column, FINGER_HIDDEN] <= get_turn(skyjo) + 1:
        return True
    return bool(table[0, row, column, FINGER_HIDDEN])


def quick_finish_action(skyjo: Skyjo) -> int:
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


def random_valid_action(skyjo: Skyjo) -> int:
    return np.random.choice(
        len(actions(skyjo)), p=actions(skyjo) / np.sum(actions(skyjo))
    )


# MARK: Selfplay


def start_round(skyjo: Skyjo, rng: Random = random) -> Skyjo:
    """Take skyjo state an start round by flipping first card for each player."""
    players = skyjo[3]
    for _ in range(players):
        skyjo = randomize(skyjo, rng=rng)
        assert validate(skyjo)
        skyjo = flip(
            skyjo, 0, 0, rotate=True, set_draw_or_take_action=False, is_turn=False
        )
        assert validate(skyjo)
    return skyjo


def end_round(skyjo: Skyjo, rng: Random = random) -> Skyjo:
    """End the round by flipping all hidden cards."""
    assert get_countdown(skyjo) == 0, "Game isn't over we can't end the round"
    players = skyjo[3]
    for _ in range(players):
        for row in range(ROW_COUNT):
            for column in range(COLUMN_COUNT):
                if get_finger(skyjo, row, column, player=0) == FINGER_HIDDEN:
                    skyjo = randomize(skyjo, rng=rng)
                    skyjo = flip(
                        skyjo,
                        row,
                        column,
                        rotate=False,
                        set_draw_or_take_action=False,
                        is_turn=False,
                    )
                    assert validate(skyjo)
        skyjo = _rotate_skyjo(skyjo)
    return (
        skyjo[0],
        skyjo[1],
        skyjo[2],
        skyjo[3],
        skyjo[4],
        skyjo[5],
        0,
    )


def apply_action(skyjo: Skyjo, action: SkyjoAction, rng: Random = random) -> Skyjo:
    players, countdown = get_player_count(skyjo), get_countdown(skyjo)
    # Apply action to skyjo
    if action == MASK_FLIP_SECOND_BELOW:
        skyjo = randomize(skyjo, rng=rng)
        assert validate(skyjo)
        skyjo = flip(
            skyjo, 1, 0, rotate=True, set_draw_or_take_action=False, is_turn=True
        )
        assert validate(skyjo)
        # All players have flipped initial cards, start round
        if skyjo[1][:, :, :, :CARD_SIZE].sum().item() == players * 2:
            skyjo = randomize(skyjo, rng=rng)
            skyjo = begin(skyjo)
    elif action == MASK_FLIP_SECOND_RIGHT:
        skyjo = randomize(skyjo, rng=rng)
        assert validate(skyjo)
        skyjo = flip(
            skyjo, 0, 1, rotate=True, set_draw_or_take_action=False, is_turn=True
        )
        assert validate(skyjo)
        # All players have flipped initial cards, start round
        if skyjo[1][:, :, :, :CARD_SIZE].sum().item() == players * 2:
            skyjo = randomize(skyjo, rng=rng)
            skyjo = begin(skyjo)
    elif action == MASK_DRAW:
        skyjo = randomize(skyjo, rng=rng)
        assert validate(skyjo)
        skyjo = draw(skyjo)
        assert validate(skyjo)
    elif action == MASK_TAKE:
        skyjo = take(skyjo)
        assert validate(skyjo)
    elif MASK_FLIP <= action < MASK_FLIP + FINGER_COUNT:
        row, column = divmod(action - MASK_FLIP, COLUMN_COUNT)
        skyjo = randomize(skyjo, rng=rng)
        assert validate(skyjo)
        skyjo = flip(skyjo, row, column)
        assert validate(skyjo)

    elif MASK_REPLACE <= action < MASK_REPLACE + FINGER_COUNT:
        row, column = divmod(action - MASK_REPLACE, COLUMN_COUNT)
        if skyjo[1][0, row, column, FINGER_HIDDEN]:
            skyjo = randomize(skyjo, rng=rng)
            assert validate(skyjo)

        skyjo = replace(skyjo, row, column)
        assert validate(skyjo)

    else:
        raise ValueError(f"Invalid action {action!r}")

    countdown = skyjo[6]
    if countdown == 0:
        skyjo = end_round(skyjo, rng=rng)
        assert validate(skyjo)

    assert validate(skyjo)
    return skyjo


def selfplay(
    model: Callable[[Skyjo], int],
    *,
    players: int = 4,
    rng: Random = random,
) -> None:
    """Self-play a model until a game finishes."""

    skyjo = new(players=players)
    assert validate(skyjo)
    skyjo = start_round(skyjo, rng=rng)
    countdown = skyjo[6]
    while countdown != 0:
        mask = actions(skyjo)
        choice = model(skyjo)
        assert mask[choice]
        skyjo = apply_action(skyjo, choice, rng=rng)
        countdown = skyjo[6]
    winner = get_fixed_perspective_winner(skyjo)
    print(winner)
    print(get_fixed_perspective_round_scores(skyjo))


if __name__ == "__main__":
    selfplay(random_valid_action, players=2)
