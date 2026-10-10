from __future__ import annotations

import dataclasses
import math
import random
from collections.abc import Iterable
from typing import Protocol

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
# Cumulative totals before the current round, in current-player order.
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

type Game = np.ndarray[tuple[int], np.int16]
"""Top discard/drawn card, count of discards, player scores, and last revealed turns."""

type Table = np.ndarray[tuple[int, int, int, int], np.int16]
"""A tensor representing (player, row, column, card) tuples."""

type Deck = np.ndarray[tuple[int], np.int16]
"""Cards remaining in the deck in lieu of a random seed."""


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class Skyjo:
    """Current-player-oriented game state with copy-on-write NumPy storage.

    ``table`` retains eight-player padding; observations slice active players.
    ``pending_card`` selects the next random outcome without consuming it.
    ``countdown`` counts actions remaining in the final turns of a round.
    """

    game: Game
    table: Table
    deck: Deck
    players: int
    turn: int
    pending_card: int | None = None
    countdown: int | None = None


type SkyjoAction = int
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

    game, table, players = skyjo.game, skyjo.table, skyjo.players
    new_game = game.copy()
    _rotate_scores(new_game, players)
    _rotate_last_revealed_turns(new_game, players)
    new_table = table.copy()
    _rotate_table(new_table, players)
    return dataclasses.replace(skyjo, game=new_game, table=new_table)


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

    raise ValueError("Cannot select a card from an empty deck")


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
        and (current_turn - last_revealed_turn) // players > NO_PROGRESS_TURN_THRESHOLD
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


def new(*, players: int, top: int | None = None, rng: Random = random) -> Skyjo:
    """Generate the first round with zero scores and an initial visible discard.

    The discard is removed from the draw deck immediately. Pass ``top`` to
    choose its card index deterministically; otherwise it is sampled with
    ``rng``.
    """

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
    if top is None:
        top = _choose_card(deck, rng)

    _swap_top(game, top)
    _remove_card(deck, top)
    return Skyjo(game=game, table=table, deck=deck, players=players, turn=0)


# MARK: Convenience


def get_table(skyjo: Skyjo) -> Table:
    """Get the board portion of the game state."""
    return skyjo.table[: get_player_count(skyjo)]


def get_board(skyjo: Skyjo) -> Table:
    """Get the board portion of the game state."""
    return get_table(skyjo)


def get_game(skyjo: Skyjo) -> Game:
    """Get the game portion of the game state."""
    return skyjo.game


def get_deck(skyjo: Skyjo) -> Deck:
    """Get the deck portion of the game state."""
    return skyjo.deck


def get_countdown(skyjo: Skyjo) -> int | None:
    """Get remaining final-turn actions, or None before the round is ending."""
    return skyjo.countdown


def get_turn_count(skyjo: Skyjo) -> int:
    """Get the current turn count."""
    return skyjo.turn


def get_draw_count(skyjo: Skyjo) -> int:
    """Get the number of cards left in the deck."""

    return np.sum(skyjo.deck)


def get_last_revealed_turns(skyjo: Skyjo) -> np.ndarray[tuple[int], np.int8]:
    """Get the last turn a card was revealed."""
    return get_game(skyjo)[
        GAME_LAST_REVEALED_TURNS : GAME_LAST_REVEALED_TURNS + get_player_count(skyjo)
    ]


def get_discard_count(skyjo: Skyjo) -> int:
    """Get the number of discarded cards excluding the visible discard."""

    game = skyjo.game

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

    return _get_top(skyjo.game)


def get_action(skyjo: Skyjo) -> int | None:
    """Get the current action index."""

    return _get_action(skyjo.game)


def get_finger(skyjo: Skyjo, row: int, column: int, player: int = 0) -> int:
    """Get the state of a given card in hand.

    This method returns the card index, not its value, so a return of
    0 corresponds to the -2 card. A return of 15 indicates the card is
    hidden and a return of 16 indicates the column has been cleared.
    """

    table = skyjo.table

    for i in range(FINGER_SIZE):
        if table[player, row, column, i]:
            return i

    assert False, f"{skyjo!r} has no card at ({row}, {column})"


def get_facedown_count(skyjo: Skyjo, player: int = 0) -> int:
    """Get the number of face-down cards on a player's board."""

    table = skyjo.table
    return np.sum(table[player, :, :, FINGER_HIDDEN]).item()


def get_is_visible(skyjo: Skyjo, player: int = 0) -> bool:
    """Whether the current player's board is completely revealed."""

    table = skyjo.table

    return _player_table_is_visible(table, player)


def get_score(skyjo: Skyjo, player: int = 0) -> int:
    """Get the score of visible cards on a player's board."""

    table = skyjo.table.astype(np.int16)

    return (
        (
            # get all card index of card ignoring cleared and face-down
            np.argwhere(table[player, :, :, :CARD_SIZE] == 1)[:, 2]
            - 2  # since card index starts with -2
        )
        .sum()
        .item()
    )


def get_round_score_components(
    skyjo: Skyjo, round_ending_player: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw board points and explicit scoring multipliers in player order.

    The ending-player index is relative to the current perspective. Keep the
    scoring order used by the engine, including ties and no-progress penalties.
    Flags remain meaningful when raw points are zero or negative.
    """
    players = skyjo.players
    base_scores = np.array(
        [get_score(skyjo, player=i) for i in range(players)], dtype=np.int16
    )
    raw_scores = base_scores.copy()
    doubled = np.zeros(players, dtype=np.bool_)
    turn = get_turn(skyjo)
    for player in range(players):
        if player == round_ending_player:
            round_ender_score = base_scores[round_ending_player]
            if round_ender_score >= min(
                np.delete(base_scores, round_ending_player)
            ) or (
                (turn - get_last_revealed_turns(skyjo)[round_ending_player]) // players
                > NO_PROGRESS_TURN_THRESHOLD
            ):
                doubled[round_ending_player] = True
                base_scores[round_ending_player] *= 2
        else:
            if (
                turn - get_last_revealed_turns(skyjo)[player]
            ) // players > NO_PROGRESS_TURN_THRESHOLD:
                doubled[player] = True
                base_scores[player] *= 2
    return raw_scores, doubled


def get_round_scores(
    skyjo: Skyjo, round_ending_player: int = 0
) -> np.ndarray[tuple[int], np.int16]:
    raw, doubled = get_round_score_components(skyjo, round_ending_player)
    return raw * (1 + doubled.astype(np.int16))


def get_fixed_perspective_round_scores(
    skyjo: Skyjo,
) -> np.ndarray[tuple[int], np.int16]:
    """Get the scores of all players for the current round from a fixed perspective."""
    round_scores = get_round_scores(skyjo)
    return np.roll(round_scores, get_player(skyjo))


def get_game_scores(skyjo: Skyjo) -> np.ndarray[tuple[int], np.int16]:
    """Get cumulative scores, including this round only once it is complete.

    Scores are in current-player order. The returned array is independent of
    the state; the stored totals always exclude the current round.
    """
    scores = get_game(skyjo)[GAME_SCORES : GAME_SCORES + get_player_count(skyjo)].copy()
    if get_round_over(skyjo):
        scores += get_round_scores(skyjo)
    return scores


def get_fixed_perspective_game_scores(
    skyjo: Skyjo,
) -> np.ndarray[tuple[int], np.int16]:
    """Get cumulative scores in fixed player order."""
    return np.roll(get_game_scores(skyjo), get_player(skyjo))


def get_winner(skyjo: Skyjo, round_ending_player: int = 0) -> int:
    """Get the index of the round winner relative to the current table.
    Round ending player is relative to current perspective."""

    scores = get_round_scores(skyjo, round_ending_player)
    return np.argmin(scores).item()


def get_fixed_perspective_winner(skyjo: Skyjo) -> int:
    """Get the round winner's fixed player index."""

    players = skyjo.players
    winner = (get_winner(skyjo) + get_turn(skyjo)) % players
    return winner


def get_cleared_columns(skyjo: Skyjo) -> np.ndarray[tuple[int, int], np.int8]:
    """Get the columns that have been cleared."""
    return skyjo.table[: get_player_count(skyjo), 0, :, FINGER_CLEARED]


def get_fixed_perspective_cleared_columns(
    skyjo: Skyjo,
) -> np.ndarray[tuple[int, int], np.int8]:
    """Get the columns that have been cleared from a fixed perspective."""
    return np.roll(get_cleared_columns(skyjo), get_player(skyjo), axis=0)


def get_turn(skyjo: Skyjo) -> int:
    """Get current turn count."""

    return skyjo.turn


def get_player(skyjo: Skyjo) -> int:
    """Get the index of the player taking a turn, zero for the first."""

    return skyjo.turn % skyjo.players


def get_player_count(skyjo: Skyjo) -> int:
    """Get the number of players in the game."""
    return skyjo.players


def get_round_over(skyjo: Skyjo) -> bool:
    """Whether the round's final-turn countdown has finished."""
    return skyjo.countdown == 0


def get_round_about_to_end(skyjo: Skyjo) -> bool:
    """Whether the next action completes the round."""
    return skyjo.countdown == 1


def get_game_over(skyjo: Skyjo) -> bool:
    """Whether a completed round brings any cumulative score to at least 100."""
    return get_round_over(skyjo) and bool(np.any(get_game_scores(skyjo) >= 100))


def hash_skyjo(skyjo: Skyjo) -> int:
    """Hash the `skyjo` state.

    NOTE: This is  tobytes() can return the same hash for arrays of different shape.
    However, this shouldn't be an issue for this specific game since the representation is
    fixed shape.
    """
    return hash(
        (
            skyjo.game.tobytes(),
            skyjo.table.tobytes(),
            skyjo.deck.tobytes(),
            skyjo.players,
            skyjo.turn,
            skyjo.pending_card,
            skyjo.countdown,
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
    game = skyjo.game
    table = skyjo.table
    players = skyjo.players

    curr_player = get_player(skyjo)
    state_str = f"current player: {curr_player} card: {get_top(skyjo) - 2 if get_top(skyjo) is not None else None}\n"
    state_str += f"turn: {get_turn(skyjo)} countdown: {skyjo.countdown} last_revealed_turns: {get_last_revealed_turns(skyjo)}\n"
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
    game = skyjo.game
    table = skyjo.table
    deck = skyjo.deck
    players = skyjo.players

    # Game
    assert game.shape == (GAME_SIZE,)
    assert np.sum(game[GAME_ACTION : GAME_ACTION + ACTION_SIZE]) == 1

    if not game[GAME_ACTION + ACTION_FLIP_SECOND]:
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
    ) // players <= NO_PROGRESS_TURN_THRESHOLD + 1 or get_countdown(
        skyjo
    ) is not None, (
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

    skyjo = prepare_draw_pile(skyjo)
    if skyjo.pending_card is not None:
        return skyjo
    return preordain(skyjo, _choose_card(skyjo.deck, rng))


def prepare_draw_pile(skyjo: Skyjo) -> Skyjo:
    """Recycle discards when needed, leaving the top card and input untouched."""
    if skyjo.deck.any():
        return skyjo
    game = skyjo.game.copy()
    deck = game[GAME_DISCARDS : GAME_DISCARDS + CARD_SIZE].copy()
    if not deck.any():
        raise ValueError("Cannot draw: deck and recyclable discards are empty")
    game[GAME_DISCARDS : GAME_DISCARDS + CARD_SIZE] = 0
    return dataclasses.replace(skyjo, game=game, deck=deck)


def preordain(skyjo: Skyjo, card: int) -> Skyjo:
    """Sets next 'random' next card to specified card."""
    assert 0 <= card < CARD_SIZE, (
        f"Not a valid card. Valid cards are between [0, {CARD_SIZE}), but got {card}"
    )

    return dataclasses.replace(skyjo, pending_card=card)


def begin(skyjo: Skyjo) -> Skyjo:
    """Start a round once initial cards are revealed.

    The initial discard is already visible; enable draw or take for the
    first player in the turn order.
    """

    game = skyjo.game
    deck = skyjo.deck

    assert _get_action(game) in {
        ACTION_FLIP_SECOND,
        ACTION_FLIP_OR_REPLACE,
        ACTION_REPLACE,
    }

    new_game = game.copy()

    _replace_action(new_game, ACTION_DRAW_OR_TAKE)

    new_deck = deck.copy()

    return dataclasses.replace(
        skyjo, game=new_game, deck=new_deck, pending_card=None, countdown=None
    )


def draw(skyjo: Skyjo) -> Skyjo:
    """Draw a random card from the deck, placing it in top position.

    This method constructs a copy of `skyjo` with a card randomly
    removed from its `deck` and placed at the top of its `game`.
    """

    game = skyjo.game
    deck = skyjo.deck
    card = skyjo.pending_card
    countdown = skyjo.countdown
    if card is None:
        raise ValueError("Expected a randomly-drawn card")

    new_game = game.copy()
    _push_top(new_game, card)
    _replace_action(new_game, ACTION_FLIP_OR_REPLACE)
    new_deck = deck.copy()
    _remove_card(new_deck, card)
    countdown = _decrement_countdown(countdown)
    return dataclasses.replace(
        skyjo, game=new_game, deck=new_deck, pending_card=None, countdown=countdown
    )


def take(skyjo: Skyjo) -> Skyjo:
    """Take the last discard.

    Since the discard is already stored at the top of the `game`, this
    method only has to update the action.
    """

    game = skyjo.game
    countdown = skyjo.countdown
    new_game = game.copy()
    _replace_action(new_game, ACTION_REPLACE)
    new_countdown = _decrement_countdown(countdown)

    return dataclasses.replace(skyjo, game=new_game, countdown=new_countdown)


def _reveal_card(skyjo: Skyjo, row: int, column: int) -> Skyjo:
    """Reveal and clear cards without advancing turn order or action phase."""
    if not skyjo.table[0, row, column, FINGER_HIDDEN]:
        raise ValueError(f"Cannot reveal visible card at ({row}, {column})")
    card = skyjo.pending_card
    if card is None:
        raise ValueError("Expected a randomly-drawn card")
    game, table, deck = skyjo.game.copy(), skyjo.table.copy(), skyjo.deck.copy()
    table[0, row, column, FINGER_HIDDEN] = 0
    table[0, row, column, card] = 1
    _clear_columns(game, table, column)
    _remove_card(deck, card)
    return dataclasses.replace(
        skyjo, game=game, table=table, deck=deck, pending_card=None
    )


def _finish_reveal_turn(revealed: Skyjo) -> Skyjo:
    """Advance a freshly copied reveal, recording progress before rotation."""
    _update_last_revealed_turns(revealed.game, revealed.turn)
    _rotate_scores(revealed.game, revealed.players)
    _rotate_last_revealed_turns(revealed.game, revealed.players)
    _rotate_table(revealed.table, revealed.players)
    countdown = _update_countdown(
        revealed.countdown,
        revealed.table,
        revealed.players,
        revealed.turn,
        revealed.turn,
    )
    return dataclasses.replace(revealed, turn=revealed.turn + 1, countdown=countdown)


def flip(skyjo: Skyjo, row: int, column: int) -> Skyjo:
    """Reveal a card after discarding a draw, then advance to the next player."""
    revealed = _reveal_card(skyjo, row, column)
    _replace_action(revealed.game, ACTION_DRAW_OR_TAKE)
    return _finish_reveal_turn(revealed)


def reveal_first_card(skyjo: Skyjo) -> Skyjo:
    """Reveal the setup card and rotate, without counting a player turn."""
    return _rotate_skyjo(_reveal_card(skyjo, 0, 0))


def reveal_second_card(skyjo: Skyjo, row: int, column: int) -> Skyjo:
    """Complete one player's setup turn, retaining the setup action phase."""
    return _finish_reveal_turn(_reveal_card(skyjo, row, column))


def reveal_final_card(skyjo: Skyjo, row: int, column: int) -> Skyjo:
    """Reveal a card at round end without advancing turn order or progress."""
    assert get_round_over(skyjo), "Final reveals require a completed round"
    return _reveal_card(skyjo, row, column)


def replace(skyjo: Skyjo, row: int, column: int) -> Skyjo:
    """Replace a card with the current draw, completing a turn.

    This method returns a copy of `skyjo` with the specified finger
    and draw card swapped. If the finger to replace is hidden, a card
    is randomly drawn from the deck and discarded to simulate revealing
    it. This random draw may be determined by setting `card`.
    """

    game = skyjo.game
    table = skyjo.table
    deck = skyjo.deck
    players = skyjo.players
    turn = skyjo.turn
    card = skyjo.pending_card
    countdown = skyjo.countdown
    # If the finger is currently hidden, we need to draw, but only if
    # `card` is not specified.
    new_game = game.copy()
    last_revealed_turn = get_last_revealed_turns(skyjo)[0]
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

    return dataclasses.replace(
        skyjo,
        game=new_game,
        table=new_table,
        deck=new_deck,
        turn=turn + 1,
        pending_card=None,
        countdown=countdown,
    )


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
    if get_round_over(skyjo):
        return mask
    game = skyjo.game
    table = skyjo.table
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
    table, countdown = skyjo.table, skyjo.countdown
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
    return bool(table[0, row, column, FINGER_HIDDEN])


def quick_finish_action(skyjo: Skyjo) -> int:
    """Always draw from the pile and replace the next hidden card."""

    game = skyjo.game
    table = skyjo.table

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


def random_valid_action(skyjo: Skyjo, rng: Random = random) -> int:
    legal = get_actions(skyjo)
    return int(legal[min(int(rng.random() * len(legal)), len(legal) - 1)])


# MARK: Selfplay


def start_round(skyjo: Skyjo, rng: Random = random) -> Skyjo:
    """Reveal each player's first card on a fresh round's board."""
    players = skyjo.players
    for _ in range(players):
        skyjo = randomize(skyjo, rng=rng)
        skyjo = reveal_first_card(skyjo)
    return skyjo


def start_next_round(skyjo: Skyjo, rng: Random = random) -> Skyjo:
    """Deal a fresh round starting with the previous round's ending player.

    The completed round remains unchanged. Its final totals become the new
    round's starting totals, in the same current-player orientation.
    """
    if not get_round_over(skyjo):
        raise ValueError("Cannot start the next round before this round ends")
    if get_game_over(skyjo):
        raise ValueError("Cannot start another round after the game ends")

    players = get_player_count(skyjo)
    starting_player = get_player(skyjo)
    state = new(players=players, rng=rng)
    state.game[GAME_SCORES : GAME_SCORES + players] = get_game_scores(skyjo)
    state.game[GAME_LAST_REVEALED_TURNS : GAME_LAST_REVEALED_TURNS + players] = (
        starting_player
    )
    state = dataclasses.replace(state, turn=starting_player)
    return start_round(state, rng=rng)


def end_round(skyjo: Skyjo, rng: Random = random) -> Skyjo:
    """End the round by flipping all hidden cards."""
    assert get_round_over(skyjo), "Cannot reveal final cards before the round ends"
    players = skyjo.players
    for _ in range(players):
        for row in range(ROW_COUNT):
            for column in range(COLUMN_COUNT):
                if get_finger(skyjo, row, column, player=0) == FINGER_HIDDEN:
                    skyjo = randomize(skyjo, rng=rng)
                    skyjo = reveal_final_card(skyjo, row, column)
        skyjo = _rotate_skyjo(skyjo)
    return skyjo


def apply_action(skyjo: Skyjo, action: SkyjoAction, rng: Random = random) -> Skyjo:
    """Apply one legal action without changing its input.

    Check action preconditions here; call validate explicitly when diagnosing
    full state invariants. Sampling uses only the supplied random generator.
    """
    if get_round_over(skyjo):
        raise ValueError("Cannot play an action after the round ends")
    if not 0 <= action < MASK_SIZE or not actions(skyjo)[action]:
        raise ValueError(f"Action {action!r} is not legal in this state")
    players, countdown = get_player_count(skyjo), get_countdown(skyjo)
    # Apply action to skyjo
    if action == MASK_FLIP_SECOND_BELOW:
        skyjo = randomize(skyjo, rng=rng)
        skyjo = reveal_second_card(skyjo, 1, 0)
        # All players have flipped initial cards, start round
        if skyjo.table[:, :, :, :CARD_SIZE].sum().item() == players * 2:
            skyjo = begin(skyjo)
    elif action == MASK_FLIP_SECOND_RIGHT:
        skyjo = randomize(skyjo, rng=rng)
        skyjo = reveal_second_card(skyjo, 0, 1)
        # All players have flipped initial cards, start round
        if skyjo.table[:, :, :, :CARD_SIZE].sum().item() == players * 2:
            skyjo = begin(skyjo)
    elif action == MASK_DRAW:
        skyjo = randomize(skyjo, rng=rng)
        skyjo = draw(skyjo)
    elif action == MASK_TAKE:
        skyjo = take(skyjo)
    elif MASK_FLIP <= action < MASK_FLIP + FINGER_COUNT:
        row, column = divmod(action - MASK_FLIP, COLUMN_COUNT)
        skyjo = randomize(skyjo, rng=rng)
        skyjo = flip(skyjo, row, column)

    elif MASK_REPLACE <= action < MASK_REPLACE + FINGER_COUNT:
        row, column = divmod(action - MASK_REPLACE, COLUMN_COUNT)
        if skyjo.table[0, row, column, FINGER_HIDDEN]:
            skyjo = randomize(skyjo, rng=rng)

        skyjo = replace(skyjo, row, column)

    else:
        raise ValueError(f"Invalid action {action!r}")

    countdown = skyjo.countdown
    if countdown == 0:
        skyjo = end_round(skyjo, rng=rng)

    return skyjo


def chance_outcomes(skyjo: Skyjo, action: SkyjoAction) -> list[tuple[Skyjo, float]]:
    """Enumerate an ordinary action's single-card outcomes without mutation.

    Round endings may reveal several cards and must be sampled separately.
    Recycling and a preordained next card follow the same rules as apply_action.
    """
    if get_round_about_to_end(skyjo):
        raise ValueError("Round-ending outcomes must be sampled")
    if not is_action_random(action, skyjo):
        return [(apply_action(skyjo, action), 1.0)]
    prepared = prepare_draw_pile(skyjo)
    if prepared.pending_card is not None:
        return [(apply_action(prepared, action), 1.0)]
    total = prepared.deck.sum()
    return [
        (apply_action(preordain(prepared, card), action), float(count / total))
        for card, count in enumerate(prepared.deck)
        if count > 0
    ]
