from __future__ import annotations

import random
from collections.abc import Sequence
from enum import IntEnum, auto
from typing import Iterator, Protocol, assert_never

from ._game import HAND_ROWS, Game, GameState, Rule

# MARK: Action


class ActionKind(IntEnum):
    REVEAL_SECOND_CARD = auto()
    DRAW_CARD = auto()
    DISCARD_DRAW_AND_REVEAL_CARD = auto()
    REPLACE_WITH_DRAW = auto()
    REPLACE_WITH_DISCARD = auto()


type Action = (
    tuple[ActionKind.REVEAL_SECOND_CARD, int]
    | tuple[ActionKind.DRAW_CARD]
    | tuple[ActionKind.DISCARD_DRAW_AND_REVEAL_CARD, int]
    | tuple[ActionKind.REPLACE_WITH_DRAW, int]
    | tuple[ActionKind.REPLACE_WITH_DISCARD, int]
)


def iter_actions(game: Game) -> Iterator[Action]:
    """Yield all actions that may be played for a given game."""

    if game.state == GameState.REVEAL_SECOND_CARD:
        assert game.player.hand[0].is_revealed
        # Only distinct choices are same column and different column.
        assert not game.player.hand[1].is_revealed
        yield (ActionKind.REVEAL_SECOND_CARD, 1)  # same column
        assert not game.player.hand[HAND_ROWS].is_revealed
        yield (ActionKind.REVEAL_SECOND_CARD, HAND_ROWS)  # different column

    elif game.state == GameState.DRAW_OR_REPLACE_WITH_DISCARD:
        yield (ActionKind.DRAW_CARD,)
        for i, finger in enumerate(game.player.hand):
            if not finger.is_cleared:
                yield (ActionKind.REPLACE_WITH_DISCARD, i)

    elif game.state == GameState.DISCARD_DRAW_AND_REVEAL_OR_REPLACE_WITH_DRAW:
        for i, finger in enumerate(game.player.hand):
            if not finger.is_revealed:
                assert not finger.is_cleared
                yield (ActionKind.DISCARD_DRAW_AND_REVEAL_CARD, i)
            if not finger.is_cleared:
                yield (ActionKind.REPLACE_WITH_DRAW, i)

    else:
        assert_never(game.state)


def with_action(game: Game, action: Action, rng: random.Random) -> Game:
    match game, action:
        case (
            Game(state=GameState.REVEAL_SECOND_CARD),
            (ActionKind.REVEAL_SECOND_CARD, finger_index),
        ):
            return game.with_random_second_card_revealed(finger_index, rng)
        case (
            Game(state=GameState.DRAW_OR_REPLACE_WITH_DISCARD),
            (ActionKind.DRAW_CARD,),
        ):
            return game.with_random_card_drawn(rng)
        case (
            Game(state=GameState.DISCARD_DRAW_AND_REVEAL_OR_REPLACE_WITH_DRAW),
            (ActionKind.DISCARD_DRAW_AND_REVEAL_CARD, finger_index),
        ):
            return game.with_draw_discarded_and_random_card_revealed(finger_index, rng)
        case (
            Game(state=GameState.DISCARD_DRAW_AND_REVEAL_OR_REPLACE_WITH_DRAW),
            (ActionKind.REPLACE_WITH_DRAW, finger_index),
        ):
            return game.with_random_card_replaced_with_draw(finger_index, rng)
        case (
            Game(state=GameState.DRAW_OR_REPLACE_WITH_DISCARD),
            (ActionKind.REPLACE_WITH_DISCARD, finger_index),
        ):
            return game.with_random_card_replaced_with_discard(finger_index, rng)
        case _:
            raise Rule(f"Invalid action {action} for game {game}")


# MARK: Transition


type Transition = (
    tuple[ActionKind.REVEAL_SECOND_CARD, int, int]
    | tuple[ActionKind.DRAW_CARD, int]
    | tuple[ActionKind.DISCARD_DRAW_AND_REVEAL_CARD, int, int]
    | tuple[ActionKind.REPLACE_WITH_DRAW, int, int | None]
    | tuple[ActionKind.REPLACE_WITH_DISCARD, int, int | None]
)


def iter_outcomes(game: Game, action: Action) -> Iterator[tuple[int | None, int]]:
    """Yields pairs of card index, weight as outcomes of an action."""

    if (
        # Revealing and drawing are guaranteed to roll a card.
        action[0] in {ActionKind.REVEAL_SECOND_CARD, ActionKind.DRAW_CARD}
        # Otherwise, we only roll if the selected finger is hidden.
        or not game.player.hand[action[1]].is_revealed
    ):
        yield from enumerate(game.draw_pile)
    else:
        yield None, 1


def iter_transitions(game: Game) -> Iterator[Transition]:
    """Yields all possible transitions from a game state."""

    for action in iter_actions(game):
        for outcome, _ in iter_outcomes(game, action):
            yield (*action, outcome)


def with_transition(game: Game, transition: Transition) -> Game:
    match game, transition:
        case (
            Game(state=GameState.REVEAL_SECOND_CARD),
            (ActionKind.REVEAL_SECOND_CARD, finger_index, revealed_card_index),
        ):
            return game.with_second_card_revealed(finger_index, revealed_card_index)
        case (
            Game(state=GameState.DRAW_OR_REPLACE_WITH_DISCARD),
            (ActionKind.DRAW_CARD, revealed_card_index),
        ):
            assert revealed_card_index is not None
            return game.with_card_drawn(revealed_card_index)
        case (
            Game(state=GameState.DISCARD_DRAW_AND_REVEAL_OR_REPLACE_WITH_DRAW),
            (
                ActionKind.DISCARD_DRAW_AND_REVEAL_CARD,
                finger_index,
                revealed_card_index,
            ),
        ):
            assert revealed_card_index is not None
            return game.with_draw_discarded_and_card_revealed(
                finger_index,
                revealed_card_index,
            )
        case (
            Game(state=GameState.DISCARD_DRAW_AND_REVEAL_OR_REPLACE_WITH_DRAW),
            (ActionKind.REPLACE_WITH_DRAW, finger_index, revealed_card_index),
        ):
            return game.with_card_replaced_with_draw(
                finger_index,
                revealed_card_index,
            )
        case (
            Game(state=GameState.DRAW_OR_REPLACE_WITH_DISCARD),
            (ActionKind.REPLACE_WITH_DISCARD, finger_index, revealed_card_index),
        ):
            return game.with_card_replaced_with_discard(
                finger_index,
                revealed_card_index,
            )
        case _:
            raise Rule(f"Invalid transition {transition} for game {game}")


# MARK: Play


class Actor(Protocol):
    """A protocol that enables participation in a Skyjo game."""

    def __call__(self, game: Game) -> Action:
        """Play an action based on the current state of the game."""

        raise NotImplementedError()


def play_round(
    game: Game,
    actors: Sequence[Actor],
    rng: random.Random,
    *,
    no_progress_turn_max: int | None = None,
) -> Iterator[Game]:
    # Keep track of the last time the player revealed or replaced a new card.
    last_progress_turns = [0] * len(actors)

    game = game.with_random_discard_and_first_cards_dealt(rng=rng)
    yield game

    while game.state != GameState.REVEAL_HIDDEN_CARDS:
        turn_index = game.turn_index
        player_hand_revealed_count = game.player.hand_revealed_count
        player_index = game.player_index

        action = actors[player_index](game)
        game = with_action(game, action, rng)

        # Check if the player revealed any new cards. Don't forget to account
        # for the player list rotating as the game state iterates.
        if game.players[player_index].hand_revealed_count > player_hand_revealed_count:
            last_progress_turns[player_index] = turn_index

        yield game

        # Check if the current player has exceeded the limit for turns without
        # making progress, and if so, forfeit. Doing so sets the game's state
        # to `State.NULL`, meaning we don't have to break.
        if game.end_player_index is None and no_progress_turn_max is not None:
            turn_count_per_player = turn_index // len(actors)
            no_progress_turn_count = (
                last_progress_turns[player_index] - turn_count_per_player
            )
            if no_progress_turn_count > no_progress_turn_max:
                game = game.with_forfeit()
                yield game

    # Reveal hidden cards if there are any.
    game = game.with_random_hidden_cards_revealed(rng=rng)
    yield game


def play_game(
    game: Game,
    actors: Sequence[Actor],
    rng: random.Random = random,
    *,
    score_max: int | None = 100,
    no_progress_turn_max: int | None = None,
) -> Iterator[Game]:
    """Orchestrate a game between the provided players."""

    assert 2 <= len(actors) <= 8

    while score_max is None or all(player.score < score_max for player in game.players):
        for game in play_round(
            game,
            actors,
            rng,
            no_progress_turn_max=no_progress_turn_max,
        ):
            yield game
