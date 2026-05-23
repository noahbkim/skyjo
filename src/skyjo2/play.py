from __future__ import annotations

import random
from collections.abc import Sequence
from enum import IntEnum, auto
from typing import Iterator, Protocol

from . import HAND_ROWS, Game, GameState


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


class Rule(Exception):
    """Raised when a rule is broken, i.e. an invalid action is played."""


class Player(Protocol):
    """A protocol that enables participation in a Skyjo game."""

    def play(self, game: Game) -> Action:
        """Play an action based on the current state of the game."""


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

    assert False


def play(
    players: Sequence[Player],
    rng: random.Random = random,
    *,
    round_max: int | None = None,
    score_max: int | None = 100,
    no_progress_turn_max: int | None = None,
) -> Iterator[Game]:
    """Orchestrate a game between the provided players."""

    assert 2 <= len(players) <= 8

    # Count the total number of rounds we've played.
    round_index = 0

    # Keep track of the last time the player revealed or replaced a new card.
    last_progress_turns = [0] * len(players)

    game = Game.new(players=len(players))
    yield game

    while (round_max is None or round_index < round_max) and (
        score_max is None or all(player.score < score_max for player in game.players)
    ):
        game = game.with_random_discard_and_first_cards_dealt(rng=rng)
        yield game

        while game.state != GameState.REVEAL_HIDDEN_CARDS:
            turn = game.turn
            player_index = game.player.index
            player_hand_revealed_count = game.player.hand_revealed_count

            action = players[player_index].play(game)
            match game, action:
                case (
                    Game(state=GameState.REVEAL_SECOND_CARD),
                    (ActionKind.REVEAL_SECOND_CARD, finger_index),
                ):
                    game = game.with_random_second_card_revealed(
                        finger_index,
                        rng=rng,
                    )
                case (
                    Game(state=GameState.DRAW_OR_REPLACE_WITH_DISCARD),
                    (ActionKind.DRAW_CARD,),
                ):
                    game = game.with_random_drawn_card(
                        rng=rng,
                    )
                case (
                    Game(state=GameState.DISCARD_DRAW_AND_REVEAL_OR_REPLACE_WITH_DRAW),
                    (ActionKind.DISCARD_DRAW_AND_REVEAL_CARD, finger_index),
                ):
                    game = game.with_draw_discarded_and_random_card_revealed(
                        finger_index,
                        rng=rng,
                    )
                case (
                    Game(state=GameState.DISCARD_DRAW_AND_REVEAL_OR_REPLACE_WITH_DRAW),
                    (ActionKind.REPLACE_WITH_DRAW, finger_index),
                ):
                    game = game.with_random_card_replaced_with_draw(
                        finger_index,
                        rng=rng,
                    )
                case (
                    Game(state=GameState.DRAW_OR_REPLACE_WITH_DISCARD),
                    (ActionKind.REPLACE_WITH_DISCARD, finger_index),
                ):
                    game = game.with_random_card_replaced_with_discard(
                        finger_index,
                        rng=rng,
                    )
                case _, _:
                    raise Rule(f"Invalid action {action} for game {game}")

            # Check if the player revealed any new cards. Don't forget to account
            # for the player list rotating as the game state iterates.
            if game.players[-1].hand_revealed_count > player_hand_revealed_count:
                last_progress_turns[player_index] = turn

            yield game

            # Check if the current player has exceeded the limit for turns without
            # making progress, and if so, forfeit. Doing so sets the game's state
            # to `State.NULL`, meaning we don't have to break.
            if not game.is_ending and no_progress_turn_max is not None:
                turn_per_player = (turn + len(players) - 1) // len(players)
                no_progress_turn_count = (
                    last_progress_turns[player_index] - turn_per_player
                )
                if no_progress_turn_count > no_progress_turn_max:
                    game = game.with_forfeit()
                    yield game

        # Reveal hidden cards if there are any.
        game = game.with_random_hidden_cards_revealed(rng=rng)
        yield game

        round_index += 1
