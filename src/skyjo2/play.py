from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Iterator, NamedTuple, Protocol

from . import HAND_ROWS, Game, State


class RevealSecondCard(NamedTuple):
    finger_index: int


class DrawCard(NamedTuple):
    pass


class DiscardDrawAndRevealCard(NamedTuple):
    finger_index: int


class ReplaceCardWithDraw(NamedTuple):
    finger_index: int


class ReplaceCardWithDiscard(NamedTuple):
    finger_index: int


type Action = (
    RevealSecondCard
    | DrawCard
    | DiscardDrawAndRevealCard
    | ReplaceCardWithDraw
    | ReplaceCardWithDiscard
)


class Rule(Exception):
    """Raised when a rule is broken, i.e. an invalid action is played."""


class Player(Protocol):
    """A protocol that enables participation in a Skyjo game."""

    def play(self, game: Game) -> Action:
        """Play an action based on the current state of the game."""


def explore(game: Game) -> Iterator[Action]:
    """Yield all actions that may be played for a given game."""

    if game.state == State.NULL:
        return

    elif game.state == State.REVEAL_SECOND_CARD:
        assert game.player.hand[0].is_revealed
        # Only distinct choices are same column and different column.
        assert not game.player.hand[1].is_revealed
        yield RevealSecondCard(1)  # same column
        assert not game.player.hand[HAND_ROWS].is_revealed
        yield RevealSecondCard(HAND_ROWS)  # different column

    elif game.state == State.DRAW_OR_REPLACE_WITH_DISCARD:
        yield DrawCard()
        for i, finger in enumerate(game.player.hand):
            if not finger.is_cleared:
                yield ReplaceCardWithDiscard(i)

    elif game.state == State.DISCARD_DRAW_AND_REVEAL_OR_REPLACE_WITH_DRAW:
        for i, finger in enumerate(game.player.hand):
            if not finger.is_revealed:
                assert not finger.is_cleared
                yield DiscardDrawAndRevealCard(i)
            if not finger.is_cleared:
                yield ReplaceCardWithDraw(i)

    else:
        assert False, f"unknown state {game}"


def play(
    players: Sequence[Player],
    rng: random.Random = random,
    *,
    no_progress_turn_max: int = 20,
) -> Iterator[Game]:
    """Orchestrate a game between the provided players."""

    assert 2 <= len(players) <= 8

    # Keep track of the last time the player revealed or replaced a new card.
    last_progress_turns = [0] * len(players)

    game = Game.new(players=len(players))
    yield game

    game = game.with_random_deal(rng=rng)
    yield game

    while game.state != State.NULL:
        turn = game.turn
        player_index = turn % len(players)
        player_hand_revealed_count = game.player.hand_revealed_count

        action = players[player_index].play(game)
        match game, action:
            case (
                Game(state=State.REVEAL_SECOND_CARD),
                RevealSecondCard(finger_index),
            ):
                game = game.with_second_card_revealed(finger_index)
            case (
                Game(state=State.DRAW_OR_REPLACE_WITH_DISCARD),
                DrawCard(),
            ):
                game = game.with_random_drawn_card(rng=rng)
            case (
                Game(state=State.DISCARD_DRAW_AND_REVEAL_OR_REPLACE_WITH_DRAW),
                DiscardDrawAndRevealCard(finger_index),
            ):
                game = game.with_draw_discarded_and_card_revealed(finger_index)
            case (
                Game(state=State.DISCARD_DRAW_AND_REVEAL_OR_REPLACE_WITH_DRAW),
                ReplaceCardWithDraw(finger_index),
            ):
                game = game.with_card_replaced_with_draw(finger_index)
            case (
                Game(state=State.DRAW_OR_REPLACE_WITH_DISCARD),
                ReplaceCardWithDiscard(finger_index),
            ):
                game = game.with_card_replaced_with_discard(finger_index)
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
        turn_per_player = turn // len(players)
        if last_progress_turns[player_index] - turn_per_player > no_progress_turn_max:
            game = game.with_forfeit()
            yield game


class RandomPlayer(Player):
    """Proof of concept player that picks a random action."""

    def play(self, game: Game) -> Action:
        actions = tuple(explore(game))
        assert len(actions) > 0
        return random.choice(actions)
