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

    if game.state == State.DEAL_FIRST_CARD:
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
    no_progress_turn_max: int | None = None,
) -> Iterator[Game]:
    """Orchestrate a game between the provided players."""

    assert 2 <= len(players) <= 8

    # Keep track of the raw turn index on which each player last revealed or
    # replaced a new card. This mirrors the current `skyjo.game` approach,
    # where no-progress is measured in "times this player has come around
    # again" via integer division by player count.
    last_progress_turns = [0] * len(players)

    game = Game.new(players=len(players))
    yield game

    game = game.with_random_first_cards_dealt(rng=rng)
    yield game

    while not game.is_ended_or_forfeited:
        turn = game.turn
        player_index = turn % len(players)

        player_hand_revealed_count = game.player.hand_revealed_count

        action = players[player_index].play(game)
        match game, action:
            case (
                Game(state=State.REVEAL_SECOND_CARD),
                RevealSecondCard(finger_index),
            ):
                game = game.with_random_second_card_revealed(
                    finger_index,
                    rng=rng,
                )
            case (
                Game(state=State.DRAW_OR_REPLACE_WITH_DISCARD),
                DrawCard(),
            ):
                game = game.with_random_drawn_card(
                    rng=rng,
                )
            case (
                Game(state=State.DISCARD_DRAW_AND_REVEAL_OR_REPLACE_WITH_DRAW),
                DiscardDrawAndRevealCard(finger_index),
            ):
                game = game.with_draw_discarded_and_random_card_revealed(
                    finger_index,
                    rng=rng,
                )
            case (
                Game(state=State.DISCARD_DRAW_AND_REVEAL_OR_REPLACE_WITH_DRAW),
                ReplaceCardWithDraw(finger_index),
            ):
                game = game.with_random_card_replaced_with_draw(
                    finger_index,
                    rng=rng,
                )
            case (
                Game(state=State.DRAW_OR_REPLACE_WITH_DISCARD),
                ReplaceCardWithDiscard(finger_index),
            ):
                game = game.with_random_card_replaced_with_discard(
                    finger_index,
                    rng=rng,
                )
            case _, _:
                raise Rule(f"Invalid action {action} for game {game}")

        turn_completed = game.turn > turn
        if turn_completed and turn >= len(players):
            acting_player = (
                game.player if game.is_ended_or_forfeited else game.players[-1]
            )
            if acting_player.hand_revealed_count > player_hand_revealed_count:
                last_progress_turns[player_index] = turn
            elif (
                no_progress_turn_max is not None
                and (turn - last_progress_turns[player_index]) // len(players)
                > no_progress_turn_max
            ):
                game = game.with_forfeit()

        yield game

    # Reveal all hidden cards.
    game = game.with_random_hidden_cards_revealed(rng=rng)
    yield game


class RandomPlayer(Player):
    """Proof of concept player that picks a random action."""

    rng: random.Random

    def __init__(self, rng: random.Random = random) -> None:
        self.rng = rng

    def play(self, game: Game) -> Action:
        actions = tuple(explore(game))
        assert len(actions) > 0
        return self.rng.choice(actions)
