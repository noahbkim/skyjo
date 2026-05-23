from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
import random
from collections.abc import Sequence
from typing import assert_never

import skyjo2
import skyjo2.play
import skyjo2.render


def where(finger_index: int) -> tuple[int, int]:
    return (finger_index % skyjo2.HAND_ROWS, finger_index // skyjo2.HAND_ROWS)


def parse(finger_position: str) -> int | None:
    finger_position = finger_position.removeprefix("(").removesuffix(")")
    with contextlib.suppress(ValueError):
        return int(finger_position)
    parts = finger_position.split(",")
    if len(parts) == 2:
        with contextlib.suppress(ValueError):
            return int(parts[0]) + int(parts[1]) * skyjo2.HAND_ROWS
    return None


def indices(of: Sequence[object]) -> range:
    return range(-len(of), len(of))


class RandomPlayer(skyjo2.play.Player):
    """Proof of concept player that picks a random action."""

    rng: random.Random

    def __init__(self, rng: random.Random = random) -> None:
        self.rng = rng

    def play(self, game: skyjo2.Game) -> skyjo2.play.Action:
        actions = tuple(skyjo2.play.iter_actions(game))
        assert len(actions) > 0
        return self.rng.choice(actions)


class InteractivePlayer(skyjo2.play.Player):
    """Prompts the user for their own actions."""

    def play(self, game: skyjo2.Game) -> skyjo2.play.Action:
        if game.state == skyjo2.GameState.REVEAL_SECOND_CARD:
            while True:
                choice = input(
                    "> reveal the second card below the top left (1, 0) or to the right of it (0, 1) [b/r/(r, c)] "
                )
                if "below".startswith(choice.strip().lower()):
                    return skyjo2.play.RevealSecondCard(1)
                elif "right".startswith(choice.strip().lower()):
                    return skyjo2.play.RevealSecondCard(skyjo2.HAND_ROWS)
                elif (finger_index := parse(choice)) is not None:
                    if finger_index not in indices(game.player.hand):
                        print("invalid choice: card is out of bounds")
                        continue
                    if finger_index in {0, -len(game.player.hand)}:
                        print("invalid choice: card is already revealed")
                        continue
                    return skyjo2.play.RevealSecondCard(finger_index)
                print("invalid choice")
        elif game.state == skyjo2.GameState.DRAW_OR_REPLACE_WITH_DISCARD:
            while True:
                choice = input(
                    "> draw or replace a card with the discard [d/r (r, c)] "
                )
                action, _, finger_position = choice.partition(" ")
                if "draw".startswith(choice.strip().lower()):
                    return skyjo2.play.DrawCard()
                elif "replace".startswith(choice.strip().lower()):
                    if (finger_index := parse(finger_position)) is None:
                        print("invalid choice: cannot parse card position")
                        continue
                    if finger_index not in indices(game.player.hand):
                        print("invalid choice: card out of bounds")
                        continue
                    return skyjo2.play.ReplaceCardWithDiscard(finger_index)
                print("invalid choice")
        elif (
            game.state == skyjo2.GameState.DISCARD_DRAW_AND_REVEAL_OR_REPLACE_WITH_DRAW
        ):
            while True:
                choice = input(
                    "> discard draw and reveal a card or place the draw in your hand [r (r, c)/p (r, c)] "
                )
                action, _, finger_position = choice.partition(" ")
                if "reveal".startswith(action.strip().lower()):
                    if (finger_index := parse(choice)) is None:
                        print("invalid choice: cannot parse card position")
                        continue
                    if finger_index not in indices(game.player.hand):
                        print("invalid choice: card out of bounds")
                        continue
                    if game.player.hand[finger_index].is_revealed:
                        print("invalid choice: card is already revealed")
                        continue
                    if game.player.hand[finger_index].is_cleared:
                        print("invalid choice: card is cleared")
                        continue
                    return skyjo2.play.DiscardDrawAndRevealCard(finger_index)
                if "place".startswith(action.strip().lower()):
                    if (finger_index := parse(choice)) is None:
                        print("invalid choice: cannot parse card position")
                        continue
                    if finger_index not in indices(game.player.hand):
                        print("invalid choice: card out of bounds")
                        continue
                    return skyjo2.play.ReplaceCardWithDraw(finger_index)
                print("invalid choice")
        assert_never()


PLAYERS = {
    "random": RandomPlayer,
    "RandomPlayer": RandomPlayer,
    "interactive": InteractivePlayer,
    "InteractivePlayer": InteractivePlayer,
}


class NarratedPlayer(skyjo2.play.Player):
    inner: skyjo2.play.Player

    def __init__(self, inner: skyjo2.play.Player) -> None:
        self.inner = inner

    def play(self, game: skyjo2.Game) -> skyjo2.play.Action:
        action = self.inner.play(game)
        me = f"p{game.player_index_fixed}"
        match action:
            case skyjo2.play.RevealSecondCard(finger_index):
                print(f"# {me} revealed second card {where(finger_index)}")
            case skyjo2.play.DrawCard():
                print(f"# {me} drew a card")
            case skyjo2.play.DiscardDrawAndRevealCard(finger_index):
                print(f"# {me} discarded their draw and revealed {where(finger_index)}")
            case skyjo2.play.ReplaceCardWithDraw(finger_index):
                print(f"# {me} replaced {where(finger_index)} with their draw")
            case skyjo2.play.ReplaceCardWithDiscard(finger_index):
                print(f"# {me} replaced {where(finger_index)} with the discard")
        print()
        return action


class Args:
    seed: int
    players: list[str]


def main() -> None:
    parser = argparse.ArgumentParser(description="Play Skyjo!")
    parser.add_argument("-s", "--seed", type=int, default=None)
    parser.add_argument("players", nargs="+")
    args = parser.parse_args(namespace=Args())

    seed = (
        args.seed
        if args.seed is not None
        else int.from_bytes(os.urandom(8), byteorder="big")
    )
    rng = random.Random(seed)

    players = []
    if len(args.players) < 2 or len(args.players) > 8:
        parser.error(f"unsupported player count {len(args.players)}")
    for player in args.players:
        player, _, player_data = player.partition("=")
        player_args = (json.loads(player_data),) if player_data else ()

        module_name, _, player_name = player.rpartition(":")
        if module_name:
            module = importlib.import_module(module_name)
            player_factory = getattr(module, player_name, None)
            if player_factory is None:
                parser.error(f"module {module_name} has no {player_name}")
        else:
            player_factory = PLAYERS.get(player_name)
            if player_factory is None:
                parser.error(f"no builtin player {player_name}")

        players.append(player_factory(*player_args, rng=rng))

    print(f"seed: {seed}")
    print(f"players: {len(players)}")
    print()
    for game in skyjo2.play.play(tuple(map(NarratedPlayer, players)), rng=rng):
        if game.state == skyjo2.GameState.DEAL_FIRST_CARDS:
            print("# all players were dealt their first card")
            print()
            continue
        for line in skyjo2.render.render(game):
            print(line)
        input()

    final_scores = game.final_scores
    for i in range(len(game.players)):
        player_index = (game.turn - i) % len(game.players)
        me = f"p{i}"
        ended = " (ended)" if player_index == 0 else ""
        print(f"{me}: {final_scores[player_index]}{ended}")


main()
