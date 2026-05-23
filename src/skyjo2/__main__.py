from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
import random
import sys
from collections.abc import Sequence
from typing import assert_never

import skyjo2


def where(finger_index: int) -> tuple[int, int]:
    return (finger_index % skyjo2.HAND_ROWS, finger_index // skyjo2.HAND_ROWS)


def parse(finger_position: str) -> int | None:
    finger_position = finger_position.removeprefix("(").removesuffix(")")
    with contextlib.suppress(ValueError):
        return int(finger_position)
    parts = finger_position.split(",")
    if len(parts) == 2:
        with contextlib.suppress(ValueError):
            return int(parts[0].strip()) + int(parts[1].strip()) * skyjo2.HAND_ROWS
    return None


def indices(of: Sequence[object]) -> range:
    return range(-len(of), len(of))


class RandomActor(skyjo2.Actor):
    """Proof of concept player that picks a random action."""

    def __call__(self, game: skyjo2.Game) -> skyjo2.Action:
        actions = tuple(skyjo2.iter_actions(game))
        assert len(actions) > 0
        return random.choice(actions)


class InteractiveActor(skyjo2.Actor):
    """Prompts the user for their own actions."""

    def __call__(self, game: skyjo2.Game) -> skyjo2.Action:
        if game.state == skyjo2.GameState.REVEAL_SECOND_CARD:
            while True:
                choice = input(
                    "> reveal the second card below the top left (1, 0) or to the right of it (0, 1) [b/r/(r, c)] "
                )
                if "below".startswith(choice.strip().lower()):
                    return skyjo2.reveal_second_card(1)
                elif "right".startswith(choice.strip().lower()):
                    return skyjo2.reveal_second_card(skyjo2.HAND_ROWS)
                elif (finger_index := parse(choice)) is not None:
                    if finger_index not in indices(game.player.hand):
                        print("invalid choice: card is out of bounds")
                        continue
                    if finger_index in {0, -len(game.player.hand)}:
                        print("invalid choice: card is already revealed")
                        continue
                    return skyjo2.reveal_second_card(finger_index)
                print("invalid choice")
        elif game.state == skyjo2.GameState.DRAW_OR_REPLACE_WITH_DISCARD:
            while True:
                choice = input(
                    "> draw or replace a card with the discard [d/r (r, c)] "
                )
                action, _, finger_position = choice.partition(" ")
                if "draw".startswith(action.strip().lower()):
                    return skyjo2.draw_card()
                elif "replace".startswith(action.strip().lower()):
                    if (finger_index := parse(finger_position)) is None:
                        print("invalid choice: cannot parse card position")
                        continue
                    if finger_index not in indices(game.player.hand):
                        print("invalid choice: card out of bounds")
                        continue
                    return skyjo2.replace_with_draw(finger_index)
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
                    if (finger_index := parse(finger_position)) is None:
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
                    return skyjo2.discard_draw_and_reveal_card(finger_index)
                if "place".startswith(action.strip().lower()):
                    if (finger_index := parse(finger_position)) is None:
                        print("invalid choice: cannot parse card position")
                        continue
                    if finger_index not in indices(game.player.hand):
                        print("invalid choice: card out of bounds")
                        continue
                    return skyjo2.replace_with_draw(finger_index)
                print("invalid choice")
        assert_never(game.state)


PLAYERS = {
    "random": RandomActor,
    "RandomActor": RandomActor,
    "interactive": InteractiveActor,
    "InteractiveActor": InteractiveActor,
}


class NarratedActor(skyjo2.Actor):
    inner: skyjo2.Actor

    def __init__(self, inner: skyjo2.Actor) -> None:
        self.inner = inner

    def __call__(self, game: skyjo2.Game) -> skyjo2.Action:
        action = self.inner(game)
        me = f"p{game.player_index}"
        match action:
            case (skyjo2.ActionKind.REVEAL_SECOND_CARD, finger_index):
                print(f"# {me} revealed second card {where(finger_index)}")
            case (skyjo2.ActionKind.DRAW_CARD,):
                print(f"# {me} drew a card")
            case (skyjo2.ActionKind.DISCARD_DRAW_AND_REVEAL_CARD, finger_index):
                print(f"# {me} discarded their draw and revealed {where(finger_index)}")
            case (skyjo2.ActionKind.REPLACE_WITH_DRAW, finger_index):
                print(f"# {me} replaced {where(finger_index)} with their draw")
            case (skyjo2.ActionKind.REPLACE_WITH_DISCARD, finger_index):
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

    random.seed(seed)

    actors = []
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

        actors.append(player_factory(*player_args))

    print(f"seed: {seed}")
    print(f"players: {len(actors)}")
    print()
    actors = tuple(map(NarratedActor, actors))
    for game in skyjo2.play(actors, rng=random):
        if game.state == skyjo2.GameState.DEAL_FIRST_CARDS:
            for i, player in enumerate(game.players):
                me = f"p{i}"
                ended = " (ended)" if game.end_player_index == i else ""
                print(f"{me}: {player.score}{ended}")
            print()
            continue

        for line in skyjo2.render(game):
            print(line)

        actor = actors[game.player_index % len(actors)]
        if isinstance(actor.inner, InteractiveActor):
            print()
        else:
            input()


try:
    main()
except (KeyboardInterrupt, EOFError):
    print()
    sys.exit(1)
