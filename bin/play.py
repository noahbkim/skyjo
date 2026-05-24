from __future__ import annotations

import contextlib
import random

import skyjo2


class RandomActor(skyjo2.Actor):
    """Proof of concept player that picks a random action."""

    def __call__(self, game: skyjo2.Game) -> skyjo2.Action:
        actions = tuple(skyjo2.iter_actions(game))
        assert len(actions) > 0
        return random.choice(actions)


if __name__ == "__main__":
    game = skyjo2.Game.new(players=4)
    actors = (RandomActor(), RandomActor(), RandomActor(), RandomActor())
    ticks = 0

    with contextlib.suppress(KeyboardInterrupt):
        for game in skyjo2.play_game(game, actors, random, score_max=None):
            ticks += 1

    skyjo2.print_game(game)
    print(f"stopped after {ticks} ticks")
