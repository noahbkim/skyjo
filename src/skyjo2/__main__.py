from __future__ import annotations

import os
import random

import skyjo2
import skyjo2.play
import skyjo2.render


def where(finger_index: int) -> tuple[int, int]:
    return divmod(finger_index, skyjo2.HAND_ROWS)


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


class InteractivePlayer(skyjo2.play.Player):
    pass


def main() -> None:
    seed = int.from_bytes(os.urandom(8), byteorder="big")
    print(f"seed: {seed}")

    rng = random.Random(seed)
    players = []
    for _ in range(3):
        players.append(skyjo2.play.RandomPlayer(rng=rng))
    print(f"players: {len(players)}")

    print()
    for game in skyjo2.play.play(tuple(map(NarratedPlayer, players)), rng=rng):
        if game.state == skyjo2.State.DEAL_FIRST_CARD:
            print("# all players were dealt their first card")
            continue
        for line in skyjo2.render.render(game):
            print(line)
        input()

    final_scores = game.final_scores
    for i in range(len(game.players)):
        player_index = (i + game.turn) % len(game.players)
        me = f"p{i}"
        ended = " (ended)" if player_index == 0 else ""
        print(f"{me}: {final_scores[player_index]}{ended}")


main()
