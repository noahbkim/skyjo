import random

import skyjo


class AlwaysDrawPlayer(skyjo.Player):
    """Always draws and replaces their next unflipped card."""

    def flip(self, state: skyjo.State, action: skyjo.Flip) -> None:
        action.flip_card(0)
        action.flip_card(1)

    def turn(self, state: skyjo.State, action: skyjo.Turn) -> None:
        action.draw_card()
        for i, card in enumerate(self.hand):
            if not card.is_visible:
                action.place_drawn_card(i)
                return


if __name__ == "__main__":
    rng = random.Random(0)
    players = [AlwaysDrawPlayer(), AlwaysDrawPlayer(), AlwaysDrawPlayer(), AlwaysDrawPlayer()]
    skyjo.play(players, games=1, processes=1, seed=0, interactive=True)
