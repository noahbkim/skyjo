import argparse
import multiprocessing
import importlib
import random

import skyjo


class DrawPlayer(skyjo.Player):
    """Always draws and replaces their next unflipped card."""

    def flip(self, state: skyjo.State, action: skyjo.Flip) -> None:
        action.flip_card(0)
        action.flip_card(1)

    def turn(self, state: skyjo.State, action: skyjo.Turn) -> None:
        action.draw_card()
        action.place_drawn_card(self.hand.first_unflipped_or_highest_card_index)


class RandomPlayer(skyjo.Player):
    """Uses RNG to make all decisions."""

    def flip(self, state: skyjo.State, action: skyjo.Flip) -> None:
        action.flip_card(0, 0)
        action.flip_card(*((0, 1) if random.random() >= 0.5 else (1, 0)))

    def turn(self, state: skyjo.State, action: skyjo.Turn) -> None:
        choice = random.randint(1, 2 if self.hand.are_all_cards_flipped else 3)
        if choice == 1:
            action.draw_card()
            action.place_drawn_card(self.hand.first_unflipped_or_highest_card_index)
        elif choice == 2:
            action.place_from_discard(self.hand.first_unflipped_or_highest_card_index)
        elif choice == 3:
            action.draw_card()
            action.discard_and_flip(self.hand.first_unflipped_or_highest_card_index)


class ThresholdPlayer(skyjo.Player):
    """Draws under average, tries to clear."""

    def flip(self, state: skyjo.State, action: skyjo.Flip) -> None:
        action.flip_card(0)
        action.flip_card(1)

    def turn(self, state: skyjo.State, action: skyjo.Turn) -> None:
        if state.top_discard < state.deck_average:
            action.place_from_discard(self.hand.first_unflipped_or_highest_card_index)
            return
        card = action.draw_card()
        if card < state.deck_average:
            action.place_drawn_card(self.hand.first_unflipped_or_highest_card_index)
        elif self.hand.are_all_cards_flipped:
            action.place_drawn_card(self.hand.highest_card_index)
        else:
            action.discard_and_flip(self.hand.first_unflipped_or_highest_card_index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("players", nargs="+", help="qualified player classes to instantiate.", metavar="PLAYER")
    parser.add_argument("-n", "--games", type=int, default=1, help="the number of games to play.")
    parser.add_argument("-s", "--seed", type=int, default=None, help="a seed for random number generation.")
    parser.add_argument("-i", "--interactive", default=False, action="store_true")
    parser.add_argument("-m", "--multiprocess", type=int, nargs="?", default=(), help="the number of subprocesses to use.")
    args = parser.parse_args()

    players: list[skyjo.Player] = []
    for name in args.players:
        rest, colon, data = name.partition(":")
        payload = (data,) if colon else ()
        rest, plus, number = rest.partition("+")
        times = int(number) if plus else 0
        module, dot, symbol = rest.rpartition(".")
        cls = getattr(importlib.import_module(module), symbol) if dot else globals()[symbol]
        for _ in range(times + 1):
            players.append(cls(*payload))

    if len(players) < 3:
        raise parser.error(f"Must have at least 3 players, got {len(players)}")
    elif len(players) > 8:
        raise parser.error(f"Must have 8 or fewer players, got {len(players)}")
    
    if args.multiprocess == ():
        processes = 1
    elif args.multiprocess is None:
        processes = multiprocessing.cpu_count()
    else:
        processes = args.multiprocess[0]

    try:
        skyjo.play(players, games=args.games, seed=args.seed, interactive=args.interactive, processes=processes)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
