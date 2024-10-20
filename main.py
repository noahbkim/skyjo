import argparse
import multiprocessing
import importlib

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
    