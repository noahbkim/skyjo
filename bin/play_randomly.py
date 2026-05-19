from __future__ import annotations

import skyjo2.play as sj_play

if __name__ == "__main__":
    game_count = 0
    for game in sj_play.play((sj_play.RandomPlayer(),) * 4):
        game_count += 1
    print(f"finished on turn {game.turn} in {game_count} iterations")
