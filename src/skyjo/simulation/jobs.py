"""Stable game identities and independent streams, unaffected by worker scheduling."""

from dataclasses import dataclass

import numpy as np

from .play import GameResult


def derive_game_seed(run_seed: int, game_index: int, stream: int = 0) -> int:
    if min(run_seed, game_index, stream) < 0:
        raise ValueError("Seed, game index, and stream must be nonnegative")
    return int(
        np.random.SeedSequence([run_seed, game_index, stream]).generate_state(1)[0]
    )


@dataclass(frozen=True)
class GameJob:
    index: int
    seed: int
    seats: tuple[str, ...] = ()


@dataclass(frozen=True)
class GeneratedGame:
    global_game_index: int
    play_seed: int
    result: GameResult


@dataclass(frozen=True)
class GameRandomStreams:
    environment: np.random.Generator
    actions: np.random.Generator
    search: np.random.Generator

    @classmethod
    def from_seed(cls, seed: int):
        return cls(
            *(np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3))
        )
