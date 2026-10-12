"""Matchup presets and lazy model construction at the application boundary."""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from skyjo.search.mcts import SearchConfig
from skyjo.simulation.jobs import GameRandomStreams
from skyjo.simulation.player import RandomPlayer
from skyjo.simulation.session import GameSession


@dataclass(frozen=True)
class SeatConfig:
    name: str = "Player"
    kind: str = "random"
    checkpoint: str | None = None
    iterations: int = 128
    temperature: float = 0.0
    search: SearchConfig = field(default_factory=SearchConfig)
    boundary_checkpoint: str | None = None

    def __post_init__(self):
        if not self.name.strip() or self.kind not in (
            "human",
            "random",
            "policy",
            "search",
        ):
            raise ValueError("Each seat needs a name and a supported player type")
        if self.kind in ("policy", "search") and not self.checkpoint:
            raise ValueError(f"{self.name}: choose a model checkpoint")
        if type(self.iterations) is not int or self.iterations < 1:
            raise ValueError("Search iterations must be positive")
        if not np.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("Temperature must be finite and nonnegative")


@dataclass(frozen=True)
class MatchConfig:
    seats: tuple[SeatConfig, ...] = field(
        default_factory=lambda: (
            SeatConfig("You", "human"),
            SeatConfig("Random", "random"),
        )
    )
    seed: int = 0

    def __post_init__(self):
        if not 2 <= len(self.seats) <= 8:
            raise ValueError("Choose between 2 and 8 seats")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("Seed must be a nonnegative integer")


def load_preset(path: Path) -> MatchConfig:
    path = Path(path).expanduser().resolve()
    data = tomllib.loads(path.read_text())
    if data.pop("version", None) != 1:
        raise ValueError("Unsupported matchup preset version")
    seats = []
    for raw in data.pop("seats"):
        raw = dict(raw)
        for key in ("checkpoint", "boundary_checkpoint"):
            if raw.get(key):
                raw[key] = str((path.parent / Path(raw[key]).expanduser()).resolve())
        raw["search"] = SearchConfig(**raw.get("search", {}))
        seats.append(SeatConfig(**raw))
    return MatchConfig(seats=tuple(seats), **data)


def save_preset(config: MatchConfig, path: Path):
    """Write explicit settings; absolute paths survive moving the preset."""
    lines = ["version = 1", f"seed = {config.seed}"]
    for seat in config.seats:
        values = asdict(seat)
        search = values.pop("search")
        lines.extend(("", "[[seats]]"))
        for key, value in values.items():
            if value is not None:
                if key in ("checkpoint", "boundary_checkpoint"):
                    value = str(Path(value).expanduser().resolve())
                lines.append(f"{key} = {json.dumps(value, allow_nan=False)}")
        lines.extend(("", "[seats.search]"))
        lines.extend(f"{key} = {json.dumps(value)}" for key, value in search.items())
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def prepare_session(config: MatchConfig) -> GameSession:
    """Load each frozen contestant once; a seat owns its own search RNG."""
    search = GameRandomStreams.from_seed(config.seed).search
    seeds = search.integers(0, 2**32, size=len(config.seats))
    prepared = {}
    players, metadata = [], []
    for seat, seed in zip(config.seats, seeds, strict=True):
        identity = asdict(seat)
        if seat.kind == "human":
            player = None
        elif seat.kind == "random":
            player = RandomPlayer()
        else:
            from skyjo.experiments import contestants

            configuration = contestants.ContestantConfig(
                checkpoint=str(Path(seat.checkpoint).expanduser().resolve()),
                mode=seat.kind,
                iterations=seat.iterations,
                temperature=seat.temperature,
                search=seat.search,
                boundary_checkpoint=str(
                    Path(seat.boundary_checkpoint).expanduser().resolve()
                )
                if seat.boundary_checkpoint
                else None,
            )
            if configuration not in prepared:
                contestant = contestants.prepare(configuration)
                if contestant.evaluator.model.players != len(config.seats):
                    raise ValueError(
                        f"{seat.name}: checkpoint supports {contestant.evaluator.model.players} players, table has {len(config.seats)}"
                    )
                prepared[configuration] = contestant
            player = prepared[configuration].player(np.random.default_rng(int(seed)))
            for key in ("checkpoint", "boundary_checkpoint"):
                path = getattr(configuration, key)
                if path:
                    with Path(path).open("rb") as stream:
                        identity[f"{key}_sha256"] = hashlib.file_digest(
                            stream, "sha256"
                        ).hexdigest()
                    identity[key] = path
        players.append(player)
        metadata.append(identity)
    return GameSession(players, seed=config.seed, metadata=metadata)
