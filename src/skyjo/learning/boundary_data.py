"""Observed between-round examples for the offline score-value experiment."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np


@dataclasses.dataclass(frozen=True)
class BoundaryDataset:
    scores: np.ndarray
    targets: np.ndarray
    game_ids: np.ndarray
    round_numbers: np.ndarray
    iterations: np.ndarray
    sources: list[dict]
    summary: dict


def _integer(record: dict, key: str, location: str, minimum: int = 0) -> int:
    value = record.get(key)
    if type(value) is not int or value < minimum:
        raise ValueError(f"{location}: {key} must be an integer >= {minimum}")
    return value


def _scores(record: dict, key: str, location: str, players: int | None) -> np.ndarray:
    values = record.get(key)
    if (
        not isinstance(values, list)
        or len(values) < 2
        or (players is not None and len(values) != players)
        or any(type(value) is not int for value in values)
    ):
        raise ValueError(
            f"{location}: {key} must contain matching finite integer scores"
        )
    try:
        return np.asarray(values, dtype=np.int64)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{location}: {key} contains out-of-range scores") from error


def load_boundaries(paths: Sequence[Path]) -> BoundaryDataset:
    """Read completed games, retaining only observed nonterminal boundaries.

    Scores are unscaled and rotated so slot zero starts the next round. Labels
    describe the eventual winner of the observed game, in that same ordering.
    Truncated games and games started partway through a round are excluded.
    Malformed records, missing rounds, and duplicate rounds are errors.
    """
    resolved = sorted(Path(path).resolve() for path in paths)
    if not resolved or len(set(resolved)) != len(resolved):
        raise ValueError("Provide at least one source path, with no duplicate paths")

    games = defaultdict(dict)
    provenance = {}
    sources = []
    players = None
    record_count = 0
    for path in resolved:
        content = path.read_bytes()
        source_count = 0
        for line_number, line in enumerate(content.splitlines(), start=1):
            if not line.strip():
                continue
            location = f"{path}:{line_number}"
            try:
                record = json.loads(line)
            except (ValueError, UnicodeDecodeError) as error:
                raise ValueError(f"{location}: invalid JSON") from error
            if not isinstance(record, dict):
                # Invalid serialized content is a value error, not a caller type error.
                raise ValueError(f"{location}: expected a round record object")  # noqa: TRY004
            run_id = record.get("run_id")
            if not isinstance(run_id, str) or not run_id:
                raise ValueError(f"{location}: run_id must be a nonempty string")
            game_index = _integer(record, "game_index", location)
            number = _integer(record, "round_number", location, minimum=1)
            iteration = _integer(record, "iteration", location)
            play_seed = _integer(record, "play_seed", location)
            cumulative = _scores(record, "cumulative_scores", location, players)
            players = len(cumulative)
            charged = _scores(record, "scores", location, players)
            if "raw_scores" in record:
                _scores(record, "raw_scores", location, players)
            starter = _integer(record, "ending_player", location)
            if starter >= players:
                raise ValueError(
                    f"{location}: ending_player is outside the player range"
                )
            if type(record.get("partial_start")) is not bool:
                raise ValueError(f"{location}: partial_start must be a boolean")
            game_key = (run_id, game_index)
            identity = (
                iteration,
                play_seed,
                record.get("checkpoint_artifact_id"),
                record.get("checkpoint_path"),
            )
            if game_key in provenance and provenance[game_key] != identity:
                raise ValueError(
                    f"{location}: conflicting provenance for game {game_key}"
                )
            provenance[game_key] = identity
            if number in games[game_key]:
                raise ValueError(
                    f"{location}: duplicate round {number} for game {game_key}"
                )
            games[game_key][number] = {
                "cumulative": cumulative,
                "charged": charged,
                "starter": starter,
                "partial": record["partial_start"],
                "iteration": iteration,
                "location": location,
            }
            source_count += 1
        sources.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
                "records": source_count,
            }
        )
        record_count += source_count

    score_rows, targets, game_ids, round_numbers, iterations = [], [], [], [], []
    incomplete_games = partial_games = completed_games = boundary_games = 0
    for game_key, rounds in sorted(games.items()):
        numbers = sorted(rounds)
        if numbers != list(range(1, len(numbers) + 1)):
            raise ValueError(
                f"Game {game_key}: round sequence must start at 1 without gaps"
            )
        partial = any(rounds[number]["partial"] for number in numbers)
        previous = np.zeros(players, dtype=np.int64)
        for number in numbers:
            row = rounds[number]
            if (number > 1 or not partial) and not np.array_equal(
                previous + row["charged"], row["cumulative"]
            ):
                raise ValueError(f"{row['location']}: inconsistent cumulative scores")
            if number < len(numbers) and np.any(row["cumulative"] >= 100):
                raise ValueError(
                    f"{row['location']}: round recorded after game termination"
                )
            previous = row["cumulative"]
        if partial:
            partial_games += 1
            continue
        if not np.any(previous >= 100):
            incomplete_games += 1
            continue
        completed_games += 1
        if len(numbers) > 1:
            boundary_games += 1
        winners = (previous == previous.min()).astype(np.float32)
        winners /= winners.sum()
        for number in numbers[:-1]:
            row = rounds[number]
            score_rows.append(np.roll(row["cumulative"], -row["starter"]))
            targets.append(np.roll(winners, -row["starter"]))
            # JSON makes the composite identity unambiguous even for arbitrary run IDs.
            game_ids.append(json.dumps(game_key, separators=(",", ":")))
            round_numbers.append(number)
            iterations.append(row["iteration"])

    if not score_rows:
        raise ValueError(
            "Sources contain no observed nonterminal boundaries from complete games"
        )
    return BoundaryDataset(
        scores=np.asarray(score_rows, dtype=np.float32),
        targets=np.asarray(targets, dtype=np.float32),
        game_ids=np.asarray(game_ids, dtype=str),
        round_numbers=np.asarray(round_numbers, dtype=np.int64),
        iterations=np.asarray(iterations, dtype=np.int64),
        sources=sources,
        summary={
            "players": players,
            "source_records": record_count,
            "games_total": len(games),
            "games_completed": completed_games,
            "games_with_boundaries": boundary_games,
            "games_excluded_incomplete": incomplete_games,
            "games_excluded_partial_start": partial_games,
            "boundaries": len(score_rows),
        },
    )


def split_by_game(
    dataset: BoundaryDataset,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Split unique games, preserving every boundary of a game in one partition."""
    if (
        not all(
            math.isfinite(f) and f > 0 for f in (validation_fraction, test_fraction)
        )
        or validation_fraction + test_fraction >= 1
    ):
        raise ValueError(
            "Validation and test fractions must be positive and sum to less than 1"
        )
    games = np.unique(dataset.game_ids)
    validation_count = max(1, round(len(games) * validation_fraction))
    test_count = max(1, round(len(games) * test_fraction))
    if len(games) <= validation_count + test_count:
        raise ValueError(
            "Need enough games for nonempty train, validation, and test splits"
        )
    shuffled = np.random.default_rng(seed).permutation(games)
    validation_end = validation_count
    test_end = validation_end + test_count
    groups = {
        "train": shuffled[test_end:],
        "validation": shuffled[:validation_end],
        "test": shuffled[validation_end:test_end],
    }
    return {
        name: np.flatnonzero(np.isin(dataset.game_ids, selected))
        for name, selected in groups.items()
    }
