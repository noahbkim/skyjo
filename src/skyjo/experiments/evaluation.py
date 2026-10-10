"""Seat-balanced full-game checkpoint comparisons, independent of training."""

from __future__ import annotations

import dataclasses
import multiprocessing as mp
import pathlib
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np
import torch

from skyjo.learning import checkpoint
from skyjo.simulation import play
from skyjo.experiments import runs, contestants
from skyjo.experiments.contestants import ContestantConfig
from skyjo.analytics.comparisons import summarize_match
from skyjo.simulation.jobs import GameRandomStreams, derive_game_seed


@dataclasses.dataclass(frozen=True)
class EvaluationConfig:
    control: ContestantConfig
    variant: ContestantConfig
    seed_count: int = 32
    seed: int = 0
    workers: int = 1
    threads: int = 1

    def __post_init__(self):
        for name in ("seed_count", "workers", "threads"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"evaluation.{name} must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("evaluation.seed must fit a uint32")
        if self.control.checkpoint is None or self.variant.checkpoint is None:
            raise ValueError("Checkpoint comparisons require two checkpoint paths")


@dataclasses.dataclass(frozen=True)
class MatchResult:
    settings: EvaluationConfig
    checkpoints: dict
    boundary_checkpoints: dict
    games: tuple[dict, ...]
    seconds: float
    effective_workers: int

    def to_dict(self):
        return {
            "settings": dataclasses.asdict(self.settings),
            "execution": {
                "requested_workers": self.settings.workers,
                "effective_workers": self.effective_workers,
                "threads_per_worker": self.settings.threads,
            },
            "checkpoints": self.checkpoints,
            "boundary_value_checkpoints": self.boundary_checkpoints,
            "games": list(self.games),
            **summarize_match(self.games),
            "evaluation_seconds": self.seconds,
        }


def _build_agents(configurations):
    agents = {
        name: contestants.prepare(config) for name, config in configurations.items()
    }
    if any(agent.evaluator.model.players != 2 for agent in agents.values()):
        raise ValueError("Checkpoint evaluation currently requires two players")
    return agents


def _play_game(agents, job):
    index, seed, seats = job
    streams = GameRandomStreams.from_seed(seed)
    player_seeds = streams.search.integers(0, 2**32, size=2)
    players = [
        agents[name].player(np.random.default_rng(int(player_seeds[i])))
        for i, name in enumerate(seats)
    ]
    result = play.play_game(
        players, environment_rng=streams.environment, action_rng=streams.actions
    )
    variant_seat = seats.index("variant")
    scores = result.final_scores
    return {
        "seed": seed,
        "seed_index": index,
        "seats": list(seats),
        "cumulative_scores": list(scores),
        "variant_win_credit": (
            1.0 / len(result.winners) if variant_seat in result.winners else 0.0
        ),
        "control_minus_variant": scores[1 - variant_seat] - scores[variant_seat],
    }


_worker_agents = None
_worker_inputs = None


def _initialize_worker(configurations, threads):
    global _worker_agents, _worker_inputs
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    _worker_agents = None
    _worker_inputs = configurations


def _play_worker_game(job):
    global _worker_agents
    if _worker_agents is None:
        # Load once per worker and report loading errors through the task future.
        _worker_agents = _build_agents(_worker_inputs)
    return _play_game(_worker_agents, job)


def evaluate_checkpoints(
    settings: EvaluationConfig, *, on_game: Callable[[dict], None] | None = None
) -> MatchResult:
    """Compare both seats; frozen evaluators load once per worker."""
    rng_state = checkpoint.capture_rng_state()
    previous_threads = torch.get_num_threads()
    started = time.perf_counter()
    configurations = {"control": settings.control, "variant": settings.variant}
    try:
        torch.set_num_threads(settings.threads)
        identities = {
            name: {
                "path": str(pathlib.Path(config.checkpoint).resolve()),
                "sha256": runs.file_digest(pathlib.Path(config.checkpoint)),
            }
            for name, config in configurations.items()
        }
        boundary_identities = {
            name: None
            if config.boundary_checkpoint is None
            else {
                "path": str(pathlib.Path(config.boundary_checkpoint).resolve()),
                "sha256": runs.file_digest(pathlib.Path(config.boundary_checkpoint)),
            }
            for name, config in configurations.items()
        }
        agents = _build_agents(configurations)
        jobs = []
        for index in range(settings.seed_count):
            seed = derive_game_seed(settings.seed, index, 0x4556414C)
            for seats in (("control", "variant"), ("variant", "control")):
                jobs.append((index, seed, seats))
        records = []

        def collect(record):
            record.update(
                checkpoints=identities, boundary_value_checkpoints=boundary_identities
            )
            records.append(record)
            if on_game is not None:
                on_game(record)

        effective_workers = min(settings.workers, len(jobs))
        if effective_workers == 1:
            for job in jobs:
                collect(_play_game(agents, job))
        else:
            del agents
            executor = ProcessPoolExecutor(
                max_workers=effective_workers,
                mp_context=mp.get_context("spawn"),
                initializer=_initialize_worker,
                initargs=(configurations, settings.threads),
            )
            remaining = iter(jobs)
            pending = set()
            try:
                for _ in range(effective_workers):
                    pending.add(executor.submit(_play_worker_game, next(remaining)))
                while pending:
                    finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in finished:
                        collect(future.result())
                        job = next(remaining, None)
                        if job is not None:
                            pending.add(executor.submit(_play_worker_game, job))
            finally:
                for future in pending:
                    future.cancel()
                executor.shutdown(wait=True, cancel_futures=True)
        records.sort(
            key=lambda record: (record["seed_index"], record["seats"][0] != "control")
        )
        return MatchResult(
            settings,
            identities,
            boundary_identities,
            tuple(records),
            time.perf_counter() - started,
            effective_workers,
        )
    finally:
        torch.set_num_threads(previous_threads)
        checkpoint.restore_rng_state(rng_state)


def launch_comparison(
    settings: EvaluationConfig,
    runs_dir: pathlib.Path,
    *,
    repository: pathlib.Path,
    allow_dirty=False,
):
    """Record a complete comparison, preserving completed games on failure."""
    import json
    import logging

    config = dataclasses.asdict(settings)
    recorder = runs.RunRecorder.create(
        root=runs_dir,
        repository=repository,
        input_path=pathlib.Path("evaluation.json"),
        input_bytes=json.dumps(config, indent=2).encode(),
        configuration=config,
        entrypoint="run_checkpoint_comparison.py:compare",
        invocation=[],
        allow_dirty=allow_dirty,
    )
    completed = 0

    def record_game(record):
        nonlocal completed
        completed += 1
        recorder.record_event(
            "evaluation_game_completed",
            progress={"evaluated_games": completed},
            context=record,
        )
        logging.info(
            "Evaluation game %s/%s: %s",
            completed,
            settings.seed_count * 2,
            record["cumulative_scores"],
        )

    with recorder:
        result = evaluate_checkpoints(settings, on_game=record_game)
        path = recorder.path / "comparison.json"
        path.write_text(json.dumps(result.to_dict(), indent=2, allow_nan=False) + "\n")
        recorder.register_artifact(
            path, kind="checkpoint_comparison", progress={"evaluated_games": completed}
        )
    return recorder.path
