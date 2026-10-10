"""CPU game-generation workers for evolving self-play model snapshots."""

from __future__ import annotations
import logging
import queue
import typing
import numpy as np
import torch
from skyjo.learning import models
from skyjo.engine import game as sj
from skyjo.simulation import play
from skyjo.simulation.jobs import GeneratedGame, GameRandomStreams, derive_game_seed
from skyjo.experiments import experiment_training, contestants
from skyjo.experiments.contestants import ContestantConfig

StartStateGenerator = typing.Callable[[np.random.Generator], sj.Skyjo]


def configure_torch_worker(torch_thread_count: int):
    torch.set_num_threads(torch_thread_count)
    torch.set_num_interop_threads(1)


def play_games_locally(
    model_state_dict,
    model_settings,
    model_player_config: ContestantConfig,
    players: int,
    number_of_games: int,
    run_seed: int,
    first_game_index: int,
    start_state_generator: StartStateGenerator | None = None,
    auxiliary_objectives=None,
) -> list[GeneratedGame]:
    model = models.build(
        model_settings,
        players=players,
        device="cpu",
        auxiliary_objectives=auxiliary_objectives,
    )
    model.load_state_dict(model_state_dict)
    prepared = contestants.prepare(model_player_config, model=model.eval())
    games = []
    for index in range(first_game_index, first_game_index + number_of_games):
        seed = derive_game_seed(run_seed, index)
        streams = GameRandomStreams.from_seed(seed)
        agent = prepared.player(streams.search)
        start = (
            None
            if start_state_generator is None
            else start_state_generator(streams.environment)
        )
        result = play.play_game(
            [agent] * players,
            start_state=start,
            environment_rng=streams.environment,
            action_rng=streams.actions,
        )
        games.append(GeneratedGame(index, seed, result))
    return games


def game_batch_sizes(total_games: int, games_per_task: int):
    return [
        min(games_per_task, total_games - start)
        for start in range(0, total_games, games_per_task)
    ]


def generate_iteration(
    pool,
    *,
    total_games: int,
    games_per_task: int,
    first_game_index: int,
    worker_kwargs: dict,
    progress_interval_seconds: float = 0.0,
) -> list[GeneratedGame]:
    """Collect actual completions; deterministic ordering is restored before replay."""
    sizes = game_batch_sizes(total_games, games_per_task)
    starts = np.cumsum([0, *sizes[:-1]]).tolist()
    completed = queue.Queue()
    progress = experiment_training.GenerationProgress(
        total_games, progress_interval_seconds
    )
    for size, offset in zip(sizes, starts, strict=True):
        pool.apply_async(
            play_games_locally,
            kwds={
                **worker_kwargs,
                "number_of_games": size,
                "first_game_index": first_game_index + offset,
            },
            callback=lambda result: completed.put((True, result)),
            error_callback=lambda error: completed.put((False, error)),
        )
    generated = []
    tasks = 0
    while tasks < len(sizes):
        try:
            succeeded, result = completed.get(timeout=progress.wait_seconds())
        except queue.Empty:
            progress.report()
            continue
        if not succeeded:
            raise result
        tasks += 1
        generated.extend(result)
        progress.games += len(result)
        progress.decisions += sum(
            len(r.history) - 1 for g in result for r in g.result.rounds
        )
        logging.debug("[SELF-PLAY] Completed %s/%s tasks", tasks, len(sizes))
        progress.report(final=tasks == len(sizes))
    return sorted(generated, key=lambda game: game.global_game_index)
