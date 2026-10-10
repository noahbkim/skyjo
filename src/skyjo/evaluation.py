"""Seat-balanced full-game checkpoint comparisons, independent of training."""

from __future__ import annotations

import dataclasses
import multiprocessing as mp
import pathlib
import random
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np
import torch

from . import boundary_inference, checkpoint, play, player, predictor, runs


@dataclasses.dataclass(frozen=True)
class EvaluationConfig:
    seed_count: int = 32
    seed: int = 0
    iterations: int = 128
    control_iterations: int | None = None
    variant_iterations: int | None = None
    control_boundary_samples: int = 1
    variant_boundary_samples: int = 1
    control_boundary_value_checkpoint: str | None = None
    variant_boundary_value_checkpoint: str | None = None
    control_policy_only: bool = False
    variant_policy_only: bool = False
    control_merge_symmetric_actions: bool = True
    variant_merge_symmetric_actions: bool = True

    def __post_init__(self):
        for name in (
            "seed_count", "iterations", "control_boundary_samples", "variant_boundary_samples"
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"evaluation.{name} must be a positive integer")
        for name in ("control_iterations", "variant_iterations"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"evaluation.{name} must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("evaluation.seed must fit a uint32")
        for side in ("control", "variant"):
            merge_name = f"{side}_merge_symmetric_actions"
            if type(getattr(self, merge_name)) is not bool:
                raise ValueError(f"evaluation.{merge_name} must be a boolean")
            policy_only = getattr(self, f"{side}_policy_only")
            if type(policy_only) is not bool:
                raise ValueError(f"evaluation.{side}_policy_only must be a boolean")
            name = f"{side}_boundary_value_checkpoint"
            value = getattr(self, name)
            if policy_only and (
                value is not None or getattr(self, f"{side}_boundary_samples") != 1
            ):
                raise ValueError(
                    f"evaluation.{side}_policy_only cannot use boundary settings"
                )
            if value is None:
                if getattr(self, f"{side}_boundary_samples") != 1:
                    raise ValueError(f"evaluation.{side}_boundary_samples > 1 requires {name}")
            else:
                if not isinstance(value, (str, pathlib.Path)) or not str(value).strip():
                    raise ValueError(f"evaluation.{name} must be a nonempty path")
                path = pathlib.Path(value).resolve()
                if not path.is_file():
                    raise FileNotFoundError(path)
                object.__setattr__(self, name, str(path))

    def search(
        self,
        *,
        iterations: int | None = None,
        boundary_samples: int = 1,
        boundary_value_checkpoint: str | None = None,
        merge_symmetric_actions: bool = True,
    ):
        return player.ModelPlayerConfig(
            action_softmax_temperature=0.0,
            mcts_iterations=self.iterations if iterations is None else iterations,
            mcts_dirichlet_epsilon=0.0,
            mcts_after_state_evaluate_all_children=False,
            mcts_c_puct=1.0,
            mcts_fpu_reduction=0.0,
            mcts_boundary_samples=boundary_samples,
            mcts_boundary_value_checkpoint=boundary_value_checkpoint,
            mcts_merge_symmetric_actions=merge_symmetric_actions,
        )


def load_model(path: pathlib.Path):
    """Reconstruct the ordinary runner's model from its versioned checkpoint."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (payload.get("format"), payload.get("version")) != (
        checkpoint.CHECKPOINT_FORMAT,
        checkpoint.CHECKPOINT_VERSION,
    ):
        raise checkpoint.CheckpointFormatError(
            "Expected a versioned training checkpoint"
        )
    configuration = payload["configuration"]
    players = configuration.get("players", 2)
    if players != 2:
        raise ValueError("Checkpoint evaluation currently requires two players")
    settings = dict(configuration["model"])
    from . import models

    settings.pop("non_spatial_input_shape", None)
    auxiliary = settings.pop(
        "auxiliary_objectives", configuration.get("auxiliary_objectives", {})
    )
    model = models.build(
        settings, players=players, device="cpu", auxiliary_objectives=auxiliary
    )
    checkpoint.load_checkpoint(path, model=model, restore_rng=False, map_location="cpu")
    model.eval()
    return model


def _build_agents(control, variant, search_by_player):
    for search in search_by_player.values():
        if search is None:
            continue
        path = search["mcts_boundary_value_checkpoint"]
        if path is not None:
            boundary_inference.load_boundary_model(path, players=2)
    agents = {}
    for name, path in (("control", control), ("variant", variant)):
        inference = predictor.LocalPredictor(load_model(path), max_batch_size=1)
        search = search_by_player[name]
        agents[name] = (
            player.PolicyPlayer(inference)
            if search is None
            else player.ModelPlayer(inference, **search)
        )
    return agents


def _play_game(agents, job):
    index, seed, seats = job
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    result = play.play_game([agents[name] for name in seats])
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


def _initialize_worker(control, variant, search_by_player, threads):
    global _worker_agents, _worker_inputs
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    _worker_agents = None
    _worker_inputs = (control, variant, search_by_player)


def _play_worker_game(job):
    global _worker_agents
    if _worker_agents is None:
        # Load once per worker and report loading errors through the task future.
        _worker_agents = _build_agents(*_worker_inputs)
    return _play_game(_worker_agents, job)


def evaluate_checkpoints(
    control: pathlib.Path,
    variant: pathlib.Path,
    settings: EvaluationConfig = EvaluationConfig(),
    *,
    on_game: Callable[[dict], None] | None = None,
    workers: int = 1,
    threads: int = 1,
) -> dict:
    """Compare both seats, restoring caller RNGs and Torch thread count.

    Callbacks run in the parent as games finish; returned games follow seed order.
    """
    for name, value in (("workers", workers), ("threads", threads)):
        if type(value) is not int or value < 1:
            raise ValueError(f"evaluation.{name} must be a positive integer")
    rng_state = checkpoint.capture_rng_state()
    previous_threads = torch.get_num_threads()
    started = time.perf_counter()
    try:
        torch.set_num_threads(threads)
        control, variant = pathlib.Path(control).resolve(), pathlib.Path(variant).resolve()
        identities = {
            name: {"path": str(path.resolve()), "sha256": runs.file_digest(path)}
            for name, path in (("control", control), ("variant", variant))
        }
        search_by_player = {
            name: (
                None
                if getattr(settings, f"{name}_policy_only")
                else settings.search(
                    iterations=getattr(settings, f"{name}_iterations"),
                    merge_symmetric_actions=getattr(
                        settings, f"{name}_merge_symmetric_actions"
                    ),
                    boundary_samples=getattr(settings, f"{name}_boundary_samples"),
                    boundary_value_checkpoint=getattr(
                        settings, f"{name}_boundary_value_checkpoint"
                    ),
                ).kwargs()
            )
            for name in ("control", "variant")
        }
        play_mode_by_player = {
            name: "policy" if search is None else "mcts"
            for name, search in search_by_player.items()
        }
        boundary_identities = {}
        for name, search in search_by_player.items():
            path = None if search is None else search["mcts_boundary_value_checkpoint"]
            if path is None:
                boundary_identities[name] = None
            else:
                boundary_identities[name] = {
                    "path": path, "sha256": runs.file_digest(pathlib.Path(path))
                }
        # Fail on invalid checkpoints/evaluators before starting any processes.
        agents = _build_agents(control, variant, search_by_player)
        jobs = []
        for index in range(settings.seed_count):
            seed = int(
                np.random.SeedSequence(
                    [settings.seed, index, 0x4556414C]
                ).generate_state(1)[0]
            )
            for seats in (("control", "variant"), ("variant", "control")):
                jobs.append((index, seed, seats))
        records = []

        def collect(completed):
            for record in completed:
                record.update(
                    checkpoints=identities,
                    boundary_value_checkpoints=boundary_identities,
                    search_by_player=search_by_player,
                    play_mode_by_player=play_mode_by_player,
                )
                records.append(record)
                if on_game is not None:
                    on_game(record)

        effective_workers = min(workers, len(jobs))
        if effective_workers == 1:
            collect(_play_game(agents, job) for job in jobs)
        else:
            del agents
            executor = ProcessPoolExecutor(
                max_workers=effective_workers,
                mp_context=mp.get_context("spawn"),
                initializer=_initialize_worker,
                initargs=(control, variant, search_by_player, threads),
            )
            remaining = iter(jobs)
            pending = set()
            try:
                for _ in range(effective_workers):
                    pending.add(executor.submit(_play_worker_game, next(remaining)))
                while pending:
                    finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in finished:
                        collect((future.result(),))
                    for _ in finished:
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
        return {
            "settings": dataclasses.asdict(settings),
            "execution": {
                "requested_workers": workers,
                "effective_workers": effective_workers,
                "threads_per_worker": threads,
            },
            "search": settings.search().kwargs(),
            "search_by_player": search_by_player,
            "play_mode_by_player": play_mode_by_player,
            "checkpoints": identities,
            "boundary_value_checkpoints": boundary_identities,
            "games": records,
            "variant_win_fraction": float(
                np.mean([r["variant_win_credit"] for r in records])
            ),
            "control_minus_variant_margin": float(
                np.mean([r["control_minus_variant"] for r in records])
            ),
            "evaluation_seconds": time.perf_counter() - started,
        }
    finally:
        torch.set_num_threads(previous_threads)
        checkpoint.restore_rng_state(rng_state)
