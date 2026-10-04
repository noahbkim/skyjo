"""Seat-balanced full-game checkpoint comparisons, independent of training."""

from __future__ import annotations

import dataclasses
import pathlib
import random
import time
from collections.abc import Callable

import numpy as np
import torch

from . import checkpoint, play, player, predictor, runs


@dataclasses.dataclass(frozen=True)
class EvaluationConfig:
    seed_count: int = 32
    seed: int = 0
    iterations: int = 128
    control_iterations: int | None = None
    variant_iterations: int | None = None

    def __post_init__(self):
        for name in ("seed_count", "iterations"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"evaluation.{name} must be a positive integer")
        for name in ("control_iterations", "variant_iterations"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"evaluation.{name} must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("evaluation.seed must fit a uint32")

    def search(self, *, iterations: int | None = None):
        return player.ModelPlayerConfig(
            action_softmax_temperature=0.0,
            mcts_iterations=self.iterations if iterations is None else iterations,
            mcts_dirichlet_epsilon=0.0,
            mcts_after_state_evaluate_all_children=False,
            mcts_c_puct=1.0,
            mcts_fpu_reduction=0.0,
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


def evaluate_checkpoints(
    control: pathlib.Path,
    variant: pathlib.Path,
    settings: EvaluationConfig = EvaluationConfig(),
    *,
    on_game: Callable[[dict], None] | None = None,
) -> dict:
    """Compare checkpoints with common seeds and both seats; restore caller RNGs."""
    rng_state = checkpoint.capture_rng_state()
    started = time.perf_counter()
    try:
        identities = {
            name: {"path": str(path.resolve()), "sha256": runs.file_digest(path)}
            for name, path in (("control", control), ("variant", variant))
        }
        search_by_player = {
            "control": settings.search(iterations=settings.control_iterations).kwargs(),
            "variant": settings.search(iterations=settings.variant_iterations).kwargs(),
        }
        agents = {
            name: player.ModelPlayer(
                predictor.LocalPredictor(load_model(path), max_batch_size=1),
                **search_by_player[name],
            )
            for name, path in (("control", control), ("variant", variant))
        }
        records = []
        for index in range(settings.seed_count):
            seed = int(
                np.random.SeedSequence(
                    [settings.seed, index, 0x4556414C]
                ).generate_state(1)[0]
            )
            for seats in (("control", "variant"), ("variant", "control")):
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                result = play.play_game([agents[name] for name in seats])
                variant_seat = seats.index("variant")
                scores = result.final_scores
                records.append(
                    {
                        "seed": seed,
                        "seed_index": index,
                        "seats": list(seats),
                        "checkpoints": identities,
                        "search_by_player": search_by_player,
                        "cumulative_scores": list(scores),
                        "variant_win_credit": (
                            1.0 / len(result.winners)
                            if variant_seat in result.winners
                            else 0.0
                        ),
                        "control_minus_variant": scores[1 - variant_seat]
                        - scores[variant_seat],
                    }
                )
                if on_game is not None:
                    on_game(records[-1])
        return {
            "settings": dataclasses.asdict(settings),
            "search": settings.search().kwargs(),
            "search_by_player": search_by_player,
            "checkpoints": identities,
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
        checkpoint.restore_rng_state(rng_state)
