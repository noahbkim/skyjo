"""Seat-balanced full-game checkpoint comparisons, independent of training."""

from __future__ import annotations

import dataclasses
import pathlib
import random
import time

import numpy as np
import torch

from . import checkpoint, game, play, player, predictor, runs, skynet


@dataclasses.dataclass(frozen=True)
class EvaluationConfig:
    seed_count: int = 32
    seed: int = 0
    iterations: int = 128

    def __post_init__(self):
        for name in ("seed_count", "iterations"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"evaluation.{name} must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("evaluation.seed must fit a uint32")

    def search(self):
        return player.ModelPlayerConfig(
            action_softmax_temperature=0.0,
            mcts_iterations=self.iterations,
            mcts_dirichlet_epsilon=0.0,
            mcts_after_state_evaluate_all_children=False,
            mcts_c_puct=1.0,
            mcts_fpu_reduction=0.0,
            mcts_score_utility_weight=0.0,
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
) -> dict:
    """Compare checkpoints with common seeds and both seats; restore caller RNGs."""
    rng_state = checkpoint.capture_rng_state()
    started = time.perf_counter()
    try:
        identities = {
            name: {"path": str(path.resolve()), "sha256": runs.file_digest(path)}
            for name, path in (("control", control), ("variant", variant))
        }
        agents = {
            name: player.ModelPlayer(
                predictor.LocalPredictorClient(load_model(path), max_batch_size=1),
                **settings.search().kwargs(),
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
        return {
            "settings": dataclasses.asdict(settings),
            "search": settings.search().kwargs(),
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
