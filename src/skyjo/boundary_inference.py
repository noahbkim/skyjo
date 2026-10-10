"""Frozen score-based values for completed rounds, before the next deal."""

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from . import game as sj
from . import skynet
from .boundary_value import BoundaryValueModel


def load_boundary_model(checkpoint: str | Path, players: int) -> BoundaryValueModel:
    """Load a CPU evaluator once per process and file version."""
    path = Path(checkpoint).resolve()
    stat = path.stat()
    return _load_model(str(path), players, stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=8)
def _load_model(path: str, players: int, modified: int, size: int) -> BoundaryValueModel:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or (
        payload.get("format"), payload.get("version")
    ) != ("skyjo.boundary-value", 1):
        raise ValueError("Expected a version 1 score boundary checkpoint")
    if payload.get("players") != players:
        raise ValueError("Boundary checkpoint player count does not match")
    if payload.get("score_scale") != 100.0 or payload.get("input_order") != (
        "next starter first, then cyclic seat order"
    ):
        raise ValueError("Unsupported boundary checkpoint score encoding")
    # Construction must not perturb the gameplay/training random stream.
    with torch.random.fork_rng(devices=[]):
        model = BoundaryValueModel(payload["kind"], players, payload["hidden_width"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ValueError("Boundary checkpoint contains nonfinite weights")
    model.eval().requires_grad_(False)
    return model


@torch.inference_mode()
def predict_completed_rounds(
    model: BoundaryValueModel | None, states: Sequence[sj.Skyjo]
) -> np.ndarray:
    """Evaluate each sampled ending, retaining terminal mass and absolute seats.

    Completed-state current-player order is next-starter order. The engine's
    score accessor includes the newly charged round exactly once. Full-game
    outcomes bypass the learned model, including fractional credit for ties.
    """
    if not states:
        raise ValueError("At least one completed round is required")
    players = sj.get_player_count(states[0])
    values = np.empty((len(states), players), dtype=np.float32)
    continuing = []
    scores = []
    for index, state in enumerate(states):
        if sj.get_player_count(state) != players or not sj.get_round_over(state):
            raise ValueError("Expected completed rounds with matching player counts")
        if sj.get_game_over(state):
            values[index] = skynet.skyjo_to_game_state_value(state)
        else:
            continuing.append(index)
            scores.append(sj.get_game_scores(state))
    if continuing:
        if model is None or model.players != players:
            raise ValueError("Continuing rounds require a matching boundary model")
        predictions = model(torch.as_tensor(np.asarray(scores), dtype=torch.float32)).numpy()
        for index, prediction in zip(continuing, predictions, strict=True):
            values[index] = skynet.to_state_value(prediction, sj.get_player(states[index]))
    return values
