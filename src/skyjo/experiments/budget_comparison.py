"""Preparation and historical checkpoint selection for the search-cap pilot."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from skyjo.learning import checkpoint, observations


def adapt_parent(source: Path, destination: Path, *, sampling_seed: int) -> dict:
    """Explicitly adapt old metadata; never claim an exact legacy RNG resume.

    Parameters, Adam moments, counters and existing RNG states are preserved.
    The independent replay sampler is seeded only when absent in the source.
    """
    payload = checkpoint.decode_checkpoint(source)
    config = payload["configuration"]
    model = config["model"]
    shape = model.pop("non_spatial_input_shape", None)
    if shape is not None and tuple(shape) != observations.get_non_spatial_input_shape(
        config["players"]
    ):
        raise ValueError("Legacy parent has an incompatible observation shape")
    legacy_heads = model.pop("auxiliary_objectives", {})
    if (
        legacy_heads
        and config.get("auxiliary_objectives", legacy_heads) != legacy_heads
    ):
        raise ValueError("Conflicting legacy auxiliary heads")
    config.setdefault("auxiliary_objectives", legacy_heads)
    model.setdefault("policy_normalization", "actions")
    reset = "sampling_rng_state" not in payload
    if reset:
        payload["sampling_rng_state"] = np.random.default_rng(
            sampling_seed
        ).bit_generator.state
    metadata = {
        "source": str(source.resolve()),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "replay_sampling_rng_initialized": reset,
        "replay_sampling_seed": sampling_seed if reset else None,
        "interpretation": "Shared starting state for both new arms; historical execution is not reproduced exactly.",
    }
    payload["adaptation"] = metadata
    torch.save(payload, destination)
    return metadata


def group_initial_checkpoint(adapted: Path, destination: Path) -> None:
    payload = copy.deepcopy(checkpoint.decode_checkpoint(adapted))
    payload["configuration"]["model"]["policy_normalization"] = "groups"
    torch.save(payload, destination)


def historical_checkpoint(run: Path, seconds: float) -> dict:
    """Choose by elapsed training time, before looking at any match results."""
    candidates = []
    for line in (run / "trajectory.jsonl").read_text().splitlines():
        event = json.loads(line)
        if event["kind"] != "iteration_completed":
            continue
        iteration = event["progress"]["iteration"]
        path = run / "checkpoints" / f"checkpoint_{iteration:06d}.pth"
        if path.is_file():
            candidates.append(
                {
                    "checkpoint": str(path),
                    "progress": event["progress"],
                    "seconds": event["metrics"]["time/run_seconds"],
                }
            )
    if not candidates:
        raise ValueError("Historical run has no timed, available checkpoints")
    return min(candidates, key=lambda c: abs(c["seconds"] - seconds))
