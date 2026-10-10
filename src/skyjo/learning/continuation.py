"""Validate and restore a parent checkpoint into an independent training run."""

from __future__ import annotations

import dataclasses
import hashlib
import io
from pathlib import Path

import numpy as np

from skyjo.learning import checkpoint
from skyjo.learning import models
from skyjo.learning import objectives


def configuration_changes(parent: dict, child: dict, prefix: str = "") -> list[dict]:
    changes = []
    for key in sorted(parent.keys() | child.keys()):
        before, after = parent.get(key), child.get(key)
        name = f"{prefix}{key}"
        if isinstance(before, dict) and isinstance(after, dict):
            changes.extend(configuration_changes(before, after, name + "."))
        elif before != after:
            changes.append({"setting": name, "parent": before, "child": after})
    return changes


@dataclasses.dataclass
class Continuation:
    payload: dict
    provenance: dict
    generated_positions: int | None

    @property
    def progress(self) -> checkpoint.TrainingProgress:
        return checkpoint.TrainingProgress(**self.payload["progress"])

    def next_game(self, replay) -> int:
        cursor = self.payload.get("continuation_state", {}).get("next_game_index", 0)
        if type(cursor) is not int or cursor < 0:
            raise ValueError("Invalid parent next_game_index")
        return max(
            cursor,
            self.progress.generated_games,
            max(replay.game_indices, default=-1) + 1,
        )

    def restore(self, model, optimizer, *, sampling_rng: np.random.Generator) -> None:
        """Keep moments and replay sampling, applying this run's optimizer settings."""
        groups = [
            {key: value for key, value in group.items() if key != "params"}
            for group in optimizer.param_groups
        ]
        checkpoint.restore_checkpoint(
            self.payload,
            model=model,
            optimizer=optimizer,
            restore_rng=False,
            sampling_rng=sampling_rng,
        )
        for group, settings in zip(optimizer.param_groups, groups, strict=True):
            group.update(settings)


def load(
    path: Path, configuration: dict, *, requested_settings: dict | None = None
) -> Continuation:
    """Pin checkpoint bytes and establish seed provenance before creating a run."""
    path = path.resolve()
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    payload = checkpoint.decode_checkpoint(io.BytesIO(raw), map_location="cpu")
    if not payload.get("optimizer_state_dict"):
        raise ValueError("Continuation requires saved optimizer state")
    if payload.get("scheduler_state_dict") is not None:
        raise ValueError("The ordinary runner does not support a parent scheduler")
    if "sampling_rng_state" not in payload:
        raise ValueError("Continuation requires saved replay sampling state")
    if (
        not {"python", "numpy", "torch_cpu", "torch_cuda"}
        <= payload.get("rng_state", {}).keys()
    ):
        raise ValueError("Continuation requires complete saved RNG state")
    parent = payload["configuration"]
    provenance = {"checkpoint_path": str(path), "checkpoint_sha256": digest}
    if parent.get("run_id") is not None:
        provenance["run_id"] = parent["run_id"]
    seed = parent["seed"]
    if seed != configuration["seed"]:
        raise ValueError("Continuation seed must match the parent seed")
    if parent.get("players") != configuration["players"]:
        raise ValueError("Parent player count does not match")
    if models.resolve(parent["model"]) != models.resolve(configuration["model"]):
        raise ValueError("Parent model architecture/dimensions do not match")
    if (
        objectives.resolve(parent.get("auxiliary_objectives")).weights.keys()
        != objectives.resolve(configuration.get("auxiliary_objectives")).weights.keys()
    ):
        raise ValueError("Parent enabled auxiliary heads do not match")
    optimizer_type = parent.get("optimizer", {}).get("type")
    if optimizer_type != "adam":
        raise ValueError("Cannot establish compatible parent Adam optimizer")
    saved = payload["optimizer_state_dict"]
    if not saved.get("param_groups") or any(
        "betas" not in g for g in saved["param_groups"]
    ):
        raise ValueError("Parent optimizer is not compatible with Adam")
    progress = checkpoint.TrainingProgress(**payload["progress"])
    if progress.optimizer_steps > 0 and not saved["state"]:
        raise ValueError("Parent optimizer moments are missing")
    generated_positions = payload.get("continuation_state", {}).get(
        "generated_positions",
    )
    provenance.update(
        seed=seed,
        inherited_progress=dataclasses.asdict(progress),
        inherited_generated_positions=generated_positions,
        parent_configuration=parent,
        configuration_changes=(
            configuration_changes(
                parent["requested"],
                checkpoint.normalize_configuration(requested_settings),
            )
            if "requested" in parent and requested_settings is not None
            else None
        ),
    )
    return Continuation(payload, provenance, generated_positions)
