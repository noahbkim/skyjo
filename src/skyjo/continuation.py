"""Validate and restore a parent checkpoint into an independent training run."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import io
import json
from pathlib import Path

import torch

from . import checkpoint, models, objectives


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

    def restore(self, model, optimizer) -> None:
        """Keep moments/steps, but apply this invocation's optimizer settings."""
        model.load_state_dict(self.payload["model_state_dict"], strict=True)
        groups = [
            {k: v for k, v in group.items() if k != "params"}
            for group in optimizer.param_groups
        ]
        saved = self.payload["optimizer_state_dict"]
        if len(saved["param_groups"]) != len(groups):
            raise ValueError("Parent optimizer parameter groups do not match")
        optimizer.load_state_dict(copy.deepcopy(saved))
        for group, settings in zip(optimizer.param_groups, groups):
            group.update(settings)
            for parameter in group["params"]:
                state = optimizer.state.get(parameter, {})
                if state and (
                    not {"step", "exp_avg", "exp_avg_sq"} <= state.keys()
                    or any(
                        state[key].shape != parameter.shape
                        for key in ("exp_avg", "exp_avg_sq")
                    )
                ):
                    raise ValueError(
                        "Parent optimizer state is incompatible with Adam/model"
                    )


def load(path: Path, configuration: dict) -> Continuation:
    """Pin checkpoint bytes and establish seed provenance before creating a run."""
    path = path.resolve()
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or (
        payload.get("format"),
        payload.get("version"),
    ) != (checkpoint.CHECKPOINT_FORMAT, checkpoint.CHECKPOINT_VERSION):
        raise checkpoint.CheckpointFormatError(
            "Continuation requires a versioned training checkpoint"
        )
    if not payload.get("optimizer_state_dict"):
        raise ValueError("Continuation requires saved optimizer state")
    if payload.get("scheduler_state_dict") is not None:
        raise ValueError("The ordinary runner does not support a parent scheduler")
    if "sampling_rng_state" in payload:
        raise ValueError(
            "Continuation requires an ordinary self-play checkpoint, not an offline sampler"
        )
    if (
        not {"python", "numpy", "torch_cpu", "torch_cuda"}
        <= payload.get("rng_state", {}).keys()
    ):
        raise ValueError("Continuation requires complete saved RNG state")
    parent = payload["configuration"]
    provenance = {"checkpoint_path": str(path), "checkpoint_sha256": digest}
    parent_config, artifact = None, None
    directory = path.parent.parent
    if (directory / "artifacts.jsonl").is_file():
        entries = [
            json.loads(line)
            for line in (directory / "artifacts.jsonl").read_text().splitlines()
        ]
        artifact = next(
            (
                a
                for a in reversed(entries)
                if a.get("artifact_kind") == "checkpoint"
                and (directory / a["path"]).resolve() == path
                and a.get("sha256") == digest
            ),
            None,
        )
        if artifact is not None:
            manifest = json.loads((directory / "run.json").read_text())
            parent_config = json.loads((directory / "resolved-config.json").read_text())
            config_digest = hashlib.sha256(
                json.dumps(parent_config, sort_keys=True, allow_nan=False).encode()
            ).hexdigest()
            if config_digest != manifest["config_sha256"]:
                raise ValueError("Parent run configuration provenance does not match")
            provenance.update(
                run_id=manifest["run_id"], artifact_id=artifact["artifact_id"]
            )
    seed = parent.get("seed", (parent_config or {}).get("seed"))
    if seed is None:
        raise ValueError(
            "Cannot establish parent seed; retain the checkpoint's recorded run files"
        )
    if seed != configuration["seed"]:
        raise ValueError("Continuation seed must match the parent seed")
    if parent.get("players") != configuration["players"]:
        raise ValueError("Parent player count does not match")
    model_settings = dict(parent["model"])
    auxiliary = model_settings.pop(
        "auxiliary_objectives", parent.get("auxiliary_objectives", {})
    )
    model_settings.pop("non_spatial_input_shape", None)
    if models.resolve(model_settings) != configuration["model"]:
        raise ValueError("Parent model architecture/dimensions do not match")
    if (
        objectives.resolve(auxiliary).weights.keys()
        != configuration["auxiliary_objectives"].keys()
    ):
        raise ValueError("Parent enabled auxiliary heads do not match")
    optimizer_type = parent.get(
        "optimizer", (parent_config or {}).get("derived", {}).get("optimizer", {})
    ).get("type")
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
        (artifact or {}).get("progress", {}).get("generated_positions"),
    )
    provenance.update(
        seed=seed,
        inherited_progress=dataclasses.asdict(progress),
        inherited_generated_positions=generated_positions,
        parent_configuration=parent_config or parent,
        configuration_changes=configuration_changes(
            parent_config or parent, configuration
        ),
    )
    return Continuation(payload, provenance, generated_positions)
