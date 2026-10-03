"""Versioned, resumable training checkpoints."""

from __future__ import annotations

import dataclasses
import functools
import pathlib
import random
import typing

import numpy as np
import torch

CHECKPOINT_FORMAT = "skyjo.training-checkpoint"
CHECKPOINT_VERSION = 1


class CheckpointFormatError(ValueError):
    """Raised when a file is not a supported Skyjo checkpoint."""


@dataclasses.dataclass(frozen=True, slots=True)
class TrainingProgress:
    iteration: int = 0
    epoch: int = 0
    generated_games: int = 0
    trained_positions: int = 0
    optimizer_steps: int = 0
    sampled_positions: int = 0


def _callable_name(value: typing.Any) -> str:
    return f"{value.__module__}.{value.__qualname__}"


def normalize_configuration(value: typing.Any) -> typing.Any:
    """Convert run configuration into stable, pickle-safe metadata."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: normalize_configuration(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {
            str(key): normalize_configuration(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [normalize_configuration(item) for item in value]
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, functools.partial):
        return {
            "callable": _callable_name(value.func),
            "args": normalize_configuration(value.args),
            "keywords": normalize_configuration(value.keywords or {}),
        }
    if callable(value):
        try:
            return {"callable": _callable_name(value)}
        except AttributeError:
            return {"callable": repr(value)}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def capture_rng_state() -> dict[str, typing.Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else [],
    }


def restore_rng_state(state: dict[str, typing.Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_checkpoint(
    path: pathlib.Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    configuration: typing.Any = None,
    progress: TrainingProgress | None = None,
    sampling_rng: np.random.Generator | None = None,
) -> pathlib.Path:
    """Atomically save all state needed to resume a training run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        "rng_state": capture_rng_state(),
        "configuration": normalize_configuration(configuration),
        "progress": dataclasses.asdict(progress or TrainingProgress()),
    }
    if sampling_rng is not None:
        payload["sampling_rng_state"] = sampling_rng.bit_generator.state
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)
    return path


def load_checkpoint(
    path: pathlib.Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    expected_configuration: typing.Any = None,
    restore_rng: bool = True,
    sampling_rng: np.random.Generator | None = None,
    map_location: torch.device | str | None = None,
) -> TrainingProgress:
    """Load a strict Skyjo checkpoint into supplied runtime objects."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise CheckpointFormatError(
            f"{path} is not a {CHECKPOINT_FORMAT} file; raw state_dict files are unsupported"
        )
    if payload.get("version") != CHECKPOINT_VERSION:
        raise CheckpointFormatError(
            f"unsupported checkpoint version {payload.get('version')!r}; expected {CHECKPOINT_VERSION}"
        )

    actual_configuration = payload.get("configuration")
    if expected_configuration is not None and actual_configuration is not None:
        expected = normalize_configuration(expected_configuration)
        if actual_configuration != expected:
            raise ValueError(
                "checkpoint configuration does not match the active run: "
                f"checkpoint={actual_configuration!r}, active={expected!r}"
            )

    model.load_state_dict(payload["model_state_dict"])
    optimizer_state = payload.get("optimizer_state_dict")
    if optimizer is not None and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    scheduler_state = payload.get("scheduler_state_dict")
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
    if sampling_rng is not None:
        if "sampling_rng_state" not in payload:
            raise CheckpointFormatError(
                "Checkpoint has no independent sampling RNG state"
            )
        sampling_rng.bit_generator.state = payload["sampling_rng_state"]
    if restore_rng:
        restore_rng_state(payload["rng_state"])
    return TrainingProgress(**payload["progress"])
