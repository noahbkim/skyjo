"""Versioned, resumable training checkpoints."""

from __future__ import annotations

import dataclasses
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
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"Unsupported configuration value: {type(value).__name__}")


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
    continuation_state: dict | None = None,
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
    if continuation_state is not None:
        payload["continuation_state"] = continuation_state
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)
    return path


def decode_checkpoint(source, *, map_location="cpu") -> dict[str, typing.Any]:
    """Read the single training-checkpoint format without changing runtime state."""
    payload = torch.load(source, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise CheckpointFormatError(
            f"{source} is not a {CHECKPOINT_FORMAT} file; raw state_dict files are unsupported"
        )
    if payload.get("version") != CHECKPOINT_VERSION:
        raise CheckpointFormatError(
            f"unsupported checkpoint version {payload.get('version')!r}; expected {CHECKPOINT_VERSION}"
        )

    for name in ("model_state_dict", "configuration", "progress"):
        if name not in payload:
            raise CheckpointFormatError(f"Checkpoint is missing required {name}")
    return payload


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
    payload = decode_checkpoint(path, map_location=map_location)
    return restore_checkpoint(
        payload,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        expected_configuration=expected_configuration,
        restore_rng=restore_rng,
        sampling_rng=sampling_rng,
    )


def restore_checkpoint(
    payload: dict[str, typing.Any],
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    expected_configuration: typing.Any = None,
    restore_rng: bool = True,
    sampling_rng: np.random.Generator | None = None,
) -> TrainingProgress:
    """Restore a decoded checkpoint, validating requested state before mutation."""
    actual_configuration = payload.get("configuration")
    if expected_configuration is not None:
        expected = normalize_configuration(expected_configuration)
        if actual_configuration != expected:
            raise ValueError(
                "checkpoint configuration does not match the active run: "
                f"checkpoint={actual_configuration!r}, active={expected!r}"
            )

    # Validate requested state before changing the model, optimizer, or RNGs.
    required = ["model_state_dict", "progress"]
    if optimizer is not None:
        required.append("optimizer_state_dict")
    if scheduler is not None:
        required.append("scheduler_state_dict")
    if restore_rng:
        required.append("rng_state")
    if sampling_rng is not None:
        required.append("sampling_rng_state")
    for name in required:
        if payload.get(name) is None:
            raise CheckpointFormatError(f"Checkpoint is missing required {name}")
    try:
        if not isinstance(payload["model_state_dict"], dict):
            raise ValueError("model state must be a mapping")
        expected_model = model.state_dict()
        saved_model = payload["model_state_dict"]
        if saved_model.keys() != expected_model.keys() or any(
            not isinstance(saved_model[name], torch.Tensor)
            or saved_model[name].shape != expected.shape
            for name, expected in expected_model.items()
        ):
            raise ValueError("model parameters do not match checkpoint")
        if optimizer is not None:
            saved = payload["optimizer_state_dict"]
            if not isinstance(saved, dict) or not isinstance(saved.get("state"), dict):
                raise ValueError("invalid optimizer state")
            groups = saved.get("param_groups")
            if not isinstance(groups, list) or len(groups) != len(
                optimizer.param_groups
            ):
                raise ValueError("optimizer parameter groups do not match")
            for actual, expected in zip(groups, optimizer.param_groups, strict=True):
                if not isinstance(actual, dict) or not isinstance(
                    actual.get("params"), list
                ):
                    raise ValueError("invalid optimizer parameter group")
                if len(actual["params"]) != len(expected["params"]):
                    raise ValueError("optimizer parameter counts do not match")
                if isinstance(optimizer, torch.optim.Adam):
                    for index, parameter in zip(
                        actual["params"], expected["params"], strict=True
                    ):
                        state = saved["state"].get(index, {})
                        if state and (
                            not {"step", "exp_avg", "exp_avg_sq"} <= state.keys()
                            or any(
                                state[key].shape != parameter.shape
                                for key in ("exp_avg", "exp_avg_sq")
                            )
                        ):
                            raise ValueError(
                                "Adam moments do not match model parameters"
                            )
        if scheduler is not None:
            saved = payload["scheduler_state_dict"]
            if (
                not isinstance(saved, dict)
                or scheduler.state_dict().keys() - saved.keys()
            ):
                raise ValueError("incomplete scheduler state")
        progress = TrainingProgress(**payload["progress"])
        if any(
            type(value) is not int or value < 0
            for value in dataclasses.asdict(progress).values()
        ):
            raise ValueError("progress counters must be nonnegative integers")
        if restore_rng:
            rng = payload["rng_state"]
            random.Random().setstate(rng["python"])
            np.random.RandomState().set_state(rng["numpy"])
            torch.Generator().set_state(rng["torch_cpu"].cpu())
            if not isinstance(rng["torch_cuda"], list):
                raise ValueError("CUDA RNG state must be a list")
            for state in rng["torch_cuda"]:
                if (
                    not isinstance(state, torch.Tensor)
                    or state.dtype != torch.uint8
                    or state.ndim != 1
                ):
                    raise ValueError("invalid CUDA RNG state")
        if sampling_rng is not None:
            probe = type(sampling_rng.bit_generator)()
            probe.state = payload["sampling_rng_state"]
    except (KeyError, TypeError, ValueError, RuntimeError, AttributeError) as error:
        raise CheckpointFormatError(f"Invalid checkpoint metadata: {error}") from error

    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    if sampling_rng is not None:
        sampling_rng.bit_generator.state = payload["sampling_rng_state"]
    if restore_rng:
        restore_rng_state(payload["rng_state"])
    return progress
