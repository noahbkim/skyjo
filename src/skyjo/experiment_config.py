"""Plain-data configuration for the current distributed experiment recipe."""

from __future__ import annotations

import copy
import json
import math
import pathlib
import tomllib

import torch

from . import buffer, game, models, objectives, observations

DEFAULTS = {
    "name": "full-game-baseline",
    "description": "",
    "tags": [],
    "notes": "",
    "seed": 0,
    "players": 2,
    "initial_checkpoint": None,
    "auxiliary_objectives": {name: 0.0 for name in objectives.REGISTRY},
    "auxiliary_targets": {"mode": "observed", "samples": 32},
    "experiment": {"name": "", "variant": "", "suite_run_id": ""},
    "model": {
        "embedding_dimensions": 16,
        "global_state_embedding_dimensions": 32,
        "num_heads": 2,
    },
    "training": {
        "batch_size": 256,
        "replay_ratio": 4.0,
        "replay_ratio_after_fill": None,
        "learn_rate": 0.001,
        "value_scale": 1.0,
        "policy_scale": 1.0,
        "gradient_diagnostic": False,
    },
    "selfplay": {
        "games_per_iteration": 1024,
        "games_per_task": 8,
        "start_state": "standard",
    },
    "search": {
        "iterations": 100,
        "dirichlet_epsilon": 0.25,
        "after_state_evaluate_all_children": False,
        "c_puct": 1.0,
        "fpu_reduction": 0.25,
        "action_softmax_temperature": 1.0,
        "boundary_samples": 1,
        "boundary_value_checkpoint": None,
        "merge_symmetric_actions": True,
    },
    "replay": {
        "capacity": 2_000_000,
        "initial_dataset": None,
        "dataset_id": None,
    },
    "budget": {"iterations": 10, "max_seconds": 0.0, "checkpoint_interval": 1},
    "logging": {"progress_interval_seconds": 0.0},
    "validation": {"concept_interval": 5},
    "execution": {
        "device": "cpu",
        "workers": 8,
        "threads_per_worker": 1,
        "debug": False,
    },
}


def _merge(defaults: dict, supplied: dict, prefix: str = "") -> dict:
    if not isinstance(supplied, dict):
        raise ValueError(f"{prefix or 'configuration'} must be a table")
    unknown = supplied.keys() - defaults.keys()
    if unknown:
        raise ValueError(
            f"Unknown settings in {prefix or 'configuration'}: {sorted(unknown)}"
        )
    result = copy.deepcopy(defaults)
    for key, value in supplied.items():
        expected = defaults[key]
        label = f"{prefix}{key}"
        if isinstance(expected, dict):
            result[key] = _merge(expected, value, label + ".")
            continue
        if expected is not None:
            valid = (
                type(value) in (int, float)
                if isinstance(expected, float)
                else type(value) is type(expected)
            )
            if not valid:
                raise ValueError(f"Invalid type for {label}")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{label} must be finite")
        result[key] = value
    return result


def resolve_paths(config: dict, directory: pathlib.Path) -> dict:
    """Resolve path-valued settings at their declaration, including partial overrides."""
    config = copy.deepcopy(config)
    if config.get("initial_checkpoint") is not None:
        if (
            not isinstance(config["initial_checkpoint"], str)
            or not config["initial_checkpoint"]
        ):
            raise ValueError("initial_checkpoint must be a nonempty path string")
        config["initial_checkpoint"] = str(
            (directory / config["initial_checkpoint"]).resolve()
        )
    replay = config.get("replay", {})
    if isinstance(replay, dict) and replay.get("initial_dataset") is not None:
        if not isinstance(replay["initial_dataset"], str):
            raise ValueError("replay.initial_dataset must be a path string")
        replay["initial_dataset"] = str(
            (directory / replay["initial_dataset"]).resolve()
        )
    search = config.get("search", {})
    if isinstance(search, dict) and search.get("boundary_value_checkpoint") is not None:
        checkpoint = search["boundary_value_checkpoint"]
        if not isinstance(checkpoint, str) or not checkpoint:
            raise ValueError(
                "search.boundary_value_checkpoint must be a nonempty path string"
            )
        search["boundary_value_checkpoint"] = str((directory / checkpoint).resolve())
    return config


def configuration_sources(path: pathlib.Path) -> tuple[dict, list[dict]]:
    """Load inheritance before defaults, retaining declaring-file provenance."""
    import hashlib

    def read(current, ancestors):
        current = current.resolve()
        if current in ancestors:
            raise ValueError(f"Configuration inheritance cycle: {current}")
        raw = current.read_bytes()
        if current.suffix == ".toml":
            supplied = tomllib.loads(raw.decode("utf-8"))
        elif current.suffix == ".json":
            supplied = json.loads(raw)
        else:
            raise ValueError("Configuration must be .toml or .json")
        if not isinstance(supplied, dict):
            raise ValueError("Configuration must be an object")
        parent = supplied.pop("extends", None)
        base, sources = {}, []
        if parent is not None:
            if not isinstance(parent, str) or not parent:
                raise ValueError("extends must be a configuration path")
            base, sources = read(current.parent / parent, (*ancestors, current))
        sources.append(
            {
                "path": str(current),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "content": raw.decode("utf-8"),
            }
        )
        return overlay(base, resolve_paths(supplied, current.parent)), sources

    return read(path, ())


def overlay(base: dict, overrides: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        result[key] = (
            overlay(result[key], value)
            if isinstance(value, dict) and isinstance(result.get(key), dict)
            else copy.deepcopy(value)
        )
    return result


def load_configuration(path: pathlib.Path) -> tuple[bytes, dict]:
    supplied, sources = configuration_sources(path)
    return sources[-1]["content"].encode(), resolve_configuration(
        supplied, base_directory=path.parent
    )


def resolve_configuration(supplied: dict, *, base_directory: pathlib.Path) -> dict:
    supplied = copy.deepcopy(supplied)
    if not isinstance(supplied, dict):
        raise ValueError("Configuration must be an object")
    prior_derived = supplied.pop("derived", None)
    model_settings = models.resolve(supplied.pop("model", {}))
    config = resolve_paths(_merge(DEFAULTS, supplied), base_directory)
    config["model"] = model_settings
    config["auxiliary_objectives"] = objectives.resolve(
        config["auxiliary_objectives"]
    ).weights
    if config["auxiliary_targets"]["mode"] not in ("observed", "resampled"):
        raise ValueError("auxiliary_targets.mode must be observed or resampled")
    if config["auxiliary_targets"]["samples"] < 1:
        raise ValueError("auxiliary_targets.samples must be positive")
    training, replay = (config[key] for key in ("training", "replay"))
    after_fill = training["replay_ratio_after_fill"]
    if after_fill is not None and (
        type(after_fill) not in (int, float)
        or not math.isfinite(after_fill)
        or after_fill <= 0
    ):
        raise ValueError("training.replay_ratio_after_fill must be finite and positive")
    for section, keys in {
        "training": (
            "batch_size",
            "replay_ratio",
            "learn_rate",
        ),
        "selfplay": ("games_per_iteration", "games_per_task"),
        "search": ("iterations", "boundary_samples"),
        "replay": ("capacity",),
        "budget": ("checkpoint_interval",),
        "execution": ("workers", "threads_per_worker"),
    }.items():
        for key in keys:
            if config[section][key] <= 0:
                raise ValueError(f"{section}.{key} must be positive")
    from .training_budget import validate_limits

    validate_limits(config["budget"]["iterations"], config["budget"]["max_seconds"])
    if config["initial_checkpoint"] is not None:
        if replay["initial_dataset"] is None:
            raise ValueError("initial_checkpoint requires replay.initial_dataset")
        if not pathlib.Path(config["initial_checkpoint"]).is_file():
            raise FileNotFoundError(config["initial_checkpoint"])
    if config["logging"]["progress_interval_seconds"] < 0:
        raise ValueError("logging.progress_interval_seconds cannot be negative")
    if config["validation"]["concept_interval"] < 0:
        raise ValueError("validation.concept_interval cannot be negative")
    if not 2 <= config["players"] <= game.PLAYER_COUNT or config["seed"] < 0:
        raise ValueError(
            "players must be between two and eight and seed must be nonnegative"
        )
    if config["seed"] > 2**32 - 1:
        raise ValueError("seed must fit a uint32")
    if not all(isinstance(tag, str) for tag in config["tags"]):
        raise ValueError("tags must be strings")
    for key in (
        "value_scale",
        "policy_scale",
    ):
        if training[key] < 0:
            raise ValueError(f"training.{key} cannot be negative")
    search = config["search"]
    if search["boundary_value_checkpoint"] is not None:
        if not pathlib.Path(search["boundary_value_checkpoint"]).is_file():
            raise FileNotFoundError(search["boundary_value_checkpoint"])
    elif search["boundary_samples"] != 1:
        raise ValueError("search.boundary_samples > 1 requires boundary_value_checkpoint")
    if not 0 <= search["dirichlet_epsilon"] <= 1:
        raise ValueError("search.dirichlet_epsilon must be between zero and one")
    if search["c_puct"] <= 0:
        raise ValueError("search.c_puct must be positive")
    if any(
        search[key] < 0
        for key in ("c_puct", "fpu_reduction", "action_softmax_temperature")
    ):
        raise ValueError("Search constants and temperature cannot be negative")
    if config["selfplay"]["start_state"] not in ("standard", "potential_clear"):
        raise ValueError("Unknown selfplay.start_state")
    if (
        config["selfplay"]["start_state"] == "potential_clear"
        and config["players"] != 2
    ):
        raise ValueError("potential_clear start states require two players")
    device = torch.device(config["execution"]["device"])
    if device.type not in ("cpu", "cuda", "mps"):
        raise ValueError("Supported devices: cpu, cuda, mps")
    if device.type == "cuda" and (
        not torch.cuda.is_available()
        or (device.index or 0) >= torch.cuda.device_count()
    ):
        raise ValueError("Requested CUDA device is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Requested MPS device is unavailable")
    players = config["players"]
    spatial = list(observations.spatial_input_shape(players))
    non_spatial = list(observations.get_non_spatial_input_shape(players))
    targets = [
        {"name": spec.name, "shape": list(spec.shape)}
        for spec in target_specs(players, config["auxiliary_objectives"])
    ]
    derived = {
        "spatial_input_shape": spatial,
        "non_spatial_input_shape": non_spatial,
        "action_mask_shape": list(observations.action_mask_shape()),
        "target_specs": targets,
        "optimizer": {"type": "adam", "weight_decay": 1e-4},
    }
    if prior_derived is not None and prior_derived != derived:
        raise ValueError(
            "Saved derived configuration differs; use the recorded code revision"
        )
    config["derived"] = derived
    if replay["initial_dataset"] is not None:
        if not isinstance(replay["initial_dataset"], str):
            raise ValueError("replay.initial_dataset must be a path string")
        dataset_path = (base_directory / replay["initial_dataset"]).resolve()
        manifest = json.loads((dataset_path / buffer.MANIFEST_FILE).read_text())
        if (
            manifest.get("format") != buffer.DATASET_FORMAT
            or manifest.get("version") != buffer.DATASET_VERSION
        ):
            raise ValueError("Unsupported initial replay dataset")
        for key in (
            "spatial_input_shape",
            "non_spatial_input_shape",
            "action_mask_shape",
            "target_specs",
        ):
            if manifest.get(key) != derived[key]:
                raise ValueError(
                    f"Initial dataset {key} does not match this experiment"
                )
        if (
            replay["dataset_id"] is not None
            and replay["dataset_id"] != manifest["dataset_id"]
        ):
            raise ValueError("Initial dataset identity has changed")
        replay.update(
            initial_dataset=str(dataset_path), dataset_id=manifest["dataset_id"]
        )
    elif replay["dataset_id"] is not None:
        raise ValueError("dataset_id requires initial_dataset")
    return config


def target_specs(players, configuration=None):
    return (
        *buffer.core_target_specs(players, (game.MASK_SIZE,)),
        *(
            buffer.TargetShapeSpec(name, shape)
            for name, shape in objectives.resolve(configuration).shapes(players).items()
        ),
    )
