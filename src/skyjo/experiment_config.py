"""Plain-data configuration for the current distributed experiment recipe."""

from __future__ import annotations

import copy
import json
import math
import pathlib

import tomllib
import torch

from . import buffer, game, skynet

DEFAULTS = {
    "name": "score-auxiliary",
    "description": "",
    "tags": [],
    "notes": "",
    "seed": 0,
    "players": 2,
    "model": {
        "type": "round_score",
        "embedding_dimensions": 16,
        "global_state_embedding_dimensions": 32,
        "num_heads": 2,
    },
    "training": {
        "batch_size": 256,
        "replay_ratio": 4.0,
        "learn_rate": 0.001,
        "loss": "auxiliary",
        "value_scale": 1.0,
        "policy_scale": 1.0,
        "round_score_scale": 1.0,
        "future_clear_scale": 0.0,
        "clear_positive_weight": 10.0,
    },
    "selfplay": {
        "games_per_iteration": 1024,
        "games_per_task": 8,
        "outcome_rollouts": 100,
        "start_state": "standard",
    },
    "search": {
        "iterations": 100,
        "dirichlet_epsilon": 0.25,
        "after_state_evaluate_all_children": False,
        "terminal_state_initial_rollouts": 10,
        "c_puct": 1.0,
        "fpu_reduction": 0.25,
        "score_utility_weight": 0.0,
        "action_softmax_temperature": 1.0,
    },
    "replay": {
        "capacity": 2_000_000,
        "targets": "round_score",
        "initial_dataset": None,
        "dataset_id": None,
    },
    "validation": {
        "enabled": True,
        "interval": 1,
        "value_loss_scale": 1.0,
        "policy_loss_scale": 1.0,
    },
    "faceoff": {"paired_rounds": 0, "rounds_per_task": 1, "interval": 1},
    "budget": {"iterations": 10, "checkpoint_interval": 1},
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


def load_configuration(path: pathlib.Path) -> tuple[bytes, dict]:
    """Resolve rerunnable settings without constructing a model or replay arrays."""
    raw = path.read_bytes()
    if path.suffix == ".toml":
        supplied = tomllib.loads(raw.decode("utf-8"))
    elif path.suffix == ".json":
        supplied = json.loads(raw)
    else:
        raise ValueError("Configuration must be .toml or .json")
    if not isinstance(supplied, dict):
        raise ValueError("Configuration must be an object")
    prior_derived = supplied.pop("derived", None)
    config = _merge(DEFAULTS, supplied)
    model, training, replay = (config[key] for key in ("model", "training", "replay"))
    for section, keys in {
        "model": (
            "embedding_dimensions",
            "global_state_embedding_dimensions",
            "num_heads",
        ),
        "training": (
            "batch_size",
            "replay_ratio",
            "learn_rate",
            "clear_positive_weight",
        ),
        "selfplay": ("games_per_iteration", "games_per_task", "outcome_rollouts"),
        "search": ("iterations", "terminal_state_initial_rollouts"),
        "replay": ("capacity",),
        "validation": ("interval",),
        "faceoff": ("rounds_per_task", "interval"),
        "budget": ("iterations", "checkpoint_interval"),
        "execution": ("workers", "threads_per_worker"),
    }.items():
        for key in keys:
            if config[section][key] <= 0:
                raise ValueError(f"{section}.{key} must be positive")
    if config["players"] < 2 or config["seed"] < 0:
        raise ValueError("players must be at least two and seed must be nonnegative")
    if config["seed"] > 2**32 - 1:
        raise ValueError("seed must fit a uint32")
    if not all(isinstance(tag, str) for tag in config["tags"]):
        raise ValueError("tags must be strings")
    if config["faceoff"]["paired_rounds"] < 0:
        raise ValueError("faceoff.paired_rounds cannot be negative")
    if config["players"] != 2 and (
        config["validation"]["enabled"] or config["faceoff"]["paired_rounds"]
    ):
        raise ValueError(
            "The current validation and faceoff recipes require two players"
        )
    for key in ("embedding_dimensions", "global_state_embedding_dimensions"):
        if model[key] % model["num_heads"]:
            raise ValueError(f"model.{key} must be divisible by num_heads")
    for key in (
        "value_scale",
        "policy_scale",
        "round_score_scale",
        "future_clear_scale",
    ):
        if training[key] < 0:
            raise ValueError(f"training.{key} cannot be negative")
    for key in ("value_loss_scale", "policy_loss_scale"):
        if config["validation"][key] < 0:
            raise ValueError(f"validation.{key} cannot be negative")
    search = config["search"]
    if not 0 <= search["dirichlet_epsilon"] <= 1:
        raise ValueError("search.dirichlet_epsilon must be between zero and one")
    if any(
        search[key] < 0
        for key in ("c_puct", "fpu_reduction", "action_softmax_temperature")
    ):
        raise ValueError("Search constants and temperature cannot be negative")
    if model["type"] not in ("equivariant", "round_score", "auxiliary"):
        raise ValueError("Unknown model type")
    if training["loss"] not in ("base", "auxiliary"):
        raise ValueError("Unknown loss")
    if config["selfplay"]["start_state"] not in ("standard", "potential_clear"):
        raise ValueError("Unknown selfplay.start_state")
    if (
        config["selfplay"]["start_state"] == "potential_clear"
        and config["players"] != 2
    ):
        raise ValueError("potential_clear start states require two players")
    if replay["targets"] not in ("core", "round_score", "auxiliary"):
        raise ValueError("Unknown replay target selection")
    if training["loss"] == "base":
        if training["round_score_scale"] or training["future_clear_scale"]:
            raise ValueError("Set auxiliary scales to zero when using base loss")
    else:
        if training["round_score_scale"] and (
            model["type"] == "equivariant" or replay["targets"] == "core"
        ):
            raise ValueError("Round-score loss requires a score head and score targets")
        if training["future_clear_scale"] and (
            model["type"] != "auxiliary" or replay["targets"] != "auxiliary"
        ):
            raise ValueError(
                "Future-clear loss requires the auxiliary model and targets"
            )
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
    spatial = [players, game.ROW_COUNT, game.COLUMN_COUNT, game.FINGER_SIZE]
    non_spatial = list(skynet.get_non_spatial_input_shape(players))
    target_builder = {
        "core": buffer.core_target_specs,
        "round_score": buffer.round_score_target_specs,
        "auxiliary": buffer.auxiliary_target_specs,
    }[replay["targets"]]
    targets = [
        {"name": spec.name, "shape": list(spec.shape)}
        for spec in target_builder(players, (game.MASK_SIZE,))
    ]
    derived = {
        "spatial_input_shape": spatial,
        "non_spatial_input_shape": non_spatial,
        "action_mask_shape": [game.MASK_SIZE],
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
        dataset_path = (path.parent / replay["initial_dataset"]).resolve()
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
    return raw, config
