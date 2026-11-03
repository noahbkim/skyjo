"""Construct model-backed players at the experiment boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from skyjo.learning import (
    boundary_inference,
    checkpoint,
    models,
    observations,
    predictor,
)
from skyjo.search.mcts import SearchConfig
from skyjo.search.player import SearchPlayer
from skyjo.search.value_player import ValueConfig, ValuePlayer


@dataclass(frozen=True)
class ContestantConfig:
    checkpoint: str | None = None
    mode: Literal["search", "policy", "value"] = "search"
    iterations: int = 128
    temperature: float = 0.0
    search: SearchConfig = field(default_factory=SearchConfig)
    boundary_checkpoint: str | None = None
    value: ValueConfig = field(default_factory=ValueConfig)

    def __post_init__(self):
        if self.mode not in ("search", "policy", "value"):
            raise ValueError("Player mode must be search, policy, or value")
        if type(self.iterations) is not int or self.iterations < 1:
            raise ValueError("Search iterations must be positive")
        if not np.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("Action temperature must be finite and nonnegative")
        if self.mode == "value" and self.temperature != 0:
            raise ValueError("Value-only play requires temperature zero")
        if self.mode != "value" and self.value != ValueConfig():
            raise ValueError("Value settings require value-only play")
        if self.mode == "policy" and (
            self.boundary_checkpoint is not None
            or self.search.boundary_samples != 1
            or self.search.after_state_evaluate_all_children
        ):
            raise ValueError(
                "Policy-only play cannot use search boundary/chance settings"
            )

    @classmethod
    def from_search_settings(cls, settings: dict, *, checkpoint=None, mode="search"):
        options = dict(settings)
        iterations = options.pop("iterations", 128)
        temperature = options.pop("action_softmax_temperature", 0.0)
        boundary = options.pop("boundary_value_checkpoint", None)
        return cls(
            checkpoint, mode, iterations, temperature, SearchConfig(**options), boundary
        )


def load_model(path: Path, *, device="cpu"):
    payload = checkpoint.decode_checkpoint(path, map_location=device)
    config = payload["configuration"]
    settings = dict(config["model"])
    # Missing metadata identifies the historical per-action normalization.
    settings.setdefault("policy_normalization", "actions")
    # Earlier experiment checkpoints embedded these derived settings in model.
    legacy_shape = settings.pop("non_spatial_input_shape", None)
    if legacy_shape is not None and tuple(
        legacy_shape
    ) != observations.get_non_spatial_input_shape(config["players"]):
        raise ValueError("Checkpoint observation shape does not match this engine")
    legacy_objectives = settings.pop("auxiliary_objectives", {})
    auxiliary_objectives = config.get("auxiliary_objectives", legacy_objectives)
    if legacy_objectives and auxiliary_objectives != legacy_objectives:
        raise ValueError("Checkpoint contains conflicting auxiliary objectives")
    model = models.build(
        settings,
        players=config["players"],
        device=device,
        auxiliary_objectives=auxiliary_objectives,
    )
    model.load_state_dict(payload["model_state_dict"])
    return model.eval()


@dataclass
class PreparedContestant:
    """A frozen evaluator loaded once per worker; each game gets a fresh player RNG."""

    config: ContestantConfig
    evaluator: predictor.LocalPredictor
    boundary_evaluator: object | None = None

    def player(self, rng: np.random.Generator):
        if self.config.mode == "policy":
            return predictor.PolicyPlayer(
                self.evaluator, temperature=self.config.temperature
            )
        if self.config.mode == "value":
            return ValuePlayer(
                self.evaluator,
                config=self.config.value,
                boundary_evaluator=self.boundary_evaluator,
                rng=rng,
            )
        return SearchPlayer(
            self.evaluator,
            self.config.iterations,
            config=self.config.search,
            temperature=self.config.temperature,
            boundary_evaluator=self.boundary_evaluator,
            rng=rng,
        )


def prepare(
    config: ContestantConfig, *, model=None, max_batch_size=512, fixed_eval_mode=False
):
    if model is None:
        if config.checkpoint is None:
            raise ValueError("A contestant requires a model or checkpoint")
        model = load_model(Path(config.checkpoint))
    boundary = None
    if config.boundary_checkpoint is not None:
        boundary_model = boundary_inference.load_boundary_model(
            config.boundary_checkpoint, model.players
        )
        boundary = boundary_inference.ScoreBoundaryEvaluator(boundary_model)
    return PreparedContestant(
        config,
        predictor.LocalPredictor(
            model, max_batch_size, fixed_eval_mode=fixed_eval_mode
        ),
        boundary,
    )


def from_file(
    checkpoint_path: Path, settings_path: Path | None = None, *, iterations=128
):
    """Load one contestant's optional TOML settings; checkpoint selection stays explicit."""
    import tomllib

    options = {} if settings_path is None else tomllib.loads(settings_path.read_text())
    search = SearchConfig(**options.pop("search", {}))
    value = ValueConfig(**options.pop("value", {}))
    boundary = options.pop("boundary_checkpoint", None)
    if boundary is not None:
        boundary = str((settings_path.parent / boundary).resolve())
    return ContestantConfig(
        checkpoint=str(checkpoint_path.resolve()),
        search=search,
        iterations=options.pop("iterations", iterations),
        boundary_checkpoint=boundary,
        value=value,
        **options,
    )


def from_shortcut(
    checkpoint_path: Path,
    *,
    mode="search",
    iterations=128,
    temperature=None,
    value_chance=None,
    value_samples=None,
    boundary_samples=None,
):
    """Build a quick faceoff contestant; policy shortcuts sample by default."""
    options = {
        name: value
        for name, value in (
            ("chance", value_chance),
            ("samples", value_samples),
            ("boundary_samples", boundary_samples),
        )
        if value is not None
    }
    if options and mode != "value":
        raise ValueError("Value chance/sample options require --mode value")
    return ContestantConfig(
        checkpoint=str(checkpoint_path.resolve()),
        mode=mode,
        iterations=iterations,
        temperature=(1.0 if mode == "policy" else 0.0)
        if temperature is None
        else temperature,
        value=ValueConfig(**options),
    )
