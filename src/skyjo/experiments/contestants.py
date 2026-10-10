"""Construct model-backed players at the experiment boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from skyjo.learning import boundary_inference, checkpoint, models, predictor
from skyjo.search.mcts import SearchConfig
from skyjo.search.player import SearchPlayer


@dataclass(frozen=True)
class ContestantConfig:
    checkpoint: str | None = None
    mode: Literal["search", "policy"] = "search"
    iterations: int = 128
    temperature: float = 0.0
    search: SearchConfig = field(default_factory=SearchConfig)
    boundary_checkpoint: str | None = None

    def __post_init__(self):
        if self.mode not in ("search", "policy"):
            raise ValueError("Player mode must be search or policy")
        if type(self.iterations) is not int or self.iterations < 1:
            raise ValueError("Search iterations must be positive")
        if not np.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("Action temperature must be finite and nonnegative")
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
    model = models.build(
        config["model"],
        players=config["players"],
        device=device,
        auxiliary_objectives=config.get("auxiliary_objectives", {}),
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
            return predictor.PolicyPlayer(self.evaluator)
        return SearchPlayer(
            self.evaluator,
            self.config.iterations,
            config=self.config.search,
            temperature=self.config.temperature,
            boundary_evaluator=self.boundary_evaluator,
            rng=rng,
        )


def prepare(config: ContestantConfig, *, model=None, max_batch_size=512):
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
        config, predictor.LocalPredictor(model, max_batch_size), boundary
    )


def from_file(
    checkpoint_path: Path, settings_path: Path | None = None, *, iterations=128
):
    """Load one contestant's optional TOML settings; checkpoint selection stays explicit."""
    import tomllib

    options = {} if settings_path is None else tomllib.loads(settings_path.read_text())
    search = SearchConfig(**options.pop("search", {}))
    boundary = options.pop("boundary_checkpoint", None)
    if boundary is not None:
        boundary = str((settings_path.parent / boundary).resolve())
    return ContestantConfig(
        checkpoint=str(checkpoint_path.resolve()),
        search=search,
        iterations=options.pop("iterations", iterations),
        boundary_checkpoint=boundary,
        **options,
    )
