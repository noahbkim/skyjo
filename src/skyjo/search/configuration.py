"""Serializable settings and the only name-based strategy composition boundary."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .strategies import (
    PUCT,
    AveragedTarget,
    BestResponse,
    CompletedQ,
    CompletedQAllocation,
    CompletedQPolicy,
    CurrentMixture,
    GumbelAllocation,
    GumbelPolicy,
    LegacyTarget,
    MeanReturn,
    SearchComponents,
    ValuePolicy,
    VisitPolicy,
)


@dataclass(frozen=True, slots=True)
class SearchConfig:
    dirichlet_epsilon: float = 0.0
    after_state_evaluate_all_children: bool = False
    c_puct: float = 1.5
    fpu_reduction: float = 0.1
    boundary_samples: int = 1
    merge_symmetric_actions: bool = True
    dirichlet_alpha: float | None = None
    fpu_mode: str = "parent"
    root_strategy: str = "puct"
    interior_strategy: str = "puct"
    valuation: str = "current_mixture"
    action_strategy: str = "visits"
    target_strategy: str = "legacy"
    gumbel_candidates: int = 16
    gumbel_scale: float = 1.0
    completed_q_scale: float = 0.1
    completed_q_visit_offset: float = 50.0
    completed_q_rescale: bool = True
    replicas: int = 1
    uncertainty_z: float = 1.0

    def __post_init__(self):
        for name in ("c_puct", "completed_q_scale"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "fpu_reduction",
            "gumbel_scale",
            "completed_q_visit_offset",
            "uncertainty_z",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if (
            not np.isfinite(self.dirichlet_epsilon)
            or not 0 <= self.dirichlet_epsilon <= 1
        ):
            raise ValueError("dirichlet_epsilon must be between zero and one")
        if self.dirichlet_alpha is not None and (
            not np.isfinite(self.dirichlet_alpha) or self.dirichlet_alpha <= 0
        ):
            raise ValueError("dirichlet_alpha must be finite and positive")
        for name in ("boundary_samples", "gumbel_candidates", "replicas"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "merge_symmetric_actions",
            "after_state_evaluate_all_children",
            "completed_q_rescale",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        choices = {
            "fpu_mode": ("parent", "zero"),
            "root_strategy": ("puct", "gumbel", "completed_q"),
            "interior_strategy": ("puct", "completed_q"),
            "valuation": ("current_mixture", "mean_return", "best_response"),
            "action_strategy": ("visits", "value", "gumbel", "lower_confidence"),
            "target_strategy": ("legacy", "visits", "completed_q"),
        }
        for name, allowed in choices.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"Unknown {name}: {getattr(self, name)}")
        if self.root_strategy != "puct" and self.dirichlet_epsilon:
            raise ValueError("Dirichlet noise requires PUCT root exploration")
        if self.action_strategy == "gumbel" and (
            self.root_strategy != "gumbel" or self.replicas != 1
        ):
            raise ValueError("Gumbel winner requires a single Gumbel search")
        if self.action_strategy == "lower_confidence" and self.replicas < 2:
            raise ValueError("lower_confidence requires at least two replicas")


DEFAULT_SEARCH_CONFIG = SearchConfig()


def resolve_components(config: SearchConfig) -> SearchComponents:
    transform = CompletedQ(
        config.completed_q_scale,
        config.completed_q_visit_offset,
        config.completed_q_rescale,
    )
    explorations = {
        "puct": PUCT(config.c_puct, config.fpu_reduction, config.fpu_mode),
        "completed_q": CompletedQAllocation(transform),
        "gumbel": GumbelAllocation(
            config.gumbel_candidates, config.gumbel_scale, transform
        ),
    }
    valuations = {
        "current_mixture": CurrentMixture(),
        "mean_return": MeanReturn(),
        "best_response": BestResponse(),
    }
    actions = {
        "visits": VisitPolicy(),
        "value": ValuePolicy(),
        "gumbel": GumbelPolicy(transform),
        "lower_confidence": ValuePolicy(config.uncertainty_z),
    }
    targets = {
        "legacy": LegacyTarget(),
        "visits": VisitPolicy(),
        "completed_q": CompletedQPolicy(transform),
    }
    target = targets[config.target_strategy]
    if config.replicas > 1 and config.target_strategy == "completed_q":
        target = AveragedTarget(target)
    return SearchComponents(
        explorations[config.root_strategy],
        explorations[config.interior_strategy],
        valuations[config.valuation],
        actions[config.action_strategy],
        target,
    )


def preset(name: str, **overrides) -> SearchConfig:
    configs = {
        "legacy": DEFAULT_SEARCH_CONFIG,
        "alphazero": SearchConfig(
            valuation="mean_return", fpu_mode="zero", target_strategy="visits"
        ),
        "gumbel": SearchConfig(
            root_strategy="gumbel",
            interior_strategy="completed_q",
            valuation="mean_return",
            action_strategy="gumbel",
            target_strategy="completed_q",
        ),
        "lower_confidence": SearchConfig(
            valuation="mean_return",
            fpu_mode="zero",
            action_strategy="lower_confidence",
            target_strategy="visits",
            replicas=8,
        ),
    }
    if name not in configs:
        raise ValueError(f"Unknown search preset: {name}")
    return replace(configs[name], **overrides)
