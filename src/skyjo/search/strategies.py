"""Composable, model-free search contracts and standard strategy implementations.

Strategies see immutable snapshots, never mutable tree nodes. Factories are reusable;
allocation sessions and backup accumulators belong to one search/node respectively.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

from .symmetry import ActionGroups


def frozen_array(value):
    result = np.array(value, copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True, slots=True)
class NodeView:
    groups: ActionGroups
    actions: tuple[int, ...]
    priors: np.ndarray
    model_priors: np.ndarray
    visits: np.ndarray
    values: np.ndarray
    available: np.ndarray
    player: int
    state_value: np.ndarray
    model_value: np.ndarray
    visit_count: int
    is_root: bool
    standard_error: np.ndarray | None = None
    node_id: int = 0

    @property
    def q(self):
        return self.values[:, self.player]


@dataclass(frozen=True, slots=True)
class Evidence:
    """One traversal's fresh leaf sample or initial enumerated expectation.

    A refresh is a recomputed estimate, not another independent observation.
    Boundary batches carry their fresh batch mean, never their historical mean.
    """

    value: np.ndarray
    kind: Literal["sample", "expectation", "exact", "refresh"] = "sample"

    def __post_init__(self):
        object.__setattr__(self, "value", frozen_array(self.value))


class ValueAccumulator(Protocol):
    @property
    def value(self) -> np.ndarray: ...
    def update(self, view: NodeView, evidence: Evidence) -> None: ...


class Valuation(Protocol):
    def initialize(self, model_value: np.ndarray) -> ValueAccumulator: ...


class AllocationSession(Protocol):
    def select(self, view: NodeView) -> int: ...

    observe: Callable[[NodeView], None] | None

    def finish(self, view: NodeView) -> object: ...


class Exploration(Protocol):
    def start(self, budget: int, rng: np.random.Generator) -> AllocationSession: ...


@dataclass(frozen=True, slots=True)
class PolicyInput:
    summary: NodeView
    context: object = None
    action_probabilities: np.ndarray | None = None
    replicas: tuple[PolicyInput, ...] = ()


class ResultPolicy(Protocol):
    def probabilities(self, result: PolicyInput, temperature: float) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class SearchComponents:
    root_exploration: Exploration
    interior_exploration: Exploration
    valuation: Valuation
    action_policy: ResultPolicy
    target_policy: ResultPolicy


@dataclass
class _MixtureValue:
    value: np.ndarray

    def update(self, view, evidence):
        self.value = (
            view.model_value + (view.visits[:, None] * view.values).sum(axis=0)
        ) / (1 + view.visits.sum())


@dataclass(frozen=True)
class CurrentMixture:
    def initialize(self, model_value):
        return _MixtureValue(model_value.copy())


@dataclass
class _MeanValue:
    value: np.ndarray
    count: int = 0

    def update(self, view, evidence):
        if evidence.kind == "refresh":
            return
        self.count += 1
        self.value = self.value + (evidence.value - self.value) / self.count


@dataclass(frozen=True)
class MeanReturn:
    def initialize(self, model_value):
        return _MeanValue(model_value.copy())


@dataclass
class _BestValue:
    value: np.ndarray

    def update(self, view, evidence):
        if view.available.any():
            index = int(np.argmax(np.where(view.available, view.q, -np.inf)))
            self.value = view.values[index].copy()
        else:
            self.value = view.model_value.copy()


@dataclass(frozen=True)
class BestResponse:
    def initialize(self, model_value):
        return _BestValue(model_value.copy())


@dataclass(frozen=True)
class PUCT:
    c_puct: float = 1.5
    fpu_reduction: float = 0.1
    fpu_mode: str = "parent"

    def scores(self, view):
        fpu = (
            view.state_value[view.player] - self.fpu_reduction
            if self.fpu_mode == "parent"
            else 0.0
        )
        return np.where(
            view.available, view.q, fpu
        ) + self.c_puct * view.priors * np.sqrt(view.visit_count + 1) / (
            view.visits + 1
        )

    def start(self, budget, rng):
        return _PUCTSession(self, rng)


@dataclass
class _PUCTSession:
    strategy: PUCT
    rng: np.random.Generator

    def select(self, view):
        scores = self.strategy.scores(view)
        candidates = np.flatnonzero(np.abs(scores - scores.max()) < 1e-5)
        index = (
            int(candidates[int(self.rng.integers(len(candidates)))])
            if len(candidates) > 1
            else int(candidates[0])
        )
        return view.actions[index]

    observe = None

    def finish(self, view):
        return None


@dataclass(frozen=True)
class CompletedQ:
    value_scale: float = 0.1
    visit_offset: float = 50.0
    rescale: bool = True

    def __call__(self, view):
        visited = view.visits > 0
        weights = np.maximum(view.model_priors, np.finfo(np.float64).tiny)
        weighted_q = (
            float(np.dot(weights[visited], view.q[visited]) / weights[visited].sum())
            if visited.any()
            else float(view.model_value[view.player])
        )
        n = view.visits.sum()
        mixed = (view.model_value[view.player] + n * weighted_q) / (n + 1)
        q = np.where(visited, view.q, mixed)
        if self.rescale:
            q = (q - q.min()) / max(float(np.ptp(q)), 1e-8)
        return (self.visit_offset + view.visits.max()) * self.value_scale * q


def prior_logits(view):
    # Evaluators expose probabilities, not logits. A floor handles underflow and
    # lets legal zero-prior actions participate without admitting illegal actions.
    return np.log(np.maximum(view.model_priors, np.finfo(np.float64).tiny))


def softmax(scores, temperature=1.0):
    if not np.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and nonnegative")
    if not np.isfinite(scores).any():
        raise ValueError(
            "policy requires at least one evaluated or visited legal action"
        )
    if temperature == 0:
        probabilities = np.zeros(len(scores))
        probabilities[int(np.argmax(scores))] = 1
        return probabilities
    with np.errstate(over="ignore", under="ignore"):
        weights = np.exp((scores - np.max(scores)) / temperature)
    return weights / weights.sum()


@dataclass(frozen=True)
class VisitPolicy:
    def probabilities(self, result, temperature):
        view = result.summary
        scores = np.full(len(view.actions), -np.inf)
        visited = view.visits > 0
        scores[visited] = np.log(view.visits[visited])
        return view.groups.distribute(
            softmax(scores, temperature), greedy=temperature == 0
        )


@dataclass(frozen=True)
class ValuePolicy:
    uncertainty_z: float = 0.0

    def probabilities(self, result, temperature):
        view = result.summary
        q = view.q.copy()
        if self.uncertainty_z:
            if view.standard_error is None:
                raise ValueError("uncertainty selection requires independent replicas")
            q -= self.uncertainty_z * view.standard_error[:, view.player]
        return view.groups.distribute(
            softmax(np.where(view.available, q, -np.inf), temperature),
            greedy=temperature == 0,
        )


@dataclass(frozen=True)
class LegacyTarget:
    def probabilities(self, result, temperature):
        if result.action_probabilities is None:
            raise ValueError("legacy target requires action probabilities")
        return result.action_probabilities.copy()


@dataclass(frozen=True)
class CompletedQPolicy:
    transform: CompletedQ = CompletedQ()

    def probabilities(self, result, temperature):
        view = result.summary
        return view.groups.distribute(
            softmax(prior_logits(view) + self.transform(view), temperature),
            greedy=temperature == 0,
        )


@dataclass(frozen=True)
class CompletedQAllocation:
    transform: CompletedQ = CompletedQ()

    def start(self, budget, rng):
        return _CompletedQSession(self.transform)


@dataclass
class _CompletedQSession:
    transform: CompletedQ

    def select(self, view):
        probabilities = softmax(prior_logits(view) + self.transform(view))
        return view.actions[
            int(np.argmax(probabilities - view.visits / (1 + view.visits.sum())))
        ]

    observe = None

    def finish(self, view):
        return None


# Schedule adapted from google-deepmind/mctx, Copyright 2021 DeepMind
# Technologies Limited, licensed under Apache 2.0 (docs/mctx-license.txt).
# This NumPy implementation caps candidates by the actual search budget.
def considered_visits(count, budget):
    """Sequential-halving schedule, including partial rounds (DeepMind mctx)."""
    if count <= 1:
        return tuple(range(budget))
    rounds = int(np.ceil(np.log2(count)))
    visits = [0] * count
    sequence = []
    remaining = count
    while len(sequence) < budget:
        for _ in range(max(1, budget // (rounds * remaining))):
            sequence.extend(visits[:remaining])
            for i in range(remaining):
                visits[i] += 1
        remaining = max(2, remaining // 2)
    return tuple(sequence[:budget])


@dataclass(frozen=True)
class GumbelContext:
    noise: np.ndarray


@dataclass(frozen=True)
class GumbelAllocation:
    max_candidates: int = 16
    scale: float = 1.0
    transform: CompletedQ = CompletedQ()

    def start(self, budget, rng):
        return _GumbelSession(self, budget, rng)


@dataclass
class _GumbelSession:
    strategy: GumbelAllocation
    budget: int
    rng: np.random.Generator
    noise: np.ndarray | None = None
    schedule: tuple[int, ...] = ()
    calls: int = 0
    initial_visits: np.ndarray | None = None

    def select(self, view):
        if self.noise is None:
            self.noise = frozen_array(
                self.rng.gumbel(size=len(view.actions)) * self.strategy.scale
            )
            count = min(self.strategy.max_candidates, len(view.actions), self.budget)
            self.schedule = considered_visits(count, self.budget)
            self.initial_visits = view.visits.copy()
        scores = prior_logits(view) + self.noise + self.strategy.transform(view)
        eligible = view.visits - self.initial_visits == self.schedule[self.calls]
        self.calls += 1
        return view.actions[int(np.argmax(np.where(eligible, scores, -np.inf)))]

    observe = None

    def finish(self, view):
        return GumbelContext(
            frozen_array(
                np.zeros(len(view.actions)) if self.noise is None else self.noise
            )
        )


@dataclass(frozen=True)
class GumbelPolicy:
    transform: CompletedQ = CompletedQ()

    def probabilities(self, result, temperature):
        if not isinstance(result.context, GumbelContext):
            raise TypeError("Gumbel winner policy requires Gumbel allocation context")
        view = result.summary
        scores = prior_logits(view) + result.context.noise + self.transform(view)
        scores = np.where(view.visits == view.visits.max(), scores, -np.inf)
        return view.groups.distribute(softmax(scores, 0), greedy=True)


@dataclass(frozen=True)
class AveragedTarget:
    """Compose a target over independently searched trees, without pooled scaling."""

    policy: ResultPolicy

    def probabilities(self, result, temperature):
        if not result.replicas:
            return self.policy.probabilities(result, temperature)
        return np.mean(
            [
                self.policy.probabilities(replica, temperature)
                for replica in result.replicas
            ],
            axis=0,
        ).astype(np.float32)
