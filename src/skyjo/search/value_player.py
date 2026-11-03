"""One-action value lookahead, independent of policy priors and tree search."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from skyjo.engine import game as sj
from skyjo.engine import values
from skyjo.simulation.decisions import DiagnosticSection, PlayerDecision

from .batching import EvaluationRequest, evaluate
from .evaluator import BoundaryEvaluator, Evaluator, NextDealEvaluator


@dataclass(frozen=True, slots=True)
class ValueConfig:
    chance: Literal["enumerated", "sampled"] = "enumerated"
    samples: int = 1
    boundary_samples: int = 10

    def __post_init__(self):
        if self.chance not in ("enumerated", "sampled"):
            raise ValueError("value.chance must be enumerated or sampled")
        for name in ("samples", "boundary_samples"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"value.{name} must be a positive integer")


DEFAULT_VALUE_CONFIG = ValueConfig()


class ValuePlayer:
    """Choose the highest expected successor value for the original acting seat.

    Ordinary chance is exact by default; round endings are always sampled.
    Every legal action is evaluated, without policy pruning or symmetry merging.
    The supplied RNG owns imagined outcomes, never actual gameplay randomness.
    """

    def __init__(
        self,
        evaluator: Evaluator,
        *,
        config: ValueConfig = DEFAULT_VALUE_CONFIG,
        boundary_evaluator: BoundaryEvaluator | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.evaluator = evaluator
        self.config = config
        self.boundary_evaluator = (
            boundary_evaluator
            if boundary_evaluator is not None
            else NextDealEvaluator(evaluator)
        )
        self.rng = rng if rng is not None else np.random.default_rng()

    def get_action_probabilities(self, state: sj.Skyjo) -> np.ndarray:
        return self.get_decision(state, diagnostics=False).probabilities

    def get_decision(self, state: sj.Skyjo, *, diagnostics=True) -> PlayerDecision:
        return evaluate(self.decision_steps(state, diagnostics=diagnostics))

    def _outcomes(self, state, action):
        if sj.get_round_about_to_end(state):
            count = self.config.boundary_samples
        elif self.config.chance == "enumerated":
            return sj.chance_outcomes(state, action)
        else:
            count = self.config.samples if sj.is_action_random(action, state) else 1
        return [
            (sj.apply_action(state, action, rng=self.rng), 1.0 / count)
            for _ in range(count)
        ]

    def _values_steps(self, states):
        result = np.empty((len(states), states[0].players), dtype=np.float64)
        ordinary, continuing = [], []
        for index, state in enumerate(states):
            if sj.get_game_over(state):
                result[index] = values.skyjo_to_game_state_value(state)
            elif sj.get_round_over(state):
                continuing.append(index)
            else:
                ordinary.append(index)
        if ordinary:
            predictions = yield EvaluationRequest(
                self.evaluator, [states[index] for index in ordinary]
            )
            result[ordinary] = [prediction.value for prediction in predictions]
        if continuing:
            completed = [states[index] for index in continuing]
            boundary = self.boundary_evaluator
            if type(boundary) is NextDealEvaluator:
                result[continuing] = yield from boundary.evaluate_steps(
                    completed, self.rng
                )
            else:
                result[continuing] = boundary.evaluate(completed, self.rng)
        return result

    def decision_steps(self, state, *, diagnostics=False):
        if sj.get_round_over(state):
            raise ValueError("Value lookahead requires an active decision")
        actions = tuple(map(int, np.flatnonzero(sj.actions(state))))
        outcomes = [self._outcomes(state, action) for action in actions]
        states = [child for children in outcomes for child, _ in children]
        predictions = yield from self._values_steps(states)
        player = sj.get_player(state)
        action_values = []
        offset = 0
        for children in outcomes:
            weights = np.array([weight for _, weight in children])
            action_values.append(
                float(weights @ predictions[offset : offset + len(children), player])
            )
            offset += len(children)
        probabilities = np.zeros(sj.MASK_SIZE, dtype=np.float32)
        probabilities[actions[int(np.argmax(action_values))]] = 1
        return PlayerDecision(
            probabilities,
            (
                DiagnosticSection(
                    "One-action values",
                    text="Expected win credit for the original acting seat. Ordinary "
                    "enumeration uses true chance probabilities; round endings use "
                    "sample averages. Neural values remain estimates.",
                    metrics={
                        "Acting seat": player + 1,
                        "Ordinary chance": self.config.chance,
                        "Ordinary samples": self.config.samples,
                        "Boundary samples": self.config.boundary_samples,
                    },
                    columns=(
                        "Action",
                        "Expected value",
                        "Outcomes",
                        "Selection probability",
                    ),
                    rows=tuple(
                        (
                            sj.get_action_name(action),
                            value,
                            len(children),
                            float(probabilities[action]),
                        )
                        for action, value, children in zip(
                            actions, action_values, outcomes, strict=True
                        )
                    ),
                    row_actions=actions,
                ),
            )
            if diagnostics
            else (),
        )
