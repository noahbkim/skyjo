"""NumPy contracts shared by search and its supplied evaluators."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from skyjo.engine import game as sj
from skyjo.engine import values


@dataclass(frozen=True, slots=True)
class Prediction:
    """Value probabilities in fixed seat order and legal-action probabilities."""

    value: np.ndarray
    policy: np.ndarray


class Evaluator(Protocol):
    def evaluate(self, states: Sequence[sj.Skyjo]) -> list[Prediction]:
        """Evaluate states in input order, returning an empty list for no states."""
        ...


class BoundaryEvaluator(Protocol):
    def evaluate(
        self, completed_rounds: Sequence[sj.Skyjo], rng: np.random.Generator
    ) -> np.ndarray:
        """Return one fixed-seat value per completed round, including exact ties."""
        ...


@dataclass(frozen=True, slots=True)
class NextDealEvaluator:
    """Bootstrap continuing rounds from a fresh deal; terminal values are exact."""

    evaluator: Evaluator

    def evaluate(self, completed_rounds, rng):
        result = np.empty(
            (len(completed_rounds), completed_rounds[0].players), dtype=np.float32
        )
        indices, deals = [], []
        for index, state in enumerate(completed_rounds):
            if sj.get_game_over(state):
                result[index] = values.skyjo_to_game_state_value(state)
            else:
                indices.append(index)
                deals.append(sj.start_next_round(state, rng=rng))
        for index, prediction in zip(
            indices, self.evaluator.evaluate(deals), strict=True
        ):
            result[index] = prediction.value
        return result
