"""Soft invocation budgets, checked only at completed-iteration boundaries."""

import dataclasses
import math
import time


def validate_limits(iterations: int, max_seconds: float) -> None:
    if type(iterations) is not int or iterations < 0:
        raise ValueError("budget.iterations must be a nonnegative integer")
    if (
        type(max_seconds) not in (int, float)
        or not math.isfinite(max_seconds)
        or max_seconds < 0
    ):
        raise ValueError("budget.max_seconds must be finite and nonnegative")
    if iterations == 0 and max_seconds == 0:
        raise ValueError("At least one training budget must be enabled")


@dataclasses.dataclass(frozen=True)
class TrainingBudget:
    iterations: int
    max_seconds: float
    started: float = dataclasses.field(default_factory=time.perf_counter)

    def __post_init__(self):
        validate_limits(self.iterations, self.max_seconds)

    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    def stop_reason(self, completed: int) -> str | None:
        iterations = self.iterations > 0 and completed >= self.iterations
        timed = self.max_seconds > 0 and self.elapsed() >= self.max_seconds
        if iterations and timed:
            return "both"
        if iterations:
            return "iteration_limit"
        if timed:
            return "time_limit"
        return None

    def metrics(self) -> dict:
        elapsed = self.elapsed()
        return {
            "budget/iterations": self.iterations,
            "budget/max_seconds": self.max_seconds,
            "time/run_seconds": elapsed,
            "time/budget_overshoot_seconds": max(0.0, elapsed - self.max_seconds)
            if self.max_seconds
            else 0.0,
        }
