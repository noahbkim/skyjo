"""A module for function benchmarking with all-in-one timing, result statistics, and visualizations."""

import abc
import dataclasses
import time
import typing

import numpy as np
import pandas as pd
import plotly.express as px
import scipy.stats as st
import tqdm

# MARK: Dataclasses


@dataclasses.dataclass(slots=True)
class ConfidenceInterval:
    """A dataclass for a confidence interval."""

    lower: float
    upper: float
    confidence_level: float  # always

    def __post_init__(self) -> None:
        assert 0 < self.confidence_level < 1, "confidence level must be between 0 and 1"


class AbstractBenchStats(abc.ABC):
    """A dataclass for benchmark results."""

    pass


class ScalarBenchStats(AbstractBenchStats):
    """A dataclass for benchmark results for functions that return a single value."""

    def __init__(self, run_times: np.ndarray[float], results: np.ndarray) -> None:
        self.df = pd.DataFrame({"run_time": run_times, "result": results})

    @property
    def sample_size(self) -> int:
        return len(self.run_times)

    def run_times_histogram(self, **plotly_histogram_kwargs: dict[str, typing.Any]):
        return px.histogram(self.df.run_time, **plotly_histogram_kwargs)

    def results_histogram(self, **plotly_histogram_kwargs: dict[str, typing.Any]):
        return px.histogram(self.df.result, **plotly_histogram_kwargs)

    def summary_statistics(self) -> pd.DataFrame:
        stats_df = self.df.describe()
        runtime_mean_ci = scalar_mean_clt_confidence_interval(self.df.run_time)
        results_mean_ci = scalar_mean_clt_confidence_interval(self.df.result)
        stats_df.loc["Mean 95% CI (Lower)"] = (
            runtime_mean_ci.lower,
            results_mean_ci.lower,
        )
        stats_df.loc["Mean 95% CI (Upper)"] = (
            runtime_mean_ci.upper,
            results_mean_ci.upper,
        )
        return stats_df


# MARK: Utility


def scalar_mean_clt_confidence_interval(
    samples: np.ndarray[float], alpha: float = 0.05
) -> tuple[float, float]:
    """Calculate the confidence interval for a single variable using the Central Limit Theorem."""
    assert alpha <= 0.5, "alpha must be less than 0.5"
    mean = np.mean(samples)
    std = np.std(samples)
    std_dev_multiplier = st.norm.ppf(1 - alpha / 2)
    return ConfidenceInterval(
        mean - std_dev_multiplier * std / np.sqrt(len(samples)),
        mean + std_dev_multiplier * std / np.sqrt(len(samples)),
        confidence_level=1 - alpha,
    )


def vector_mean_clt_separate_confidence_intervals(
    samples: np.ndarray[tuple[int], float], alpha: float = 0.05
) -> list[ConfidenceInterval]:
    """Calculate the confidence interval for a vector using the Central Limit Theorem."""
    mean = np.mean(samples, axis=0)
    std = np.std(samples, axis=0)
    std_dev_multiplier = st.norm.ppf(1 - alpha / 2)
    return [
        ConfidenceInterval(
            mean[i] - std_dev_multiplier * std[i],
            mean[i] + std_dev_multiplier * std[i],
            confidence_level=1 - alpha,
        )
        for i in range(samples.shape[1])
    ]


# MARK: Benchmark


def initial_function_timing(
    func: typing.Callable, wall_time_threshold: float = 0.01
) -> int:
    """Determine the number of loops to run to get a desired runtime."""
    wall_time = 0
    loops = 1
    while wall_time < wall_time_threshold:
        start_time = time.perf_counter()
        for _ in range(loops):
            func()
        end_time = time.perf_counter()
        wall_time = end_time - start_time
        if wall_time < wall_time_threshold:
            loops *= 10
    return loops, wall_time


def scalar_benchmark(
    func: typing.Callable, repeats: int | None = None
) -> AbstractBenchStats:
    """Benchmark a function and return the results."""
    loops, wall_time = initial_function_timing(func)
    if repeats is None:
        repeats = int(60 / wall_time)
    print(loops, wall_time, repeats)
    run_times = np.empty(repeats)
    results = np.empty(repeats)
    for i in tqdm.tqdm(range(repeats)):
        start_time = time.perf_counter()
        for _ in range(loops):
            func()
        end_time = time.perf_counter()
        run_times[i] = end_time - start_time
        results[i] = func()
    return ScalarBenchStats(run_times, results)
