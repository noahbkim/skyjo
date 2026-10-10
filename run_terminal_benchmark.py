"""Status: experimental
Purpose: Compare final-action sampling to exact or independent large references.
Promote when: Boundary evaluation is selected for integration with search.
"""

import os
from pathlib import Path
from typing import Annotated

import typer


def benchmark(
    source_run: Annotated[Path, typer.Option("--source-run")],
    runs_dir: Annotated[Path, typer.Option("--runs-dir")] = Path(".runs"),
    exact_cases: Annotated[int, typer.Option("--exact-cases")] = 32,
    sampled_cases: Annotated[int, typer.Option("--sampled-cases")] = 8,
    reference_samples: Annotated[int, typer.Option("--reference-samples")] = 8192,
    repetitions: Annotated[int, typer.Option("--repetitions")] = 32,
    seed: Annotated[int, typer.Option("--seed")] = 20261006,
    allow_dirty: Annotated[bool, typer.Option("--allow-dirty")] = False,
) -> Path:
    """Run a bounded public-state audit without changing models or gameplay."""
    from skyjo.experiments.terminal_benchmark import Settings, run_benchmark

    result = run_benchmark(
        source_run,
        runs_dir,
        Settings(
            exact_cases=exact_cases,
            sampled_cases=sampled_cases,
            reference_samples=reference_samples,
            repetitions=repetitions,
            seed=seed,
        ),
        allow_dirty=allow_dirty,
    )
    typer.echo(f"Terminal benchmark: {result}")
    return result


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    typer.run(benchmark)


if __name__ == "__main__":
    main()
