"""Status: experimental
Purpose: Compare modular search strategies on frozen tactical and endgame positions.
Promote when: A strategy is selected for recurring playing-strength evaluation.
"""

from pathlib import Path
from typing import Annotated

import typer


def benchmark(
    checkpoint: Annotated[Path, typer.Option(help="Frozen iteration 119 checkpoint")],
    output: Annotated[Path, typer.Option(help="New directory for incremental results")],
    fixtures: Annotated[Path, typer.Option()] = Path(__file__).parent
    / "configs/benchmarks/search_positions.json",
    budget: Annotated[
        list[int] | None, typer.Option(help="Repeat for each total search budget")
    ] = None,
    seeds: Annotated[int, typer.Option()] = 8,
    seed: Annotated[int, typer.Option()] = 20261012,
    case: Annotated[
        list[str] | None, typer.Option(help="Repeat to select fixture names")
    ] = None,
    variant: Annotated[
        list[str] | None, typer.Option(help="Repeat to select strategy names")
    ] = None,
    chance: Annotated[
        str, typer.Option(help="enumerated, sampled, or both (sampled repeat at 2048)")
    ] = "both",
    ensemble_budget: Annotated[
        int, typer.Option(help="Total budget for eight-replica variants")
    ] = 2048,
    workers: Annotated[int, typer.Option()] = 1,
) -> Path:
    from skyjo.experiments.search_strategy_benchmark import run_benchmark

    result = run_benchmark(
        fixture_path=fixtures,
        checkpoint=checkpoint,
        output=output,
        budgets=tuple(budget) if budget is not None else (256, 2048, 8192),
        seeds=seeds,
        base_seed=seed,
        case_names=tuple(case or ()),
        variant_names=tuple(variant or ()),
        chance=chance,
        ensemble_budget=ensemble_budget,
        workers=workers,
    )
    typer.echo(f"Search strategy benchmark: {result}")
    return result


def main() -> None:
    typer.run(benchmark)


if __name__ == "__main__":
    main()
