"""Status: experimental
Purpose: Compare baseline, score, doubling, and combined auxiliary configurations.
Promote when: This becomes the standard experiment runner.
"""

import json
import logging
from pathlib import Path
from typing import Annotated

import torch
import typer

from skyjo.ablation import AblationConfig, run_ablation


def compare(
    output: Path = Path("data/auxiliary_ablation.json"),
    seeds: Annotated[str, typer.Option(help="Comma-separated seeds")] = "0,1",
    games: int = 8,
    training_steps: int = 32,
    batch_size: int = 32,
    evaluation_pairs: int = 4,
    terminal_rollouts: int = 32,
    mcts_iterations: int = 16,
    torch_threads: int = 1,
) -> None:
    torch.set_num_threads(torch_threads)
    report = run_ablation(AblationConfig(
        seeds=tuple(int(seed) for seed in seeds.split(",")), games=games,
        training_steps=training_steps, batch_size=batch_size,
        evaluation_pairs=evaluation_pairs, terminal_rollouts=terminal_rollouts,
        mcts_iterations=mcts_iterations,
    ))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    typer.echo(f"Saved matched ablation results to {output}")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    typer.run(compare)


if __name__ == "__main__":
    main()
