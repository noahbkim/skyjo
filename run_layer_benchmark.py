"""Status: manual
Purpose: Time fixed transition, inference, search and optimization workloads.
Promote when: These timings become a recurring performance check.
"""

import json
import random
import statistics
import time
from collections.abc import Callable

import typer


def _time(work: Callable[[], None]) -> dict:
    work()
    samples = []
    for _ in range(3):
        started = time.perf_counter()
        work()
        samples.append(time.perf_counter() - started)
    return {"median_seconds": statistics.median(samples), "seconds": samples}


def benchmark() -> dict:
    """Print fixed-workload timings as JSON; run on an otherwise idle machine.

    One warmup precedes three timed repetitions per layer. Model weights evolve
    during the optimization phase. Search budgets are fixed, not trajectories.
    """
    import numpy as np
    import torch
    from skyjo.engine import game
    from skyjo.learning import batches, checkpoint, losses, models, predictor, train
    from skyjo.search import mcts

    caller_rng, caller_threads = checkpoint.capture_rng_state(), torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        torch.manual_seed(0)
        model = models.build(
            {
                "embedding_dimensions": 16,
                "global_state_embedding_dimensions": 32,
                "num_heads": 2,
            },
            players=2,
            device="cpu",
        )
        inference = predictor.LocalPredictor(model, 64)
        rng = random.Random(120)
        state = game.start_round(game.new(players=2, rng=rng), rng=rng)
        states = []
        for index in range(1000):
            if game.get_round_over(state):
                state = game.start_round(game.new(players=2, rng=rng), rng=rng)
            states.append(state)
            legal = np.flatnonzero(game.actions(state))
            state = game.apply_action(state, int(legal[index % len(legal)]), rng=rng)
        inputs = batches.states_to_batch(states[:64])
        training = batches.TrainingBatch(
            inputs.spatial_inputs,
            inputs.non_spatial_inputs,
            inputs.action_masks,
            {
                "value": np.full((64, 2), 0.5, dtype=np.float32),
                "policy": inputs.action_masks
                / inputs.action_masks.sum(axis=1, keepdims=True),
            },
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        def transitions():
            for index, state in enumerate(states):
                action = int(
                    np.flatnonzero(game.actions(state))[
                        index % int(game.actions(state).sum())
                    ]
                )
                game.apply_action(state, action, rng=random.Random(index))

        def infer():
            for _ in range(20):
                inference.evaluate(states[:64])

        def search():
            for index in range(10):
                mcts.run_mcts(
                    states[50 + index], inference, 32, rng=np.random.default_rng(index)
                )

        def optimize():
            for _ in range(20):
                train.train_step(model, training, losses.configured_loss, optimizer)

        results = {
            name: _time(work)
            for name, work in (
                ("transitions_1000", transitions),
                ("inference_1280", infer),
                ("search_10x32", search),
                ("optimizer_20x64", optimize),
            )
        }
        typer.echo(json.dumps(results, indent=2))
        return results
    finally:
        checkpoint.restore_rng_state(caller_rng)
        torch.set_num_threads(caller_threads)


def main() -> None:
    typer.run(benchmark)


if __name__ == "__main__":
    main()
