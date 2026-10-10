# Skyjo

A NumPy game engine, Monte Carlo tree search, and neural self-play experiments
for full-game Skyjo. The maintained model learns game-win probabilities and
search policies, with optional round-score objectives.

## Start here

Use Python 3.12+ and [uv](https://docs.astral.sh/uv/). From the checkout:

```sh
uv sync --group dev
uv run pytest
uv run python distributed_main.py --config configs/smoke.toml --allow-dirty
```

The smoke recipe generates two complete games on CPU, trains, and saves a
checkpoint and replay. It verifies the pipeline; its results say nothing about
playing strength. `--allow-dirty` permits development runs and records that the
working tree was modified. For reproducible experiments, commit first and omit
the flag.

Each recorded launch prints a new `.runs/<run-id>/` directory. Outputs are
gitignored. Use CLI `--help` for options and the checked-in configurations for
portable starting points:

```sh
uv run python distributed_main.py --help
uv run python run_checkpoint_comparison.py --help
uv run python run_offline_comparison.py --help
```

If a platform's OpenMP runtime cannot initialize shared memory, run the suite
in an environment that supports it. A failed worker startup is not a completed
experiment.

## Code map

| Package | Responsibility |
| --- | --- |
| `skyjo.engine` | NumPy state, rules, transitions, scoring, chance outcomes |
| `skyjo.search` | MCTS, evaluator contracts, action symmetry, diagnostics |
| `skyjo.simulation` | Players, game execution, histories, random streams |
| `skyjo.learning` | Models, encoding, targets, replay, optimization, persistence |
| `skyjo.analytics` | Statistics and reports from recorded results |
| `skyjo.experiments` | Configuration, workers, recording, and experiment recipes |

The engine, search, and simulation can be imported without Torch. Experiments
assemble these components; the game engine does not know about checkpoints or
training labels. The separate `skyjo2` prototype is outside this architecture.

Read the [architecture guide](docs/architecture.md) for dependency direction,
player perspectives, randomness, and extension points.

## Run and compare experiments

The [experiment guide](docs/experiments.md) covers supported workflows, fair
comparisons, artifacts, and the process for preserving findings. Begin with the
smoke configuration, then choose an explicit generation/training budget.

Measure playing strength with balanced-seat checkpoint matches at stated
inference settings. Training loss, generated-game scores, and runtime answer
different questions and should be reported separately.

Current research conclusions are organized by question:

- [Score learning](docs/findings/score-learning.md): predictable score errors
  remain despite adequate capacity for a narrow arithmetic task.
- [Training progress and replay](docs/findings/training-progress.md): measured
  gains, regression, and the limits of larger replay interventions.
- [Search budget](docs/findings/search-budget.md): inference improvements do not
  establish better training per hour.
- [Boundary evaluation](docs/findings/boundary-evaluation.md): better local
  estimates have not established stronger full-game play.
- [Symmetry and performance](docs/findings/symmetry.md): smaller trees did not
  produce a material replay-wide throughput gain in the saved benchmark.

Each finding includes small committed evidence; the original local run artifacts
are not required to understand it. These are historical measurements, not
performance guarantees for the current implementation.

## Contributing

Keep rule changes in the engine, tensor computation in learning, and execution
policy in experiments. Add tests at the boundary whose behavior changes. For
performance changes, compare fixed workloads and report the environment as well
as the result.

Update the relevant guide when an interface changes. Promote an experiment
result only when it changes a decision or prevents a costly repeated mistake;
routine runs stay in their run directory. Use the short
[findings pattern](docs/experiments.md#preserve-a-finding) instead of appending
launch diaries to this README.
