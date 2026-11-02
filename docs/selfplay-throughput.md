# CPU self-play throughput

Independent games can share neural evaluation batches within each generation
worker. Enable this in the run configuration:

```toml
[selfplay]
games_per_task = 8
concurrent_games = 4

[execution]
threads_per_worker = 1
```

`concurrent_games` defaults to **1**. It caps active games inside each worker
task, so it cannot exceed the number of games assigned to that task in practice.
Finished slots are refilled until the task runs out of games. `games_per_task`
still controls multiprocessing task size; `execution.workers` controls processes.
Four active games is a starting configuration, not a measured universal optimum.

Each game retains separate environment, action, search, and randomized-budget
RNG streams. Only one evaluation request is outstanding per game: its search
finishes the current backup before choosing another leaf. There is no virtual
loss or within-tree parallelism. Results return in game-index order. Randomized
search caps, policy-only play, replicas, custom search strategies, and supplied
round-boundary evaluators remain supported. Fresh-next-deal bootstrap evaluations
join the shared batches; supplied boundary evaluators execute synchronously.

The serial and cooperative paths share search, policy construction, and game
simulation code. With an evaluator whose outputs are independent of batch shape,
batching preserves search and game results. Real neural batch shapes can introduce
floating-point differences near selection ties; changing concurrency or task
partitioning therefore need not reproduce identical trajectories. Record these
settings with the run, as the normal configuration/checkpoint machinery does.

Built-in PUCT/current-mixture search now keeps per-action statistics between
traversals and constructs child nodes when first accessed. This optimization is
automatic, including at concurrency 1. Other strategies retain the generic path.
Public search snapshots still own immutable copies. Explicitly iterating a node's
`children.values()` or `children.items()` materializes unvisited children for
inspection; taking `view()` does not. Generation owns a frozen model snapshot and
sets evaluation mode once, avoiding repeated module traversal per prediction.

The November 1 prototype experiments on the small CPU model measured roughly
11–18% more serial search throughput and 32–38% more throughput with 16 concurrent
searches from the storage changes. Four complete games at 400 iterations and four
active slots improved by 24.7%, with all 1,310 decisions and RNG streams matching
the paired reference. These are local workload measurements, not estimates of
training improvement or worker scaling. The original storage prototype is
preserved in commit `7740007`; local evidence is under
`.runs/diagnostics/search-core-20261101/`.

The maintained implementation was also checked against its generic path using
the same scheduler and frozen model: all 24 paired searches at 400/800/1,600
iterations matched policies, root statistics, and RNG state exactly. Two full
games at 400 iterations and two active slots matched all 534 decisions and saved
histories, taking 124.9s versus 103.6s (+20.6% throughput). This was one paired
sweep; evidence is in `.runs/diagnostics/selfplay-throughput-integration-20261102/`.
