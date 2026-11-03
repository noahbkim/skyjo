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
There is no barrier between cohorts of `concurrent_games` games. At the end of
each task, however, its remaining games drain before that worker takes a new
task. Small tasks can therefore lose batching efficiency to long-game tails.
Four active games is a starting configuration, not a measured universal optimum.

Progress counts include only returned worker tasks, not completed games still
inside an unfinished task. `logging.progress_interval_seconds = 0` prints only
the final generation summary. A positive value enables roughly 10% game-count
milestones with that minimum spacing in seconds; idle waits emit nothing.
Milestones show cumulative returned games/second and decisions/second. After
25% of games return, they also show a rough generation ETA based on that rate.
These task-batched estimates omit in-flight work and can jump when tasks return;
the ETA does not include subsequent target construction or training. Idle waits
do not repeatedly recalculate an ETA from a stale count. Use the completed
generation's decisions/second to compare throughput, alongside game lengths and
rounds/game to explain changes in games/hour.

Each completed iteration logs game-length mean/median/p90/max, round scores,
clears, no-progress endings, action rates per eligible decision, replay fitting
losses, policy entropy/KL, and phase timings. Replay fitting losses are training
diagnostics, not held-out validation. The default `validation.concept_interval`
is now 1: the cheap handcrafted suite logs action matches, target probability,
and its three weakest examples each iteration. Explicit intervals remain honored;
0 disables it. Its heuristic value MSE is not calibrated win-rate error, and the
examples never become training labels. RNG and module modes are preserved.

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
