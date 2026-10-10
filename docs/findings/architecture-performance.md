# Does the architecture cleanup regress CPU runtime?

**Conclusion, October 10, 2026:** these fixed workloads show no material runtime
regression. Transition and search time decreased substantially; inference and
optimization remained within the scale of short-run timing variation.

The comparison uses the self-play branch at `0ebb5f8` and the cleanup working
tree on the same local host: two players, a 16/32-dimensional model with two
attention heads, CPU inference, and one Torch thread. Each workload has one
warmup and three timed repetitions. Values below are median milliseconds;
ratios greater than one favor the cleanup.

| Workload | Before, ms | After, ms | Before/after |
| --- | ---: | ---: | ---: |
| 1,000 transitions | 230.80 | 36.67 | 6.29 |
| 1,280 inferred states, batches of 64 | 120.26 | 108.58 | 1.11 |
| 10 searches × 32 iterations | 491.23 | 224.95 | 2.18 |
| 20 optimizer updates × 64 positions | 226.58 | 192.10 | 1.18 |

Transition savings coincide with moving deep invariant validation out of every
transition; legal-action checks remain in the engine and full validation remains
explicit. Search uses the same starting-state generation and iteration budgets,
but its owned random generator changes traversal trajectories. This comparison
cannot isolate each change's contribution.

Three samples, sequential measurements, and one host do not establish a stable
performance ratio across environments. The modest inference and optimizer
changes should not be presented as proven improvements. These measurements say
nothing about training convergence, playing strength, GPU/MPS performance, or
worker scaling.

**Decision:** retain the separation and explicit validation. Use
`uv run python run_layer_benchmark.py` for a quick repeat of the fixed workloads; use the
replay-based MCTS benchmark and a profile when investigating a specific search
bottleneck. Compare playing strength separately when behavior changes.

[Evidence and reproduction details](evidence/architecture-performance.json)
retain every timing sample, original script text/hash, API adaptations, and
measured source hashes. The before/after JSON files were local task artifacts,
not recorded experiment runs. The after measurement preceded the review commits; the evidence identifies a
commit whose listed source files exactly match the measured digests.

The full baseline suite passed 315 tests; the cleanup suite passed 299 tests
in a multiprocessing-capable environment, with no failures or errors. Obsolete
interface assertions were replaced with behavioral coverage of the new
contracts. The evidence also records these results and validation scope.
