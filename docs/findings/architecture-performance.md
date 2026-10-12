# Does the architecture cleanup regress CPU runtime?

**Conclusion, October 10, 2026:** no material CPU regression in these workloads.
Transitions and search became substantially faster; inference and optimization
changes were too small to distinguish from short-run variation.

Self-play revision `0ebb5f8` versus the cleanup, on one host: two players,
16/32-dimensional model, two attention heads, one Torch thread. Medians of three
repetitions after warmup; ratios above one favor the cleanup.

| Workload | Before, ms | After, ms | Before/after |
| --- | ---: | ---: | ---: |
| 1,000 transitions | 230.80 | 36.67 | 6.29 |
| 1,280 inferred states, batches of 64 | 120.26 | 108.58 | 1.11 |
| 10 searches × 32 iterations | 491.23 | 224.95 | 2.18 |
| 20 optimizer updates × 64 positions | 226.58 | 192.10 | 1.18 |

Moving deep invariant validation out of every transition coincided with the
largest saving. Cheap legal-action checks remain. Search budgets and starting
states matched, but the new RNG changed traversal; individual causes are not
isolated.

Three sequential samples on one host cannot establish portable speedups,
playing strength, convergence, or accelerator/worker scaling.

**Decision:** retain explicit validation. Repeat with
`uv run python run_layer_benchmark.py`; profile replay-based search before further
optimization and evaluate strength separately.

[Evidence](evidence/architecture-performance.json): samples, scripts, source
hashes, matching cleanup revision, and suite results. Timings came from local
task artifacts rather than recorded runs.
