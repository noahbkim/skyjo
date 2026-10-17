# Search throughput: inference setup, copying, and snapshot caching

**Recommendation:** pursue a small production follow-up that establishes frozen
inference ownership and sets evaluation mode once. Do not promote the snapshot
cache. Batching independent searches is the strongest larger architectural
candidate, but needs its own scheduling and numerical-equivalence experiment.

Production code and defaults are unchanged. Both PoCs remain explicitly selected
experimental alternatives. No training, model-quality comparison, or merge into
production was performed.

## Workload and limitations

Measurements used an Apple M1 Pro with 10 logical CPUs, Python 3.12.7, NumPy 2.1.3,
Torch 2.5.1, and one inference thread. **The user's eight-worker overnight
training remained active throughout.** Wall-time ratios describe this contended
host; CPU time is a useful cross-check, not a substitute for an idle rerun.

The checkpoint is the frozen `checkpoint_000000.pth` from
`overnight-group256-from19`, with its resolved group-normalized model and search
settings: sampled chance, mixture backup, grouped actions/noise, one boundary
sample, c_puct=1, FPU reduction=0, and temperature=1. The replay source is the
completed fixed-budget group pilot. Checkpoint/configuration hashes, replay
provenance, and selected row/game IDs are in the evidence.

The 128 states balance 12 phase/hidden-count/boundary strata from 8,192 seeded
candidate rows, including 21 round-boundary positions. This is diagnostic
coverage, not a population-weighted sample. Each variant used the same states
and RNG seeds. Three timed sweeps reversed variant order on alternating sweeps.
Profiling, memory tracing, and inference microbenchmarks ran separately.

## Measured search results

Ratios above one favor the variant. These are medians of paired whole-sweep
throughput ratios, not averages of per-position speedups.

| Variant | 32 traversals, wall / CPU | 128 traversals, wall / CPU | 256 traversals, wall / CPU |
| --- | ---: | ---: | ---: |
| Inference PoC | 1.142 / 1.122 | 1.090 / 1.090 | 1.138 / 1.111 |
| Snapshot cache | 1.042 / 1.013 | 1.009 / 0.999 | 1.034 / 1.014 |
| Both PoCs | 1.130 / 1.111 | 1.062 / 1.047 | 1.084 / 1.098 |

A separate 128-traversal ablation isolates the inference changes:

| Change | Wall throughput ratio | CPU throughput ratio |
| --- | ---: | ---: |
| Set evaluation mode once | 1.128 | **1.110** |
| Direct observation packing | 1.002 | 1.005 |
| Share NumPy storage with CPU tensors | 1.003 | 1.005 |
| All three | 1.085 | 1.091 |

The mode-only result corresponds to about **10% less CPU time**. Combining the
changes was not reliably better than mode-only. Packing and sharing alone are
too close to measurement variation to recommend production changes from this
evidence. The early 16-position screen is retained locally but excluded from
these conclusions.

All **6,528 primary and ablation trials** matched baseline exactly for root
values, priors, visits, acting/training policies, evaluator batch counts,
symmetry fallback status, and final RNG state. Repetitions reuse the same seeds;
they are timing repetitions, not independent playing-strength evidence.

## Why these results make sense

In the baseline 16-position, 256-traversal profile, neural forward computation
accounted for 48.0% of profiled time; repeated recursive `model.eval()` calls
accounted for 14.9%; observation packing plus tensor conversion accounted for
3.6%; engine transitions for 9.0%; and tree snapshots for 6.9%. These are
inclusive function shares: model forward includes group-policy normalization
(7.8%), and evaluator time includes encoding and mode setup. Do not add
overlapping categories. cProfile also distorts relative costs, so use unprofiled
paired timings to estimate gains.

The copying hypothesis identifies real work, but not the dominant bottleneck in
this workload. Baseline observations already start as NumPy-backed engine state;
extra casts, stacks, and tensor copies are a small part of total search time.

The conservative snapshot cache had **zero full-snapshot reuses in 22,292 view
calls**. Static arrays were reused 18,692 times, but search normally changes node
statistics between view requests. Content-key checking raised profiled view time
from 0.450 to 0.548 seconds. Four retained trees increased traced peak memory
from **13.34 to 16.40 MiB**, about **23%**, and live traced allocations from
98,234 to 142,654. The inference PoC remained at 13.33 MiB. Tracemalloc misses
some native Torch memory; these are not whole-process peak-memory estimates.

Keep the unsuccessful cache as evidence. A versioned or array-backed tree is a
different proposal, not a conclusion that this cache can simply be switched on.

## Complete games and larger alternatives

Eight fixed complete games, 32 traversals per decision, one game per task,
produced the same trajectories, RNG streams, evaluator work, scores, and
**2,788 decisions** under both variants and all worker counts:

| Workers | Baseline seconds | Inference PoC seconds | Throughput ratio |
| --- | ---: | ---: | ---: |
| 1 | 88.81 | 80.87 | 1.098 |
| 4 | 35.46 | 32.70 | 1.084 |
| 8 | 28.04 | 24.09 | 1.164 |

Pool startup, model construction, and result transfer are included. This is one
paired sweep of eight games at 32 traversals, not a sustained 256-traversal
capacity measurement. Do not choose a production worker count from these
contended-host results.

Larger changes ranked for follow-up:

1. **Batch independent games/searches within a worker.** The inference PoC
   processed approximately 1,752 states/s at batch size 1, 9,797 at size 16,
   10,859 at size 64, and 11,150 at size 128. This is roughly 6.2× evaluator
   throughput at size 64. With evaluator work occupying 63.6% of its profiled
   search time, an optimistic zero-scheduling-cost estimate is about **2.1×
   overall throughput**. This is an upper-bound calculation, not a measured
   search speedup. The microbenchmark repeats one frozen state; batching changed
   value predictions by up to 2.98e-8 even there. Next, prototype resumable
   searches with one outstanding evaluation per tree, batching across games
   while retaining each game's RNG and traversal order. Measure queue/memory
   costs and check whether numerical drift changes actual choices or targets.
2. **Array-backed tree statistics with explicit mutation versions.** Remove
   repeated dictionary/list-to-array reconstruction from built-in selection and
   backup. Preserve owned immutable `NodeView` snapshots for diagnostics and
   custom strategies. This requires a separate internal interface and mutation
   audit; the failed content-key cache does not establish its speedup.
3. **Compiled fixed-shape inference.** Measure whether a suitable backend can
   reduce module, reshape, and group-normalization overhead. Treat compilation
   startup, shape specialization, backend support, and exact output comparison
   as acceptance criteria. No compiler/backend speedup was measured here.
4. **Compact engine state or compiled transition kernels.** The padded table
   stores 8×3×4×17 int16 entries: 3,264 bytes before other state. Card IDs could
   store the table in 96 bytes, or 24 for an active two-player board, with an
   observation adapter preserving checkpoint/replay layouts. This may matter
   for memory and broad trees, but transition time is only about 9% of this
   profile, limiting its first-order throughput payoff. Preserve copy-on-write
   ownership, current-player orientation, chance enumeration, and RNG semantics.

The first production follow-up should be the smaller frozen-model ownership
change. Leave packing/sharing optional until a quieter or batched workload shows
a material benefit. The larger batching experiment deserves separate design
and validation rather than expanding these PoCs into a scheduler.

## Reproduce and retain

Final validation passed **114 focused tests**, including 2/4/8-player packing,
partial batches and buffer reuse, sampled/enumerated chance, continuing and
terminal boundaries, all three valuation families, Gumbel and replica search,
recycling fallback, retained snapshot immutability, and existing multiprocessing
generation tests. The verified quick preset added 128 exact-parity trials; its
inference PoC measured 1.075× wall and 1.081× CPU throughput on the same busy host.

See [runner instructions](../search-performance.md) for freezing inputs, all
commands, variants, and measurement caveats. The short rerun is:

```sh
uv run python run_search_performance.py --stage searches --quick \
  --output .runs/search-performance-quick-02
```

It reuses `.runs/search-performance-inputs`, checks exact parity, and emits
`summary.json` plus paired wall/CPU ratios. Use a new output folder. The recorded
runs reused the primary checkout's installed Python environment with
`PYTHONPATH=src`; the documented `uv run` route may first need environment setup.

[Numerical evidence](evidence/search-performance.json) includes provenance,
paired samples, profiles, memory/cache measurements, batch results, game results,
and hashes of raw artifacts. Full frozen inputs, profiles, and JSONL records
remain under `.runs/search-performance-*` in this worktree; preserve those ignored
artifacts before archiving it. Experimental implementations are retained in Git.
