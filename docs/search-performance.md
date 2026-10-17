# Search performance experiments

`run_search_performance.py` is an experimental, CPU-only runner. It does not
change production search, training, checkpoint formats, or defaults. Its two
PoCs live in `skyjo.experiments.search_performance_poc`:

- `mode`: a private frozen inference model enters evaluation mode once.
- `packing`: fill reusable float32 observation buffers directly.
- `sharing`: wrap NumPy arrays with CPU tensor views instead of copying them.
- `inference`: all three inference changes together.
- `snapshots`: cache static node arrays and reuse unchanged immutable snapshots.
- `combined`: inference and snapshot changes together.
- `baseline`: the existing `LocalPredictor` and tree implementation.

The inference alternatives clone the supplied model, so they must not be used
as a drop-in replacement for a predictor sharing a live training model.
Returned predictions remain valid after buffer reuse. Snapshot caches compare
contents, including in-place array updates; old snapshots retain immutable
copies. This deliberately conservative cache spends time constructing keys and
retains more memory. It is not a versioned or array-backed tree implementation.
Snapshot substitution is scoped to an experimental context, restored on error,
and process-local. It is **not thread-safe**. No production caller installs it.

## Freeze inputs

Choose a group-normalized checkpoint, the resolved JSON configuration for its
run, and a stable replay dataset directory. The following command copies the
checkpoint/configuration and exports 128 self-contained states:

```sh
uv run python run_search_performance.py --stage prepare \
  --checkpoint "$CHECKPOINT" --config "$RESOLVED_CONFIG" --replay "$REPLAY" \
  --output .runs/search-performance-inputs
```

Selection uses seed 20261017 and balances phase, hidden-card-count bands, and
round boundaries from 8,192 uniformly sampled candidate rows. This is a
**stratified diagnostic sample**, not a population-weighted throughput estimate.
The input record includes replay row/game IDs, the replay manifest hash, complete
selected states, checkpoint/configuration hashes, and the selection distribution.
It does not hash or copy the entire replay dataset. Subsequent stages read only
the frozen inputs and verify their hashes.

## Run separately

Each output directory must be new. Run on an idle machine for clean timing;
avoid running these stages or another benchmark concurrently.

```sh
# Primary paired comparison: 128 positions, 32/128/256 traversals, three sweeps.
uv run python run_search_performance.py --stage searches \
  --output .runs/search-performance-searches

# Individual inference ablations.
uv run python run_search_performance.py --stage searches --budget 128 \
  --variant baseline --variant mode --variant packing --variant sharing \
  --variant inference --output .runs/search-performance-ablations

# Attribution and memory measurement are excluded from wall-time comparisons.
uv run python run_search_performance.py --stage profile --positions 16 \
  --output .runs/search-performance-profile

# Inference-only upper-bound evidence for batching independent games.
uv run python run_search_performance.py --stage batches \
  --variant baseline --variant inference --output .runs/search-performance-batches

# Eight complete games at 32 traversals, with 1/4/8 workers.
uv run python run_search_performance.py --stage games --budget 32 --sweeps 1 \
  --variant baseline --variant inference --output .runs/search-performance-games
```

Use `--inputs` to select another frozen bundle. `--variant`, `--budget`, and
`--workers` are repeatable. For profile/games, the last budget is used; searches
use every supplied budget. Defaults for games are expensive: three sweeps at
256 traversals. Each worker uses one Torch intra-op and one inter-op thread.
The public `benchmark(...)` function is directly callable; importing the script
does not load models or start work.

## Interpret evidence

Search trials alternate variant order by sweep and use fixed paired seeds.
They first compute baseline reference signatures outside timing. Every trial
must exactly match root values, priors, visits, availability, acting and training
policies, evaluator batch counts, symmetry fallback status, and final RNG state.
Timing excludes signature calculation and JSON output, includes search result
construction, and leaves garbage collection enabled. Tiny common evaluator
counting overhead remains in every timed variant.

Game trials compare full trajectories, training policies, final RNG streams,
and scores across variants and worker counts. Pool startup, per-game model
construction, and output transfer are included in pool wall time; per-game
search/play CPU and wall time are also recorded. One game per task mirrors the
overnight configuration. Use identical game IDs for scaling comparisons. Small
game sets and a single sweep are feasibility screens, not stable capacity claims.

Profiles contain inclusive and exclusive function times: do not add inclusive
parent and child times together. `tracemalloc` reports live allocations and peak
traced memory for four retained trees; it misses some native Torch memory and
does not count all allocations made and freed over time. RSS is a process
lifetime high-water mark, not an independent per-variant measurement in the
profile stage. Game pools use fresh processes for each configuration.

Batch-size measurements repeat one frozen state, isolating inference scaling.
They record numerical drift relative to single-state inference. They do not
implement a batched search scheduler or establish search-equivalent trajectories.

Manifests record source hashes, Git revision/status, runtime versions, host load,
and completion/failure status. Partial JSONL results survive errors. Raw profiles,
frozen binaries, and measurements belong under `.runs/`; summarize conclusions
and selected numerical evidence under `docs/findings/`.
