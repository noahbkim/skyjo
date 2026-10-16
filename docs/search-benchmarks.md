# Frozen search strategy comparisons

`run_search_strategy_benchmark.py` compares allocation, backup, final selection,
and learning targets using one frozen checkpoint. It writes a manifest, an
incremental `searches.jsonl`, and per-position summaries. No training runs occur.

`configs/benchmarks/search_positions.json` contains the two reported giveaways
(zero-based recording frames 53 and 112), the 17 previously frozen exact endgames,
and eight previously frozen replay controls stratified by phase and hidden-card
count. Each state is self-contained. Provenance includes recording, checkpoint,
and previous fixture hashes. The checkpoint remains a local input and its digest
must match the recorded iteration 119 checkpoint.

The GIVE → TAKE transitions are deterministic. The opponent's last replacement
reveals a hidden card, but always clears the visible 9/9/? or 12/12/? column. The
opponent's visible score drops 22→4 or 50→26 regardless of that reveal. The engine
regression tests check this distinction rather than freezing a random successor.
Avoiding a giveaway does not prove that a replacement action is optimal: another
replacement can reveal the same rank by chance.

The 17 exact endgame action-value references are rechecked using engine
enumeration, including optimal final placement after DRAW or TAKE. They measure
lost terminal win probability. The eight replay controls have no oracle and
only describe distributions, values, work, and timing; this small stratified
sample cannot establish population-level playing strength.

## Run the comparison

From the repository root, set `SEARCH_CHECKPOINT` to the iteration 119 `.pth`
path recorded in the fixture's `source.checkpoint_relative_path`, then run:

```sh
uv run python run_search_strategy_benchmark.py \
  --checkpoint "$SEARCH_CHECKPOINT" \
  --output .runs/search-strategy-tactical \
  --case giveaway_9 --case giveaway_12 --workers 6
```

The defaults run eight paired seeds, total traversal budgets 256/2,048/8,192,
enumerated chance, and a sampled-chance repeat at 2,048. Repeated-search variants
use eight replicas sharing a **total** budget of 2,048; change this with
`--ensemble-budget`. The same state, budget, and repetition produce the same
seed across variants. Noise in each search remains owned by its runner.

The matrix contains:

- `control`: existing settings, including parent-based unvisited values.
- `puct-{valuation}`: PUCT with zero unvisited values and visit targets.
- `gumbel0-{valuation}` / `gumbel1-{valuation}`: sequential halving, completed-Q
  interior allocation and targets; Gumbel scale 0 or 1 respectively.
- `{puct,gumbel}-{valuation}-replicas8-z{0,1,2}`: independent-search means minus
  `z` times their standard error. Gumbel replica variants use noise scale 1.

Each valuation is `current_mixture`, `mean_return`, or `best_response`.
The optional `gumbel-puct-{valuation}` variants combine a Gumbel root (noise
scale 1) with PUCT interior allocation; select them explicitly with `--variant`
to isolate the two allocation choices. They are excluded from the default matrix.
Use repeated `--variant`, `--case`, and `--budget` arguments to choose a subset;
`--chance enumerated` or `--chance sampled` disables the automatic repeat.
For example:

```sh
uv run python run_search_strategy_benchmark.py \
  --checkpoint "$SEARCH_CHECKPOINT" --output .runs/search-strategy-smoke \
  --case giveaway_9 --case giveaway_12 --budget 256 --seeds 1 \
  --variant control --variant puct-mean_return \
  --variant gumbel1-mean_return --chance enumerated
```

`benchmark(...)` is also directly callable from Python. Imports do not load a
model or start searches. The output directory must be new; completed rows are
flushed after each result and remain readable if a later trial fails.

## Read the output

Every trial records grouped priors, visits, value estimates, acting probabilities,
learning-target probabilities, optional replica standard errors, evaluator work,
boundary samples, and elapsed time. It also records the retained search visits
on GIVE → TAKE → CLEAR without adding evaluator calls. Diagnostics report both
the configured final action and a highest-value choice on the same tree.

The manifest separately re-evaluates the original recording's saved visits and
values: frame 53 changes from action 18 to 21 under value selection, gaining about
0.01339 estimated win probability; frame 112 still selects action 27. This
separates a final-selection failure from a value-estimation failure in the
original evidence.

The uncertainty penalty measures variation between independent searches. It
does not quantify shared model error. Likewise, simulation counts do not imply
equal model work: chance enumeration and boundary sampling can evaluate many
states in a single traversal, so compare evaluator-state counts and elapsed time
alongside the fixed total budget.

## Clearing-card AlphaZero baseline

`configs/benchmarks/clearing-alphazero.toml` compares the named `alphazero`
preset with uniform group priors, and repeats both with highest-estimated-Q
final selection. Allocation stays PUCT, valuation stays `mean_return`, and
learning targets stay visit-based in all four methods. Uniform priors apply at
every decision node, equally across symmetry groups. Q selection considers only
evaluated actions. All methods use sampled ordinary chance, one sample on the
first boundary visit, zero FPU, c_puct=1.5, no root noise, and temperature zero.
This is the named AlphaZero preset, **not** the historical recording's mixture
backup / enumerated chance / ten boundary samples configuration.

Run the existing per-iteration diagnostic runner on the two giveaway positions:

```sh
mkdir -p .runs/clearing-suite-from-exports
cp data/search-positions/clear-9-giveaway.json .runs/clearing-suite-from-exports/
cp data/search-positions/clear-12-giveaway.json .runs/clearing-suite-from-exports/
uv run python run_search_diagnostics.py run \
  --checkpoint "$SEARCH_CHECKPOINT" \
  --suite .runs/clearing-suite-from-exports \
  --plan configs/benchmarks/clearing-alphazero.toml \
  --output .runs/clearing-alphazero-8192
```

Use an empty suite directory; the exported giveaway states and annotations match
`giveaway_9` and `giveaway_12` in the benchmark exactly. The initial recorded run
used those benchmark IDs in otherwise equivalent exported positions. The
checkpoint must be the fixture's iteration 119 variant (SHA-256
`dcbe5ec51a529460de5c8f6ad91d37fcdef5e0e83f183474832976c3be5d976f`).
The diagnostic runner records but does not enforce that hash; verify it before
comparing a later run with this baseline.

The plan fixes seeds 0–3 and a maximum of 8,192 traversals per search. Inspect
`first_accepted_iteration`, `stable_accepted_from`, and the witness discovery
counts in `trials.jsonl`. “Stable from” means no later failure **within that
run**, not convergence beyond the tested horizon. A null recovery is censored
at the budget. Logarithmic trace snapshots show choices at smaller budgets on
the same tree; they are not independent repeats. Read successful-only recovery
medians alongside their success counts. Avoiding a labeled card is a diagnostic
criterion, not proof of optimal play.

### Frozen-evaluator faceoffs (next stage)

The four matching contestant files are in `configs/search-strategies/`.
For example, compare visits against Q selection at 256 traversals:

```sh
uv run python run_checkpoint_comparison.py \
  --control "$SEARCH_CHECKPOINT" \
  --control-settings configs/search-strategies/alphazero.toml \
  --variant-settings configs/search-strategies/alphazero-q.toml \
  --seed-count 32 --seed 20261016 --workers 4 --threads 1 \
  --runs-dir .runs/clearing-faceoffs
```

Omitting `--variant` deliberately reuses the control checkpoint. Use
`alphazero-uniform.toml` to isolate priors, or pair it with
`alphazero-uniform-q.toml` to isolate final selection under uniform priors.
All recipes fix the same mean-return valuation and next-deal evaluator; no
training is involved. These contestant files explicitly set 256 iterations, so
change both files to compare another budget (`--iterations` only supplies a
default when a settings file omits it).

Use both seats for every seed, report paired win credit and score-margin
intervals, and keep the tactical tuning seeds separate from strength evaluation.
Equal traversals need not mean equal evaluated states or elapsed time. Start
with the 64-game screen, then choose a separate confirmation budget before
claiming strength. These faceoffs have not yet been run for this baseline.
