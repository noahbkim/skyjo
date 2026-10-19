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

## Learned/uniform priors and visit/Q selection

`configs/benchmarks/clearing-alphazero.toml` compares four methods using the
existing position-diagnostic runner: learned or uniform group priors, each with
visit-based or highest-Q final selection. All retain PUCT, mean-return valuation,
zero FPU, c_puct=1.5, sampled ordinary chance, one boundary sample and visit targets.
Uniform priors apply at every decision node; Q selection considers evaluated
choices only. This is the named AlphaZero preset, not the historical mixture
search. These are comparison recipes, not recommended training changes.

```sh
uv run python run_search_diagnostics.py run \
  --checkpoint "$SEARCH_CHECKPOINT" \
  --suite data/search-positions \
  --plan configs/benchmarks/clearing-alphazero.toml \
  --output .runs/clearing-comparison
```

The plan uses seeds 0–3 and 8,192 visits. Discovery, first acceptance and stable
acceptance within that horizon are different measurements. The suite also includes
TAKE/CLEAR positions; the archived initial experiment selected only the giveaways.
[Findings and archived provenance](findings/search-budget.md) preserve that result.

Four matching contestant files live in `configs/search-strategies/`. To isolate
visit-versus-Q acting with the same frozen model and 256 visits:

```sh
uv run python run_checkpoint_comparison.py \
  --control "$SEARCH_CHECKPOINT" \
  --control-settings configs/search-strategies/alphazero.toml \
  --variant-settings configs/search-strategies/alphazero-q.toml \
  --seed-count 128 --seed 20261016 --workers 8 --threads 1 \
  --runs-dir .runs/search-faceoffs
```

Omitting `--variant` reuses the control checkpoint. These files specify 256 visits;
edit both to change the budget (`--iterations` only fills an omitted value).
Each seed plays both seats, so 128 seeds produce 256 games. Use the uniform recipes
to isolate priors, keeping evaluator and valuation fixed. Compare evaluator work
and elapsed time as well as visits; tactical success is not a strength result.

Add `--first-round` to stop at the first round boundary from zero cumulative
scores. This preserves normal scoring, including doubling, and reports round
scores plus paired score-margin/win-credit summaries. Search still evaluates
full-game outcomes: this measures opening-round performance, not full-game
strength. Use separate fresh seeds for a confirmation comparison.
