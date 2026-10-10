# Skyjo

AI model, training, and gameplay for Skyjo

## How the code fits together

The maintained model is `EquivariantSkyNet`: its attention blocks summarize cards
within columns and columns within each board. It predicts full-game win
probabilities and masked action logits, with optional configured round objectives.
`ModelPlayer` chooses actions through scalar MCTS; `RandomPlayer` and `HumanPlayer`
provide a baseline and interactive gameplay.

The training path is:

1. `experiment_config` resolves the recipe; `models` constructs the network.
2. `distributed_main.py` is a thin CLI for `selfplay_training`, which distributes
   self-play games across CPU workers. Each worker uses `ModelPlayer` → `mcts` → `LocalPredictor` for local inference.
   The predictor returns results directly and chunks exact-chance evaluations
   without a separate process or request queue.
3. `game` owns rules and transitions; `play` collects complete games. `targets`
   and `objectives` build training labels, while `game_stats` summarizes results.
4. `buffer.ReplayBuffer` retains and samples complete-game data. `train` owns
   optimizer steps and shared evaluation; `experiment_training` records progress
   and artifacts. Startup builds the model and optimizer once and saves an explicit
   initial checkpoint.
5. `checkpoint` saves model/training state. `evaluation` compares checkpoints;
   `run_train_epoch.py` and `run_offline_comparison.py` train on saved replay data.

`game.Skyjo` is a named, frozen dataclass containing copy-on-write NumPy arrays.
Read fields by name; use `dataclasses.replace` to construct a modified state.
Setup reveals, ordinary turn reveals, and final-round reveals have explicit
operations. `observations` owns the persisted feature order and dimensions;
`batches` owns NumPy/tensor batches and float32 device conversion. Every batch has
one `targets` mapping. `losses` defines core losses, and `objectives` composes them
with the configured auxiliary losses.

The objective registry is shared by model heads, target generation, and losses.
Supported auxiliary objectives are `round_score`, `round_raw_score`, and
`round_doubled`. They affect training; search uses game-win probability alone.

### Retired interfaces

The older model players, batched tree search, predictor process/queue clients,
`SimpleSkyNet`, unused `ResidualAttentionBlock`, legacy auxiliary model subclasses,
future-clear objective, and standalone `stats`/`benchstats` utilities have been
removed. The active network still uses `TransformerBlock`. Historical notebooks
may require the original Git revision and its environment.

`SkyNetModelFactory`, `train_utils`, positional game-state tuples, and the
flag-based reveal API have also been removed. Observation helpers now live in
`observations`; training targets are mappings rather than tuple-like wrappers.
Historical notebook calls to these APIs or retired search helpers need updating.
The root command names remain supported. Programmatic callers use
`selfplay_training.launch(..., repository=checkout)` or
`experiments.launch_suite(..., repository=checkout)`; recorded runs still require
a Git checkout and its lockfile, regardless of the current working directory.

Use `LocalPredictor.predict(state)` or `predict_many(states)` for local inference.
The offline CLI uses `--auxiliary-objectives` instead of `--experiment-arm`;
omitting it trains only the core value/policy outputs, even when a dataset contains
extra labels. Enabled objectives require matching labels.

Current model parameter names and replay/checkpoint container formats are
unchanged. Loading retired model architectures and resuming older
`run_train_epoch.py` configurations are unsupported; use their original revision
for historical runs. Existing saved artifacts are not migrated. Checkpoint
resume now rejects missing configuration or requested optimizer, scheduler, or
RNG state before changing runtime objects. Weights-only evaluation remains
supported. Online training starts fresh; importing replay is explicit, and online
resume remains unsupported. Offline resume remains supported within a matching
configuration. Removing redundant model initialization changes incidental old RNG
consumption; seeded CPU workflows remain reproducible within the revised
implementation.

## Recorded experiments

### Offline round-boundary value experiment

`run_boundary_value_experiment.py` tests whether cumulative scores between rounds
predict eventual game wins. It compares logistic regression with a small
two-layer MLP using observed `metrics/rounds-*.jsonl` logs from one source run.
It does not change MCTS or self-play. Run from the repository root:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python run_boundary_value_experiment.py \
  --source-run .runs/20261004T023958Z-continue-search32-31b456dd3e \
  --config configs/boundary_value.toml --allow-dirty
```

The default recipe uses three training seeds and 200 epochs on a single CPU
thread. Inputs are cumulative charged scores divided by 100, in cyclic seat order
starting with the next round's starter. Targets are the observed full game's
eventual winners, with shared credit for ties. Completed-game rows provide labels
but are excluded as inputs; incomplete and partial-start games are counted and
excluded. Malformed or duplicate round records are rejected.

Games with nonterminal boundaries are split 70%/15%/15% into training,
validation, and test sets. All boundaries of a game remain together. Both models
fit soft-target cross entropy with paired minibatch orders; validation value MSE
selects the checkpoint. The test set is evaluated after selection. Reports include
value MSE (averaged over players), Brier score (summed over players), calibration
bins, and performance near the 100-point threshold. The paired uncertainty
interval resamples whole test games and conditions on the fitted models and split.

Each recorded run saves `report.md`, `results.json`, per-epoch training/validation
curves, selected model weights, test predictions, a compact boundary dataset,
source-log hashes, split identities, and an experiment source-code snapshot.
These boundary checkpoints are separate score evaluators, not replacements for
the ordinary policy/value gameplay checkpoint. They can now be used explicitly
by the optional search configuration described below.
The random holdout measures the recorded mixture of self-play policies;
generalization to future policies and any improvement in play require later
experiments. The 1/10/100 ending-sample comparison is also a separate experiment.

`run_boundary_methods.py` compares those fitted score models against the frozen
full-game network evaluated over 1, 10, and 100 fresh next-round deals. Use logs
generated after the checkpoint's training, by that same checkpoint; the runner
rejects earlier games, mismatched generating checkpoints, and overlap with the
boundary dataset. For the first overnight batch, which was generated by model50:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python run_boundary_methods.py \
  --round-log .runs/20261005T021353Z-overnight-search32-from50-3d081cbe76/metrics/rounds-000051.jsonl \
  --checkpoint .runs/20261005T021353Z-overnight-search32-from50-3d081cbe76/checkpoints/checkpoint_000000.pth \
  --boundary-run .runs/20261006T130144Z-boundary-value-a8bf832495 --allow-dirty
```

The default comparison evaluates every nonterminal boundary with three deal
replicates, sharing sample prefixes across budgets. Reports include paired
whole-game bootstrap intervals, per-replicate calibration, and batched CPU cost.
This measures held-out prediction, not whether changing targets improves
training or search strength. It holds the observed final round scores fixed.

`run_terminal_benchmark.py` separately tests final-action/reveal sampling against
exact enumeration and independent large simulation references:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python run_terminal_benchmark.py \
  --source-run .runs/20261005T021353Z-overnight-search32-from50-3d081cbe76 \
  --exact-cases 32 --sampled-cases 8 --reference-samples 8192 --allow-dirty
```

Exact one-hidden-card cases evaluate all legal actions and require every outcome
to end the full game. Multi-hidden cases test up to four disclosed candidate
actions, using separate screening, reference, and estimate random streams.
K=1/10/100 estimators share prefixes within each repetition. Continuing outcomes
retain their full probability mass in eventual-win bounds; they are never
discarded or given a learned label disguised as ground truth. The report gives
finite-simulation uncertainty and distinguishes exact action regret from regret
against an estimated reference. Both tools save inputs, predictions/outcomes,
provenance, and source snapshots in independent recorded runs.

### Self-play and replay experiments

The experiment workflow preserves the inputs and evidence needed for architecture
research. Config resolution belongs to the current distributed training recipe;
the reusable `skyjo.runs.RunRecorder` accepts arbitrary JSON-compatible metrics,
progress counters, context, and artifact metadata. New diagnostics do not require
a metric registry or changes to the recorder.

Run the small pipeline check first, then the full-game baseline recipe:

```sh
uv run python distributed_main.py --config configs/smoke.toml
uv run python distributed_main.py --config configs/baseline.toml
```

The smoke configuration performs two complete self-play games and optimizer updates
on CPU, then saves checkpoints and replay data. It checks the pipeline, not model
quality. The baseline runs 100 iterations of 256 full games (25,600 total),
with 32 search iterations per decision, eight single-threaded CPU workers, and
a 524,288-position replay buffer. Replay ratio 4 means four sampled training
positions per newly generated position, sampled with replacement; it is not an
epoch count. The prior 15,360-game run took 7h 41m. Its late throughput projects
roughly 14 hours for this budget, but changing game lengths affect runtime.
Evaluate saved checkpoints against fixed opponents with balanced seats to measure
playing strength; that evaluation is separate from this training recipe.

For an expanded buffer, optional `training.replay_ratio_after_fill` changes the
ratio after the first iteration that fills replay or evicts an old game. That
iteration still uses `training.replay_ratio`; the following iteration uses the
new ratio. The switch is recorded in `trajectory.jsonl`, and checkpoints preserve
the active ratio. Continuing with the same capacity and schedule retains it;
changing capacity or the schedule starts from the requested base ratio.

By default, launching requires a Git commit and no staged edits, tracked edits,
or non-ignored untracked files. During development, explicitly permit dirty code:

```sh
uv run python distributed_main.py --config configs/smoke.toml --allow-dirty
```

The commit, dirty flag, and override are recorded. Patches, changed-file lists,
and untracked source files are **not** saved; a dirty run may be impossible to
reconstruct. Ignored run outputs do not make the repository dirty.

Each launch prints a fresh `.runs/<run-id>/` directory (gitignored). Use
`--runs-dir PATH` to choose another local root. It contains:

| File or directory | Purpose |
| --- | --- |
| `run.json` | Identity, status, code/runtime provenance, config digest, final progress |
| `input-config.toml` or `.json` | Exact original input bytes |
| `resolved-config.json` | Rerunnable settings, expanded defaults, derived shapes, input dataset identity |
| `notes.md` | Editable description, observations, caveats, and findings |
| `trajectory.jsonl` | Ordered structured events and measurements |
| `artifacts.jsonl` | Artifact registrations and supersession events |
| `checkpoints/` | Existing Torch checkpoint format, with recorded SHA-256 checksums |
| `data/replay/` | Latest replay dataset in the existing NumPy format |
| `metrics/rounds-*.jsonl` | Immutable per-round observations with game/seed/checkpoint provenance |
| `metrics/concepts-*.jsonl` | Per-example heuristic concept checks |
| `logs/train.log` | Compact progress, iteration summaries, and failure tracebacks |

The baseline uses `EquivariantSkyNet`, the policy/game-win base loss, and core
replay targets. The ordinary runner also supports configured round objectives. All decisions across a
game receive its observed final winner label, with ties shared equally. Cumulative
scores are already part of the model observation. Auxiliary objectives are
disabled in the baseline.

MCTS searches within the current round. By default, on first reaching a round boundary, it
applies the final action once. A finished game supplies its exact outcome;
otherwise it deals the next round once and uses the model's prediction there.
That value is cached for subsequent visits to the same boundary node. This is a
single-sample baseline approximation: observed full-game value targets are never
resampled. Ordinary chance-node sampling during a round is unchanged.
Search utility is game-win probability alone.

Symmetric legal actions share one MCTS child by default. The exact action key
combines flip/replace, the target card state, and its column's card multiset.
Group priors are summed; positive-temperature policies distribute each group's
visit probability equally among its actions. At temperature zero, the lowest
index in the most-visited group is selected. Opening and draw/take choices stay
separate. A conservative search-budget check falls back to ordinary search when
deck recycling could break symmetry during final reveals. The simulator and
stored replay format are unchanged.

To disable sharing, set `merge_symmetric_actions = false` under `[search]`, or
pass `merge_symmetric_actions=False` to `mcts.run_mcts`. The comparison command
accepts independent `--no-control-merge-symmetric-actions` and
`--no-variant-merge-symmetric-actions` switches (and their positive forms).
Reusing a pooled root requires matching settings and a safe cumulative budget;
an unsafe extension raises before changing the tree.

The manual speed benchmark compares both modes on frozen positions using the
same gameplay weights, at 32 and 128 simulations and with both boundary methods:

```sh
uv run python run_mcts_benchmark.py \
  --control-replay /path/to/control/data/replay \
  --variant-replay /path/to/variant/data/replay \
  --checkpoint /path/to/control/checkpoints/checkpoint_000119.pth \
  --boundary-checkpoint /path/to/variant/data/boundary_value.pth \
  --allow-dirty
```

Defaults select 128 positions per replay with seed 20261009, one CPU thread,
warm-up, and three alternating paired sweeps. Each recorded run includes source
snapshots, input hashes, selected positions, raw timings, phase summaries,
separate node/inference diagnostics, and a `report.md`. This measures speed;
it imposes no performance gate and does not measure playing strength.

To replace the boundary approximation with sampled endings and a frozen score
evaluator, set these options in an ordinary training TOML:

```toml
[search]
boundary_samples = 10
boundary_value_checkpoint = "path/to/boundary-run/checkpoints/mlp-0.pth"
```

The path is relative to the declaring config file. Each visited final-action
edge samples the action and hidden reveals independently, computes charged
cumulative totals, and evaluates each result before averaging. Completed games
use exact shared winner credit; continuing games use the score evaluator before
any new deal. The mean is cached per edge. Fully visible deterministic endings
need only one outcome. The default remains one ending plus one next-round deal;
multiple samples require a score evaluator. This setting is independent of the
MCTS iteration budget and the number of training iterations.

Training validates the score model's player count and saves a frozen copy at
`data/boundary_value.pth` in each run, recording its source hash. Workers and
saved player settings reference that copy. The score model is not updated by
ordinary training; the main policy/value network continues to train normally
from observed complete-game results and search targets. The checkpoint comparison
runner uses the same legacy boundary evaluator for both sides by default, so it
can compare learned weights without granting only one side the new evaluator.

Every generated/replayed game count refers to a complete game, not a round.
Replay retains and evicts complete games, and dataset splits keep all rounds of
a game together. Full-game scores
and outcomes remain available alongside round statistics. Completed turns count
post-setup flips or replacements; a draw followed by a replacement is one turn
and two model decisions. Partial starting rounds are flagged and excluded from
whole-round length distributions. Score and clear distributions use player-rounds;
action rates pool counts and eligible opportunities. No-progress endings are
identified before automatic final reveals, and score adjustments compare raw
board points with the scored round points.

`logging.progress_interval_seconds` defaults to 0: one completion summary per
generation phase, including elapsed time, games/sec, and decisions/sec. Set a
positive interval (for example, 300 seconds) to also report progress and ETA while
generation runs. Task and target-conversion messages require `execution.debug = true`.
Game summaries show round averages and the no-progress ending rate; full
distributions, action rates, loss components, and phase timings remain in the
structured metrics and DEBUG logs. Training summaries
show sampled positions, optimizer updates, replay ratio, and replay-equivalent
passes. Structured policy diagnostics include position-weighted target entropy,
predicted entropy, and KL(target || prediction), overall and by decision phase.

`validation.concept_interval` defaults to 5 iterations, plus initialization and
completion; 0 disables checks. The current concept suite is two-player-only,
so games with more players skip it with an explicit log message. These handcrafted positions are heuristic concept
checks, not calibrated win-probability or playing-strength validation. Evaluating
them preserves training mode and RNG state. Checkpoint frequency is independent
of both reporting intervals.

Checkpoints are saved initially, every `budget.checkpoint_interval` iterations,
and at the final iteration. A final iteration on the periodic schedule is saved
once.

The pool runner is full-game-only. Single-round gameplay helpers remain available.
Optional `selfplay.start_state = "potential_clear"` starts a two-player game from that
position, then continues through later rounds; `standard` starts a fresh game.

Replay and checkpoint loading check artifact format, configuration, and tensor
shapes. They do not identify or guard against older training objectives; use fresh
artifacts for this baseline.

An optional `replay.initial_dataset` path is relative to the input config file.
The resolved config stores its absolute path and dataset ID; reruns reject a
different dataset at that path. Move an input dataset by updating its path while
retaining its identity. Output directories never appear as rerunnable inputs.

To repeat a run under the current compatible code, launch its saved settings:

```sh
uv run python distributed_main.py --config .runs/RUN_ID/resolved-config.json
```

Without `initial_checkpoint`, this creates a new run with fresh model initialization
and the saved seed. A continuation config instead starts again from its named parent,
not from the child run's latest checkpoint. Neither operation resumes a run in place.
For historical reproduction, read the commit from `run.json`, create
a separate Git worktree at that commit, run `uv sync --locked` there, restore any
required input dataset, and launch the saved JSON by absolute path. Saved code and
seeds do not guarantee bit-identical results across devices or runtime versions.
Only committed implementation changes can be recovered from the recorded commit.

An iteration generates games, adds their positions to replay, and trains for a
replay-ratio budget of optimizer steps. One cumulative progress record counts
iterations, generated games, optimizer steps, and sampled positions. The saved counters describe the saved
model's continuous training history. The shared checkpoint format is unchanged.
A snapshot reference contains only a checkpoint path and artifact ID. A null checkpoint reference
means the in-memory model has no exact saved checkpoint at that point. Loss means
are over optimizer steps; game statistics are means over the generated games.
Compare their definitions, data, and budgets before interpreting similarly named
metrics as equivalent.

Artifact paths are relative to the run directory. Replay is latest-only: apply
`superseded` records before resolving historical registrations. Supersession is
recorded before replacement; on save failure it conservatively marks the old
snapshot unavailable even if its files survived. Checkpoints and replay are saved
independently and do **not** constitute a coherent run recovery point.

Replay provenance records the initial buffer's path and dataset ID (when supplied),
the previous replay dataset ID, and the latest batch's run ID, generation iteration,
generating checkpoint, and game count. Generation iterations are one-based: games
for iteration 1 use the initial model, and games for iteration N use the model
after iteration N-1. Between saved boundaries the checkpoint ID and path are null;
an older checkpoint is never substituted for the actual unsaved model. The
generating checkpoint describes only that newly added batch, not the entire mixed
replay buffer. No per-game provenance index is maintained; these references
describe how the buffer was built, not precisely which sources remain after
eviction. The replay manifest therefore has no single `source_checkpoint` for the
whole snapshot.

Completed, failed, and caught interrupted runs have explicit lifecycle events.
An uncatchable termination may leave status `running` or an incomplete final JSONL
line; only complete lines are evidence. There is no automatic recovery service.

### Runtime budgets and checkpoint continuations

The ordinary `[budget]` supports `iterations` and `max_seconds`. Zero disables a
cap; at least one must be enabled. Existing configs default to `max_seconds = 0.0`.
Time starts at runner entry, including configuration/model/replay loading, worker
startup, generation, training, diagnostics, and saves. Every started iteration
finishes completely, then the runner stops if either cap has been reached. It
always registers a final checkpoint, even between periodic saves. A setup that
exhausts the budget leaves an initialized final checkpoint without generating games.
This is a soft time limit, not a deadline: the remainder of an iteration and
finalization can exceed it. Separate faceoffs do not count toward training time.

For a continuation, supply `initial_checkpoint` and `replay.initial_dataset` in
an ordinary run config. Paths resolve relative to the file declaring them, including
inherited files. Both inputs are required; use a stable replay snapshot (normally
the completed parent's final replay). The runner verifies compatibility and detects
replay replacement during loading. Inputs are never modified; a new run gets its
own replay and checkpoints. Initial replay is copied before generation, and its
child copy follows the ordinary latest-only retention policy.

Weights, Adam moments/step counters, saved RNG state, and cumulative progress are
restored. The child seed must match the parent's; legacy checkpoints obtain their
seed from hash-verified parent-run records, so retain those records beside the
checkpoint. New checkpoints retain the seed and next game index directly. Generation
continues with unused game indices. Startup concept checks preserve RNG state.
Model/player/head structure must match. Search, loss weights, batch size, replay
ratio, and learning rate may change; child optimizer settings take precedence over
saved settings without resetting moments. Offline sampler checkpoints and parent
schedulers are not supported as ordinary self-play continuations.

Iteration caps mean **additional** iterations in the child. For example, a parent
at iteration 40 with `budget.iterations = 5` produces checkpoints through 45. Set
`iterations = 0` explicitly for a time-only child, including when inheriting a config
with an iteration cap. Cumulative counters remain available alongside
`additional_*` progress. Unknown historical generated-position counts remain null;
additional positions are always recorded. The final `budget_completed` event reports
the stopping reason, actual invocation time, and overshoot. Parent checkpoint hashes,
replay identity, inherited counters, and config changes appear in `continuation_started`.

The checked-in `configs/continue_search32.toml` and
`configs/continue_search128.toml` continue the completed iteration-40 run with
32 and 128 search iterations respectively. They resolve the parent checkpoint
and replay under this checkout's `.runs/20261003T045231Z-long-selfplay-1024-fdc1843ade`;
change those two paths to use another compatible parent.

Each config uses four self-play workers and one Torch thread per worker and
learner. The higher-search config inherits all settings from the lower-search
config except its name and search budget. Keep the 1,024-game generation batch
and replay ratio unchanged so this only changes the available parallelism.
Four workers halves the configured worker count, not necessarily throughput:
concurrent learners, memory bandwidth, and different search costs also affect
wall time.

Launch each run separately in its own terminal. Both may run concurrently;
there is no paired launcher, scheduler, or automatic evaluation:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python distributed_main.py --config configs/continue_search32.toml --runs-dir .runs
```

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python distributed_main.py --config configs/continue_search128.toml --runs-dir .runs
```

Use `--allow-dirty` during development. Each invocation independently copies the
same parent state. Compare actual runtime, overshoot, generated positions, and
optimizer steps; an eight-hour setting does not guarantee equal compute or volume.

The next increments are curated Git-tracked experiment reports and findings,
coherent resume, and focused comparison/evaluation tools. Historical compatibility,
schema migrations, automated Git checkout, and dashboards are outside this milestone.

## Complete-run round objective comparisons

For a single baseline-budget training run with raw-score and doubling losses,
use the standalone configuration:

```sh
uv run python distributed_main.py --config configs/round_score_doubling.toml
```

This keeps the baseline model, search, replay ratio, and 25,600-game budget,
adds both auxiliary losses at weight `0.1`, uses observed round endings, and
enables the first-batch gradient diagnostic. Outputs go to a fresh `.runs/` directory.

The suite launcher calls the ordinary runner sequentially for each named variant
and paired training seed. Every invocation initializes fresh weights, replay,
checkpoints, and RNG streams, and generates its own self-play. Nested overrides
are validated by the ordinary configuration resolver before any training starts.
Paths are resolved relative to the file declaring them. Failures stop the suite;
completed child runs remain recorded and are not automatically rerun.

```sh
# Three variants, one training seed, two games per run, four evaluation games total.
uv run python run_auxiliary_ablation.py --config configs/round_objectives_smoke.toml

# Full training budgets: configured for later use, not part of the smoke check.
uv run python run_auxiliary_ablation.py --config configs/round_objectives.toml
```

The full suite inherits `configs/baseline.toml` and uses training seeds `[0, 1, 2]`.
Its variants are control (no auxiliaries), penalized score (`round_score = 0.1`),
and raw score plus doubling (both `0.1`). Add `--allow-dirty` for development runs.
Any combination of the three heads is supported in an ordinary training config:

```toml
[auxiliary_objectives]
round_raw_score = 0.1
round_doubled = 0.1
round_score = 0.1

[auxiliary_targets]
mode = "observed" # alternatively "resampled"
samples = 32      # used only for auxiliary terminal resampling

[training]
gradient_diagnostic = true
```

Omitted or zero-weight objectives create no head, replay label, or loss term.
Raw scores predict `raw / 144` with a linear head and MSE; doubling predicts
logits with BCE; charged scores retain the sigmoid head and `(score + 48) / 336`
normalization with MSE. Score errors are logged as MAE in points. All auxiliary
losses have separate unweighted and weighted metrics. The experimental `0.1`
weights are fixed; there is no automatic balancing.

The learner builds labels separately for each completed round, before replay
insertion. Observed endings are the default. Optional terminal resampling fixes
the round's last decision, samples that transition and its automatic reveals,
and applies clearing and penalties before averaging. Enabled heads share the
same endings; deterministic transitions run once. This does not simulate earlier
alternative continuations. Seed streams depend on training seed, global game
index, and round index. Full-game outcome labels, policy symmetrization, observed
statistics, subsequent rounds, and outcome-only MCTS utility stay unchanged.
Auxiliary initialization preserves core initialization and subsequent RNG streams.

Each experimental checkpoint is evaluated against the same training seed's
control. The default is 32 evaluation seeds with both seat assignments (64 full
games per pair), 128 MCTS iterations, temperature zero, and no Dirichlet noise.
The reusable `skyjo.evaluation.evaluate_checkpoints` accepts two versioned
checkpoint paths and an `EvaluationConfig`, restores caller RNG state, and
returns per-game identities, scores, shared-tie win credit, and summary metrics.
Evaluation currently requires two-player checkpoints, checked before suite launch.

Compare a frozen checkpoint at two search budgets without training:

```bash
uv run python run_checkpoint_comparison.py \
  --control /path/to/checkpoint.pth \
  --control-iterations 32 --variant-iterations 128 \
  --seed-count 16 --seed 0 --threads 1 --runs-dir .runs
```

Omitting `--variant` uses the control checkpoint for both players. To compare
training progress instead, supply `--variant /path/to/later-checkpoint.pth` and
use `--iterations 32` for both players. Either per-player override falls back to
`--iterations` (default 128). Add `--allow-dirty` for an uncommitted worktree.
The library accepts the same overrides in `EvaluationConfig`; existing shared
budgets keep their behavior.

Compare the same checkpoint's greedy policy against MCTS32 over 64 games:

```bash
uv run python run_checkpoint_comparison.py \
  --control /path/to/checkpoint.pth --control-policy-only \
  --iterations 32 --seed-count 32 --seed 9300 \
  --workers 8 --threads 1 --runs-dir .runs
```

`--control-policy-only` and `--variant-policy-only` independently select the
network's highest-scored legal action without search. These flags preserve the
same checkpoint weights; boundary evaluator settings are invalid for a
policy-only player. Reports identify each side in `play_mode_by_player` and set
its `search_by_player` entry to null when search is disabled.

Compare the full sampled boundary evaluator with the original evaluator using
the same frozen gameplay checkpoint:

```bash
uv run python run_checkpoint_comparison.py \
  --control /path/to/checkpoint.pth --iterations 32 \
  --variant-boundary-samples 10 \
  --variant-boundary-value-checkpoint /path/to/boundary_value.pth \
  --seed-count 16 --seed 0 --threads 1 --runs-dir .runs
```

Both players default to one sampled ending and the original next-deal evaluator.
Each side has independent `--control-boundary-samples` /
`--variant-boundary-samples` and `--control-boundary-value-checkpoint` /
`--variant-boundary-value-checkpoint` options. More than one sample requires that
side's score evaluator. Supplying an evaluator with one sample supports isolating
the effect of sample count in a separate comparison. Reports include the frozen
boundary evaluator paths and hashes. Equal MCTS iterations do not imply equal
evaluation time.

Each seed plays both seats, so 16 seeds produce 32 full games. Evaluation uses
CPU inference, temperature zero, no Dirichlet noise, and outcome-only search.
Seeds are common starting seeds, not guaranteed identical card sequences after
search and actions diverge. `search_by_player` records the actual settings for
each participant; the older `search` field retains the shared defaults.
Add `--workers 8 --threads 1` to run eight worker processes, each using one
PyTorch CPU thread. `--seed-count` is the total number of seed pairs across all
workers, so `--seed-count 128 --workers 8` still plays 256 full games. Worker count
does not change the evaluation seeds or search settings. Each worker loads its
models once; the parent records progress and writes one combined report, with
games ordered by seed and seat. Progress can arrive out of seed order.

The CLI prints progress for returned games and creates a recorded run containing
checkpoint paths/hashes, per-game scores and win credit in `trajectory.jsonl`,
and the final `comparison.json`. Completed game records survive interruption.
Positive control-minus-variant score margins favor the variant; ties split win
credit. Small comparisons screen for large effects, not proof of equal strength.

The parent recorded run contains child configurations, independent child runs,
per-pair comparison artifacts, and `comparison.json`. The report contains per-seed
win fractions and control-minus-variant score margins, then averages over training
seeds. It also retains elapsed training time, generated games/positions, optimizer
steps, sampled positions, and phase timings. Evaluation time is separate. Equal
iterations do not guarantee equal compute or sampled-position budgets.

Comparison configs enable a first-actual-batch diagnostic of shared-network
weighted core and unweighted/weighted auxiliary gradient norms and their ratios.
It reuses the forward graph and does not sample a batch, write `.grad`, advance
RNG, or take an extra optimizer step. Diagnostic time is recorded separately
(and remains included in elapsed training time). Smoke results only establish
working integration, not improved playing strength.

Shared-data diagnostics remain available through the offline trainer:

```sh
uv run python run_train_epoch.py PATH_TO_REPLAY --steps 10 \
  --auxiliary-objectives '{"round_raw_score": 0.1, "round_doubled": 0.1}'
```

The offline selection uses the same heads and losses, requires its named labels,
and accepts extra unused replay labels. Online replay compatibility continues to
use the ordinary runner's versioned dataset checks. No legacy dataset migration
or sidecar checkpoint format is introduced.

## Usage

Install the project in editable mode while developing:

```sh
uv sync
```

Then import the core game API directly from `skyjo`, or import supporting
modules from the package:

```python
import skyjo as sj
from skyjo import play, skynet

state = sj.new(players=2)
model_cls = skynet.EquivariantSkyNet
```

## Full games and individual rounds

Play a full game with the existing player implementations:

```python
from skyjo import play, player

result = play.play_game([player.RandomPlayer(), player.RandomPlayer()])
print(result.final_scores)  # Cumulative scores in fixed player order
print(result.winners)       # All winning player indices, including ties

for round_result in result.rounds:
    print(round_result.round_scores, round_result.cumulative_scores)
    print(round_result.ending_player)
    history = round_result.history  # Decisions followed by the final snapshot
```

A game ends after a fully scored round brings any cumulative score to **100
or more**. The lowest cumulative total wins, with shared winners for ties.
Player zero starts the first round; the player who ends a round starts the
next one. Existing round penalties still apply.

`play.play_round(players)` plays one round and returns a `RoundHistory` of
`RoundHistoryEntry` decisions followed by a terminal snapshot. Both gameplay
functions accept an optional `start_state`. Use
`play.game_result_to_game_data(result)` for observed full-game win/policy labels;
every round's terminal snapshot is excluded from training rows.

For direct state control, `sj.get_round_over(state)` identifies a completed
round, while **`sj.get_game_over(state)` now checks the full-game threshold**.
`sj.apply_action()` stops at the completed round; call
`sj.start_next_round(state)` explicitly to reset and deal another round. It
raises `ValueError` if the round is still active or the game has finished.
Completed rounds have no legal actions and remain available for inspection.

`sj.get_game_scores(state)` returns cumulative scores in current-player
order; `sj.get_fixed_perspective_game_scores(state)` uses fixed player order.
During a round they include only previous rounds; at completion they also
include the current round's penalized points. Reading them never mutates the
state. The stored `GAME_SCORES` slots always exclude the current round.

`sj.get_round_about_to_end(state)` indicates that the next action ends the
round. Existing winner helpers report round winners; use `GameResult.winners`
for full-game winners.

The separate interactive gameplay application is also available:

```sh
uv run python -m skyjo2 interactive random
```

## Reusable components

Scalar MCTS, local inference, players, configured objectives, replay datasets,
optimizer primitives, checkpoint persistence, and the run recorder are available
independently of the pool recipe. The recipe exposes model dimensions, objective
weights, base loss scales, replay budget, self-play/search settings, checkpoint
schedule, and execution settings in
`configs/baseline.toml` and `configs/smoke.toml`.

Handcrafted concept-check positions and their expected policies remain in
`skyjo.explain.VALIDATION_EXAMPLES`. For a standalone inspection of a loaded model,
call `explain.validate_model_on_validation_examples(model)` with INFO logging
enabled to see predictions and target comparisons. Their value targets are
heuristic round-level expectations, not calibrated full-game win probabilities.
The recipe also records compact checks on the configured concept schedule.

## Offline training

Replay buffers are saved as versioned NumPy dataset directories containing only
retained positions and complete-game metadata. Legacy replay pickle files are
not supported.

Train to an exact cumulative optimizer-step target with a deterministic
game-level validation split:

```sh
uv run python run_train_epoch.py .runs/RUN_ID/data/replay \
  --steps 100 \
  --output-checkpoint models/offline/step_100.pth
```

Resume by passing the prior checkpoint and a larger cumulative target:

```sh
uv run python run_train_epoch.py .runs/RUN_ID/data/replay \
  --checkpoint models/offline/step_100.pth \
  --steps 200 \
  --output-checkpoint models/offline/step_200.pth
```

## Compare run configurations on fixed replay

Use ordinary run configs directly to compare architecture, loss, and supported
training settings without generating new self-play. For example:

```sh
uv run python run_offline_comparison.py \
  --config configs/round_score_doubling.toml \
  --config configs/capacity_medium.toml \
  --config configs/capacity_large.toml \
  --dataset /path/to/saved/replay \
  --seeds 0,1,2 \
  --steps 2000 \
  --runs-dir .runs
```

The medium and large configs inherit the original 16/32 model and change its
embedding dimensions to 32/64 and 64/128. Ordinary TOML or JSON run configs can
specify one parent with `extends`; nested settings override the parent before
normal defaults and validation. Paths are relative to the file declaring them.
This works for both self-play and offline comparisons:

```toml
extends = "round_score_doubling.toml"
name = "my-wider-model"

[model]
embedding_dimensions = 32
global_state_embedding_dimensions = 64
```

Inherited auxiliary weights can be disabled by explicitly setting them to zero.
An empty table does not erase inherited keys. Future architectures register their
constructor and settings validator in `skyjo.models`; comparison orchestration
and checkpoint readers use that registry. No additional architecture is supplied
by this change.

The comparison owns the dataset, paired seeds, and optimizer-step budget. Each
config supplies the model, configured losses, batch size, learning rate, device,
thread count, and optional first-batch gradient diagnostic. Self-play, replay-ratio,
and iteration-budget settings do not drive offline training. The explicit dataset
also replaces any `replay.initial_dataset` reference in the run config. Models
start fresh; these comparisons do not resume prior training.

A suite-owned snapshot freezes the input data. The default game-level validation
fraction is 0.1 (`--validation-fraction`), with `--split-seed 0`. Evaluations occur
at step zero, every 200 steps (`--evaluation-interval`), and at completion. Curves
use a fixed training probe of up to 8,192 positions and the full validation split;
full training-set metrics are recorded separately at completion. Sampling is
independent of model initialization and evaluation. Equal batch sizes get identical
minibatches for each paired seed. Different batch sizes consume the same seeded
stream but receive different numbers of examples at the matched step budget.

The recorded parent run contains `comparison.json`, long-form `curves.csv`,
`curves.jsonl`, the snapshot, split membership, and isolated child runs with final
versioned checkpoints. Input configurations, inherited source contents, resolved
settings, parameter counts, sampled positions, and timing are recorded. The first
config is the control; paired differences are variant minus control, so negative
error differences favor the variant. Unavailable heads have no metric; weighted
totals across different loss configurations are not directly comparable. Recorded
known-board MAE counts player examples, not positions.

Training time excludes the separately reported gradient diagnostic; evaluation
and child elapsed times are also reported. Matched steps do not imply matched
sample exposure or compute. This measures fixed-data learning, not playing strength.
Completed child artifacts survive failures; restarting the command starts a new
suite rather than silently resuming. Use `--allow-dirty` when intentionally testing
uncommitted code. A minimal wiring check can use `--seeds 0 --steps 2
--evaluation-interval 1` with a small saved dataset.
