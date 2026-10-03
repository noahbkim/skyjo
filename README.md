# Skyjo

AI model, training, and gameplay for Skyjo

## Recorded experiments

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
scores are already part of the model observation. Round-score and future-clear
auxiliaries are disabled.

MCTS searches within the current round. On first reaching a round boundary, it
applies the final action once. A finished game supplies its exact outcome;
otherwise it deals the next round once and uses the model's prediction there.
That value is cached for subsequent visits to the same boundary node. This is a
single-sample baseline approximation: observed full-game value targets are never
resampled. Ordinary chance-node sampling during a round is unchanged.
Search utility is game-win probability alone.

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

This creates a new run with fresh model initialization and the saved seed; it is
not a resume. For historical reproduction, read the commit from `run.json`, create
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

The shared scalar and batched MCTS implementations, player implementations,
auxiliary models and losses, replay datasets, optimizer primitives, checkpoint
persistence, and run recorder remain available independently of the pool recipe.
The recipe exposes only its model dimensions, base loss scales, replay budget,
self-play/search settings, checkpoint schedule, and execution settings in
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
