# Skyjo

AI model, training, and gameplay for Skyjo

## Recorded experiments

The experiment workflow preserves the inputs and evidence needed for architecture
research. Config resolution belongs to the current distributed training recipe;
the reusable `skyjo.runs.RunRecorder` accepts arbitrary JSON-compatible metrics,
progress counters, context, and artifact metadata. New diagnostics do not require
a metric registry or changes to the recorder.

Run the small pipeline check first, then the full score-auxiliary recipe:

```sh
uv run python distributed_main.py --config configs/smoke.toml
uv run python distributed_main.py --config configs/score_aux.toml
```

The smoke configuration performs two real self-play games and optimizer updates
on CPU, then saves checkpoints and replay data. It checks the pipeline, not model
quality. The full configuration preserves the existing 10-iteration, 1024-games-
per-iteration recipe and is substantially more expensive.

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
| `logs/train.log` | Verbose training diagnostics and failure tracebacks |

Config sections cover model, training, self-play, search, replay, validation,
faceoff, budget, and execution settings. Supported model types are `equivariant`,
`round_score`, and `auxiliary`; loss types are `base` and `auxiliary`; replay
target selections are `core`, `round_score`, and `auxiliary`. Set both auxiliary
scales to zero for base loss. The recipe validates incompatible heads and targets
before allocating replay arrays or starting workers. Built-in validation and
faceoff are two-player recipes. `selfplay.start_state` can be `standard` or
`potential_clear` (two players).

Training continues from the current model regardless of evaluation results.
The initial model is saved with optimizer state, configuration, and progress.
After training, a checkpoint is saved every `budget.checkpoint_interval`
iterations and at the final iteration. A final iteration already on the schedule
is saved and evaluated once.

Built-in validation runs on the initial checkpoint as `initial_validation`, then
on each saved checkpoint as `validation` when enabled. In this recorded recipe,
`validation.interval` remains accepted but the checkpoint schedule controls
validation timing. Existing validation loss scales still apply.

With faceoffs enabled (`faceoff.paired_rounds > 0`), `faceoff.interval` must match
`budget.checkpoint_interval`. Each saved trained model plays against the last
passing checkpoint, initially the starting model. More candidate wins than
reference wins is a pass; ties fail. A pass advances the evaluation reference.
A failure records the result and training continues without restoring weights or
optimizer state. Faceoff events identify both checkpoints, the protocol, win
counts, and `passed`. Setting paired rounds to zero disables faceoffs, while
saving and enabled built-in validation still run. Paired-game seeds are global
pair indices, starting at zero, independent of `faceoff.rounds_per_task`.

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
iterations, generated games, optimizer steps, and sampled positions. Since
evaluation never rolls back training, the saved counters describe the saved
model's continuous training history. The shared checkpoint format is unchanged.
A snapshot reference contains only a checkpoint path and artifact ID; the last
passing reference is separate from the current model. A null checkpoint reference
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

`play.play_round(players)` plays just one round. `play.play(players)` remains
a compatibility alias, and existing training, search, and faceoff routines
continue to operate on individual rounds. `RoundHistory` names that history
format explicitly; the legacy `GameHistory` name remains available. Each
`RoundResult.history` can be passed to `play.game_history_to_game_data()` for
existing round statistics and targets. AI objectives and model formats have
not changed.

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
round. The old `sj.get_game_about_to_end()` name remains a compatibility alias
for that round condition. Existing winner helpers still report round winners;
use `GameResult.winners` for full-game winners.

## Model checkpoint faceoffs

Run an even number of games so each checkpoint plays both seats from the same
seeded starting conditions:

```sh
uv run skyjo-faceoff \
  models/run_a/checkpoint_a.pth \
  models/run_b/checkpoint_b.pth \
  --games 20 \
  --mcts-iterations 100 \
  --workers 4
```

The command reads the `EquivariantSkyNet` architecture settings stored in the
checkpoints. Use `uv run skyjo-faceoff --help` to see model-setting overrides,
the random seed, device, terminal-rollout, progress, and worker options.

## Offline training

Replay buffers are saved as versioned NumPy dataset directories containing only
retained positions and complete-game metadata. Legacy replay pickle files are
not supported.

Train to an exact cumulative optimizer-step target with a deterministic
game-level validation split:

```sh
uv run python run_train_epoch.py data/training_data/RUN/dataset \
  --steps 100 \
  --output-checkpoint models/offline/step_100.pth
```

Resume by passing the prior checkpoint and a larger cumulative target:

```sh
uv run python run_train_epoch.py data/training_data/RUN/dataset \
  --checkpoint models/offline/step_100.pth \
  --steps 200 \
  --output-checkpoint models/offline/step_200.pth
```
