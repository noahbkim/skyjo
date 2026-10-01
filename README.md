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

`faceoff.interval` controls promotion evaluation independently of
`budget.checkpoint_interval`, which controls periodic persistence only. With
faceoffs enabled, the champion changes only after an accepted evaluation; skipped
evaluation iterations remain provisional, even if saved. Rejection restores the
last accepted model and optimizer. Initial, generation, candidate, and final
snapshots may also be saved when needed for evidence, regardless of the periodic
save interval. A final candidate may still be unevaluated if the budget ends
between promotion intervals.

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

Trajectory progress counts total work performed. Checkpoint metadata and rollback
events additionally identify `active_state_progress`: rejected optimizer updates
remain in history but are absent from the restored model. A null checkpoint
reference means the measured in-memory model has no exact saved checkpoint at that
point. Loss means are over optimizer steps; game statistics are means over the
generated games. Compare their definitions, data, and budgets before interpreting
similarly named metrics as equivalent.

Artifact paths are relative to the run directory. Replay is latest-only: apply
`superseded` records before resolving historical registrations. Supersession is
recorded before replacement; on save failure it conservatively marks the old
snapshot unavailable even if its files survived. Checkpoints and replay are saved
independently and do **not** constitute a coherent run recovery point.

Replay provenance records the initial buffer's path and dataset ID (when supplied),
the previous replay dataset ID, and the latest batch's generating checkpoint and
game count. The generating checkpoint describes only that newly added batch,
not the entire mixed replay buffer. No per-game provenance index is maintained;
these references describe how the buffer was built, not precisely which sources
remain after eviction. The replay manifest therefore has no single
`source_checkpoint` for the whole snapshot.

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
