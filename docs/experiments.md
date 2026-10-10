# Experiments and findings

Keep the current workflow here, conclusions in topic findings, and detailed
evidence in run directories. CLI `--help` and example configurations are the
sources for available options and defaults.

## Choose the question and workflow

Run commands from the repository root. Start with
`uv run python distributed_main.py --config configs/smoke.toml --allow-dirty`.
For a recorded research run, commit the source and omit `--allow-dirty`.

| Question | Entry point | Starting input |
| --- | --- | --- |
| Does self-play training improve? | `distributed_main.py` | `configs/baseline.toml` |
| What changes after resuming a training state? | `distributed_main.py` | `configs/continuation.toml`, edited to identify parent artifacts |
| Can a model fit saved replay? | `run_train_epoch.py` | A saved replay dataset |
| Does an objective or architecture help on fixed data? | `run_offline_comparison.py` | `configs/round_score_doubling.toml` versus an `extends` overlay and one replay snapshot |
| Does an objective improve complete training runs? | `run_auxiliary_ablation.py` | `configs/round_objectives.toml` (`round_objectives_smoke.toml` for a pipeline check) |
| Which contestant plays better? | `run_checkpoint_comparison.py` | Frozen checkpoints and `configs/evaluation-{policy,search}.toml` |
| Can cumulative scores predict eventual winners? | `run_boundary_value_experiment.py` | `configs/boundary_value.toml` and completed round logs |
| Which boundary estimator predicts better? | `run_boundary_methods.py` | Held-out round logs, gameplay and score checkpoints |
| Is terminal sampling accurate? | `run_terminal_benchmark.py` | Replay containing final-action states |
| Does symmetric action grouping speed search? | `run_mcts_benchmark.py` | Fixed replay positions and frozen evaluators |

Use `uv run python <entrypoint> --help` for the required inputs. Copy a checked-in
configuration for a new recipe and keep resource paths portable; resolve paths
relative to the declaring config. Historical run paths belong in provenance, not
in reusable configurations or onboarding commands.

`extends` overlays change only the settings under test. The score-weight and
capacity examples inherit the same baseline. The continuation template requires
real parent checkpoint/replay paths and matching seed, players, dimensions, and
enabled heads; its iteration budget counts additional iterations. Put optional
run-specific notes in a copied config, rather than inheriting historical prose.

For fixed-data training and paired objective comparisons, replace the dataset
placeholder with a current-format replay directory:

```sh
uv run python run_train_epoch.py path/to/replay --steps 100 --output-checkpoint path/to/offline.pth
uv run python run_offline_comparison.py --dataset path/to/replay \
  --config configs/round_score_doubling.toml --config configs/round_score_weight_1.toml \
  --steps 2000 --seeds 0,1,2 --allow-dirty
```

The auxiliary comparison requires matching raw-score and doubling labels.
The single-dataset command prints metrics and optionally saves a checkpoint;
the comparison command creates a recorded run. For exact offline resume, pass
`--checkpoint` and the same learning/data settings; `--steps` is the cumulative
target including restored updates.

For example, compare one checkpoint's policy argmax with its search player:

```sh
uv run python run_checkpoint_comparison.py \
  --control path/to/checkpoint.pth --variant path/to/checkpoint.pth \
  --control-settings configs/evaluation-policy.toml \
  --iterations 32 --seed-count 32 --seed 0 --allow-dirty
```

This requests 32 seed pairs with both seats. Use contestant settings files for
different play/search behavior; leave the shared CLI budget explicit. It matches
weights while deliberately changing inference compute.

## Make comparisons interpretable

Before an expensive run, write the question, control, changed factor, budget,
primary measurement, and stopping rule. This can be a few sentences in the run
description. Distinguish generated games, optimizer updates, sampled positions,
search visits, and elapsed time; matching one does not match the others.

- **Playing strength:** freeze checkpoints, use both seats for each seed, report
  win credit including shared ties and control-minus-variant score margin.
  State policy/search mode, visits, noise, temperature, and boundary evaluator
  for both contestants. Positive margin favors the variant.
- **Learning quality:** split by complete game, pair initialization/minibatch
  seeds, and use the same held-out data. Select on validation and reserve the
  test set. Self-play loss and scores come from a changing distribution.
- **Uncertainty:** resample whole games or two-seat seed pairs as appropriate.
  Report the interval level and method. These intervals usually condition on
  fitted checkpoints; independent training seeds are a separate uncertainty.
- **Performance:** fix positions, weights, workload, hardware, threads, and
  warm-up. Alternate paired timings and separate diagnostics from timed work.
  Keep speed and playing-strength claims separate.

Mark exploratory selection and follow-up confirmation explicitly. Inconclusive
means the test did not resolve the difference, not equivalence. A failed or
interrupted run supplies only its completed measurements.

## Record once

Each `.runs/<run-id>/` contains identity/status and Git/runtime provenance in
`run.json`, original and resolved inputs, structured events in `trajectory.jsonl`,
and registered artifacts in `artifacts.jsonl`. Checkpoints, replay, round logs,
predictions, curves, and detailed reports stay there. Exact artifacts depend on
the recipe; inspect the registry rather than assuming every run has every file.

Use the structured result/report for comparisons instead of copying values from
terminal logs. Check completion status, actual budgets, dataset/checkpoint hashes,
and which revision ran. A dirty Git revision alone cannot reconstruct local edits.
Keep source snapshots where a recipe records them.

Generated reports should contain setup, principal results, material limitations,
and artifact references. Run-local narrative notes are optional; avoid empty
templates or repeated copies of this guide. Preserve original artifacts when
summarizing a run. If archiving large replay files, verify a lossless archive and
retain its manifest and checksum before removing the unpacked copy.

The [architecture performance check](findings/architecture-performance.md) records
fixed workloads for transitions, search, inference, and optimizer updates. Run
`uv run python run_layer_benchmark.py --help` to reproduce the workload.

## Preserve a finding

Promote a result when it changes a decision, rules out a plausible approach,
explains a material limitation, or prevents repeating an expensive mistake.
Routine launches and smoke tests stay in run records.

Update an existing question's note in `docs/findings/`; create a new topic only
for a distinct question. Aim for a few hundred words and one small table:

```text
# Question
Current conclusion and evidence date.
Evidence: control, changed factor, budget/sample size, result and uncertainty.
Interpretation: what this supports and the important limits.
Decision or next unresolved question.
Sources: linked compact evidence, run IDs, revision and original artifact names.
```

Commit a small adjacent JSON evidence summary with selected measurements,
comparison settings, source paths/run IDs, source hashes, and recorded revision.
Give units and denominators; mark unknown provenance explicitly. Keep enough
context to understand the result without local `.runs` access. Do not copy raw
games, whole reports, or every metric. Verify each promoted number against its
structured source; label projections and hypotheses as such.

When new evidence arrives, replace the stale conclusion with the current
synthesis and retain a short supersession reference if useful. Do not append a
chronological diary. Review documentation in the same change as interfaces;
review findings when an experiment warrants it. No separate database or
automatic conclusion generation is needed.

## Findings index

| Question | Current evidence |
| --- | --- |
| [Why are predictable scores underlearned?](findings/score-learning.md) | Capacity exists for a narrow arithmetic task; emphasis and representation remain relevant. |
| [Is training still improving?](findings/training-progress.md) | Gains and regression both occurred; broader replay helped one continuation, with later gains unresolved. |
| [Is more search worth its cost?](findings/search-budget.md) | Checkpoint-dependent inference effects; no established higher-search training advantage. |
| [Do better boundary values improve play?](findings/boundary-evaluation.md) | Better local estimates, no established full-game benefit at tested budgets. |
| [Does symmetry make search faster?](findings/symmetry.md) | Fewer nodes, essentially unchanged throughput in the saved replay benchmark. |

These notes consolidate the October 2–9, 2026 local handoff and completed
diagnostics. Their committed JSON summaries retain selected source fields and
hashes. Historical `.runs/...` paths identify originals; they are not required
files or runnable instructions for a fresh checkout. Some old diagnostics lack
an execution revision and some runs used dirty source; the summaries retain
that uncertainty rather than inventing provenance.
