# Experiments and findings

Use [example configurations](../configs/) and CLI `--help` for execution details.
Keep this guide for methodology, topic notes for conclusions, and run directories
for detailed artifacts.

## Shared and local configurations

Commit portable recipes and controlled comparisons in `configs/`. Keep personal
artifact paths, machine tuning, and one-off runs in ignored `configs/local/`.
For a self-play run, create `configs/local/baseline.toml`:

```toml
extends = "../baseline.toml"

[execution]
workers = 8
```

Pass that file to `distributed_main.py --config`. Inherited files and artifact
paths resolve relative to the file declaring them. Run records preserve the
settings and source configs, including local overrides. Promote reusable changes
back into the shared recipe.

New runs use a group-normalized policy and group-level root exploration noise.
To reproduce the old policy parameterization and noise distribution, set:

```toml
[model]
policy_normalization = "actions" # default: "groups"

[search]
group_root_noise = false # default: true
```

Both settings are recorded in resolved run configurations. The model option
controls training and inference together; it is not a uniform-prior override.
Search still learns and uses nonuniform group probabilities. Targets retain the
existing per-action representation, dividing each group's mass among its members.
The offline CLI accepts `--policy-normalization groups` or `actions`; omitted,
it uses groups for new training and preserves the checkpoint mode on exact resume.
Old checkpoints without the field load as actions. An online continuation adopts
the new run's settings, so specify actions explicitly to retain the old mode.
See [policy normalization](architecture.md#boundaries-that-matter) for the math.

## Make comparisons interpretable

Before an expensive run, state the question, control, changed factor, budget,
primary measurement, and stopping rule. Generated games, optimizer updates,
sampled positions, search visits, and elapsed time are different budgets.

- **Playing strength:** freeze checkpoints and pair both seats for each seed.
  Report win credit (including shared ties) and score margin; positive
  control-minus-variant margin favors the variant. State each contestant's
  inference settings, since equal weights do not imply equal compute.
- **Learning quality:** split by complete game, pair initialization and sampling
  seeds, select on validation, and reserve the test set. Self-play losses and
  scores come from a changing distribution and cannot establish strength alone.
- **Uncertainty:** resample whole games or two-seat seed pairs, not correlated
  positions. Intervals conditional on fitted checkpoints do not capture variation
  across training seeds. Inconclusive results do not establish equivalence.
- **Performance:** fix positions, weights, workload, hardware, threads, and warmup;
  alternate paired timings and keep diagnostics outside timed work. Speed and
  playing strength require separate evidence.

Distinguish exploratory selection from confirmation. A failed or interrupted run
supports claims only about its completed measurements.

## Continuation and recording

For a weights-only restart, set `bootstrap_checkpoint` instead of
`initial_checkpoint`. This copies compatible model weights, permits changing
policy normalization, and starts with fresh replay, Adam state, RNGs and counters.
It rejects an initial replay dataset or simultaneous continuation checkpoint.
The initial saved checkpoint records the source hash and source progress.
Set the search budget, games per iteration, workers and time cap in a local
configuration extending `configs/baseline.toml`. Time caps are soft: finishing
an active iteration can produce substantial overrun, recorded in the results.

Optional randomized self-play caps use `selfplay.fast_search_iterations` and
`selfplay.full_search_probability` together. `search.iterations` remains the full
cap. A per-game independent RNG selects each decision's cap. Fast searches disable
root noise, act greedily, and contribute no training rows; full searches keep the
configured exploration and targets. All played decisions remain in game histories,
and retained rows receive the actual final game outcome and round auxiliaries.
Replay ratio counts retained rows, not all played moves. Metrics distinguish
`generation/positions`, `generation/training_positions`, and `generation/fast_positions`.
Defaults disable randomization. Both new settings must be supplied to enable it.

For example, this optional schedule retains about one quarter of decisions:

```toml
[selfplay]
fast_search_iterations = 16
full_search_probability = 0.25

[search]
iterations = 128
```

Its expected cap is 44 traversals per decision, not 32. Size replay and optimizer
exposure using retained positions; a larger game count alone need not provide
more training signal. This capability is experimental: the initial pilot did not
establish a strength benefit. See the [closed investigation](findings/search-budget.md).

Offline exact resume requires matching learning/data settings; `--steps` is the
cumulative update target. Online continuation creates a new run from a parent
checkpoint and replay, permits compatible setting changes, and counts additional
iterations. The [continuation template](../configs/continuation.toml) requires
explicit parent artifacts.

Use structured results rather than terminal logs. Check status, actual budgets,
artifact hashes, and execution revision before interpreting a comparison. A dirty
Git revision alone cannot reconstruct the executed source.

Keep checkpoints, replay, predictions, curves, and detailed reports in the run
directory. Reports need only setup, principal results, material limits, and
artifact references; narrative notes are optional. Preserve original artifacts
when promoting a finding.

## Preserve a finding

Promote results that change a decision, rule out an approach, explain a material
limitation, or prevent repeating an expensive mistake. Routine launches and smoke
tests stay in run records.

Update the existing topic note with:

1. The question and current conclusion.
2. Evidence: control, changed factor, budget/sample size, key result, uncertainty.
3. Interpretation and material limits.
4. The decision or next unresolved question.
5. A link to compact JSON evidence with run IDs, revision, and source artifacts.

Aim for a few hundred words and one small table. Verify selected measurements
against structured reports; retain units, denominators, and provenance in the
JSON so the conclusion is understandable without local artifacts. Label
projections, hypotheses, and missing provenance explicitly.

Replace stale conclusions with the current synthesis, retaining a short
supersession reference where useful. Avoid chronological diaries, copied reports,
and repeated metrics. Update design rationale when an interface decision changes.

## Findings

- [Score learning](findings/score-learning.md)
- [Training progress and replay](findings/training-progress.md)
- [Search budget](findings/search-budget.md)
- [MCTS mixture backups](findings/mcts-mixture-backups.md)
- [Round-boundary evaluation](findings/boundary-evaluation.md)
- [Symmetric action grouping](findings/symmetry.md)
- [Architecture performance](findings/architecture-performance.md)
- [Search throughput and performance PoCs](findings/search-performance.md)

These notes include negative and inconclusive results. Their evidence summaries
preserve historical source references and provenance gaps; local run paths are
identifiers, not required files for a fresh checkout.
