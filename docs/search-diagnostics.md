# Search position diagnostics

Use this suite to study **discovery, valuation, and move selection separately**.
A search that sees a clear but still gives it away has a different problem from
one that never explores the reply. These are selected tactical cases, not a
playing-strength benchmark or a training objective.

The initial six cases come from `.interactive/dont_give_clear_game.json`:

| Case family | Recording frames (zero-based) | Expectation |
| --- | --- | --- |
| Give away 9 / 12 | 53 / 112 | Avoid replacing the visible clearing card |
| Opponent takes 9 / 12 | 54 / 113 | Prefer TAKE |
| Opponent clears 9 / 12 | 55 / 114 | Prefer replacement into the final hidden slot |

The recording numbers these rounds **1 and 2**. The giveaway and TAKE are
deterministic. The clear is guaranteed for every hidden-card outcome, but the
revealed/discarded hidden card is random. The suite keeps the public unseen pool;
it never inserts the hidden realization from a later recorded frame.

All six expectations are **tactical hypotheses**. Avoiding the recorded giveaway
can still choose a worse move. Taking/clearing is also not an exact full-game
win-probability oracle. The per-case rationale should say what a label establishes.
Do not report these related six positions as six independent games.

## Run and compare

```sh
uv run python run_search_diagnostics.py validate
uv run python run_search_diagnostics.py run \
  --checkpoint /path/to/checkpoint_000119.pth \
  --output .runs/diagnostics/search-positions-119
```

The default [plan](../configs/search-diagnostics.toml) compares sampled and exact
ordinary chance at 32/128/512/2048 iterations with four paired seeds. Weights are
loaded once, CPU inference uses one thread, and every case/method/budget/seed gets
a fresh root and RNG. Supply `--plan` for another grid, `--suite` for another set
of JSON cases, or `--allow-dirty` while developing. Existing output directories
are never overwritten. The checkpoint must match every case's player count.

Method tables accept single-tree `SearchConfig` settings, plus an optional
`boundary_checkpoint` relative to the plan. Unspecified fields use the current
search defaults, recorded in full in each trial. Per-iteration tracing requires
`replicas = 1`; replica settings are rejected explicitly. Use the
[search strategy benchmark](search-benchmarks.md) for independent-replica and
exact-endgame comparisons. The standard plan disables
symmetry merging to hold the partition fixed across budgets. If enabled, the
engine starts with grouping when the root is safe and retries without grouping
if an explored path reaches recycling risk. Inspect
`effective_merge_symmetric_actions` before interpreting budget curves as prefixes.
On a retry, traces and discovery/recovery iterations describe the final tree;
evaluator work and elapsed time include the abandoned attempt.
The same integer seed does not imply identical random outcomes across methods
that consume randomness differently.

Outputs:

- `report.md` and `summary.json`: per-case, per-method, per-budget acceptance and
  witness discovery counts, plus evaluation work. There is no blended strength score.
- `policy.json`: policy-only action, fixed-seat neural values, and each annotated
  line's action priors, pre-action values, and exact probability-weighted neural
  value after its final action. These probes are outside search work counters.
- `trials.jsonl`: one row per paired seed, including every root group's prior,
  visits and acting-seat Q; selected action; fixed-seat root value; and compact
  snapshots at powers of two and the final iteration.
- `positions/`, `plan.toml`, `manifest.json`, `sources/`: frozen inputs, checkpoint
  hashes, source snapshots, Git revision/dirty state, resolved per-trial settings,
  and run status. Failed runs retain completed raw rows and an explicit status.

Interpret the trial's witness fields carefully:

| Metric | Meaning |
| --- | --- |
| `first_evaluated_iteration` | First post-iteration observation with an evaluated final witness edge; allocating an unvisited edge does not count |
| `first_visited_iteration` | First traversal of that final edge; exact chance can evaluate nodes before traversing them |
| `first_accepted_iteration` | First selected root action meeting the annotation |
| `stable_accepted_from` | Start of the final uninterrupted accepted suffix **within this run's budget**; null if the final choice fails or is unscored |
| `witnesses[].edges` | Current visits and fixed-seat values along the annotated line |
| `evaluator_states` / `boundary_samples` | Actual neural states / completed-round evaluations requested, including next-deal neural work |

Discovery and selection are checked after every iteration, not just at the
stored snapshots. Null first-discovery means not observed within that budget;
a witness with `symmetry_ambiguous` or `unavailable` status is excluded from the
availability denominator. A grouped prefix can change slot coordinates, so the
runner refuses to silently follow the wrong continuation. Witness discovery
means the search evaluated that action, not that it valued it correctly.

Search values are estimates, not ground truth. Compare the neural drop along a
line, the searched value on that line, and the final root visit allocation.
`seconds` includes tracing overhead, excludes model loading and standalone neural
probes, and is descriptive only. Compare inference counts as well as iterations;
exact ordinary chance can do much more work per iteration. Learned boundary
model calls are reported separately from ordinary model states. This tool does
not attempt a throughput benchmark or run full-game matches.

## Add a position

A JSON file in the suite is a case; no Python registry or checkpoint is required.
The `add` command validates and exports a recording frame with provenance:

```sh
uv run python run_search_diagnostics.py add \
  --recording .interactive/dont_give_clear_game.json --frame 53 \
  --output data/search-positions/my-case.json --id my-case \
  --description 'Replacing 9 gives the next player a clear.' \
  --kind avoid --action 18 \
  --rationale 'Tactical hypothesis; avoiding this move is not proof of optimality.' \
  --tag giveaway-clear --witness 18 --witness 3 --witness 27
```

Action indices are the engine's 28-action mask indices, as recorded in the game.
`--action` is repeatable for equivalent acceptable/avoided moves. `--kind prefer`
requires choosing one listed action; `avoid` excludes listed actions; `unscored`
records observations without grading and takes no actions. Include all relevant
equivalent moves when annotating a preference. The default is unscored.

A witness is an ordered tactical line. Every prefix action must be deterministic;
its final action may have ordinary single-card chance outcomes. Witnesses cannot
cross a round boundary with multiple random reveals. A case itself may be a
last-turn or other boundary decision and may have no witness. Edit the JSON to
add multiple named witnesses. Every load validates engine invariants, label
legality, and line legality. Source frame numbers are zero-based. Export refuses
to overwrite an existing case.

For synthetic or replay-derived cases, construct a `Position`, serialize it with
`to_dict()`, and validate through `from_dict()` before saving. Use stable IDs,
keep public unknowns unknown, and document source/selection criteria. Changing an
existing fixture changes the experiment; output copies and hashes preserve that.

## Add a search approach

Changing parameters needs only another `[[methods]]` table. For a new
MCTS-compatible implementation, set `adapter = "your.module:run_search"` in its
method table. The callable accepts the same arguments as `mcts.run_mcts`, returns
the existing decision-node contract, and calls `observer(root)` after every
completed iteration and backup. A symmetry retry supplies a new root at visit
one, which resets the diagnostic history. Observers must only read the live tree;
they must not mutate it or consume search randomness. Custom code is trusted local Python,
not a sandboxed plugin. Its module source is saved along with Skyjo's sources;
external dependencies must be preserved by the experiment author.

Final choice uses the configured acting policy at temperature zero, matching
`search.runner` for single-tree visit, value, and Gumbel selection. Learning
target settings do not change the reported action. The observer reads the current
allocation context through `root_session.finish(view)`; custom allocation
sessions must make this accessor read-only and safe to call during tracing.
For algorithms with a different tree representation, add an adapter for the
root/witness snapshot contract rather than encoding algorithm-specific fields in
position fixtures. Position curation remains independent of the search algorithm.

## Next case families

Expand coverage before tuning heavily against these two games:

1. **Exact final-turn decisions:** finite hidden-card pools with independently
   enumerated win values; check regret and catastrophic errors, not just tactical
   acceptance. Preserve the exact oracle and objective with the case.
2. **Tempting but bad clears:** finishing/doubling or cumulative-score situations
   where removing a column is not the right objective.
3. **Rare chance outcomes:** large value swings with skewed card counts; contrast
   discovery failure with incorrectly weighted or noisy estimates.
4. **Choice versus reply coverage:** low-prior defensive moves, opponent threats,
   and distractor replies, including re-rooted decisions as here.
5. **Symmetry and multiplayer controls:** equivalent placements, low deck slack,
   and fixed-seat values after rotation, with meaningful annotations per case.

Keep exploratory hypotheses separate from proven oracle cases and keep a held-out
set. A candidate that improves tactical recovery must still pass broader search
regressions and matched full-game evaluation before a playing-strength claim.
