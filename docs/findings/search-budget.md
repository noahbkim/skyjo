# Is more search worth its cost?

**Closed October 19, 2026.** This investigation did not establish a better
training recipe or resolve the clearing-card failures. The final candidate
improved substantially over its predecessor but did not clearly outperform the
time-matched historical baseline at equal inference search. Keep the historical
lineage as the reference; do not interpret this as proof that more search cannot
help, that the value network is calibrated, or that training needs 100× more games.

## What survives on the branch

| Change | Why keep it | Status |
| --- | --- | --- |
| Group-normalized policy and group root noise | Removes action-multiplicity bias consistently in training and inference; preserves tensor/checkpoint interfaces | Default for new models; strength benefit not isolated |
| Historical checkpoint semantics | Old checkpoints retain per-action normalization; exact resume checks the mode | Required compatibility |
| Randomized self-play budgets | Small optional mechanism for trading search quality against game/target throughput | Off by default; pilot inconclusive |
| Weights-only bootstrap | Reuse compatible weights with fresh replay, optimizer, RNGs and counters; record parent hash | Reusable training feature |
| First-round faceoffs | Paired opening-round scores without later-round contexts | Optional `--first-round` mode |
| Four frozen AlphaZero contestant recipes and tactical plan | Compare learned/uniform priors and visit/Q acting using existing runners | Evaluation options, not recommended defaults |

The general search-position diagnostics, exact-endgame benchmark, strategy
implementations and action-group tree merging already existed on the integration
base (`16a6ec7`); they are not new infrastructure from this branch. No residual
architecture, turn-completion search, posterior-value objective or alternative
backup was added to production by this investigation. See
[architecture](../architecture.md), [training options](../experiments.md), and
[benchmark commands](../search-benchmarks.md).

## Training and playing strength

The group-aware fixed 256 run bootstrapped from historical iteration 19 with fresh
replay/Adam: 4,096 games in 12h16m, checkpoint 4. It beat its start in the requested
256-versus-32 setup, but lost the opening-round comparison against time-matched
historical 44: 42.33% win credit over 1,024 rounds and 1.47 points worse per round.
That comparison mixes weights and inference budgets.

A continuation lowered value-loss weight from 1 to 0.1 while retaining auxiliaries,
search 256, replay and Adam. Another 8,192 games in 22h39m reached checkpoint 12.
Candidate cumulative training after iteration 19 was 34.9135h versus historical 87's
35.0025h. Historical code, worker counts and training history differ; this is a
wall-time comparison, not an isolated search-budget or loss-weight ablation.

Final faceoffs used eight workers, paired seats, no root noise, greedy visit
acting, 256 full games and 1,024 first rounds per row, with separate seeds for
full-game and first-round comparisons. Positive round score advantage means
opponent score minus candidate score; intervals are paired
seed-cluster bootstrap 95% intervals, unadjusted for multiple comparisons.

| Candidate 12 against | Opponent / candidate search | Full-game win credit | First-round score advantage |
| --- | --- | --- | --- |
| Historical 87 | 256 / 256 | 53.52% (47.46–59.38) | +0.79 (−0.18–1.82) |
| Checkpoint 4 | 256 / 256 | 83.20% (78.13–87.89) | +13.68 (12.33–15.01) |
| Historical 87 | 32 / 256 | 57.03% (50.78–63.09) | +2.09 (1.05–3.15) |

Other decisions supported by the experiments:

- **Keep visit acting.** Highest-Q acting with identical checkpoint 4/search 256
  lost to visits: 36.33% win credit in 256 full games (30.08–42.58), and 38.09%
  in 1,024 first rounds. A better result on a giveaway fixture did not transfer.
- **Do not promote the randomized schedule.** The short fixed 32 versus
  75% fast 16 / 25% full 128 pilot showed no clear strength advantage. Its expected
  cap was 44, and only 25.12% of decisions entered replay. It refreshed inherited
  replay more slowly and received fewer optimizer updates; it did not demonstrate
  a throughput advantage over fixed 32.
- **Value weight 0.1 was a useful lead, not a proven cause.** Fixed-replay tests
  improved policy KL/value MSE versus weight 1 (0.10867/0.18538 versus
  0.11427/0.19074). The subsequent stronger checkpoint also had much more training;
  there was no contemporaneous weight 1 continuation.

## What the tactical probes actually established

These are two selected giveaway positions and related clearing opportunities,
not independent samples of playing strength. Avoidance labels are tactical
hypotheses, not exact full-game action values. Uniform priors below mean uniform
at both root and interior decision groups, not merely group normalization.

Candidate 12, seeds 0–3, PUCT c=1, parent FPU reduction 0, mixture backup, sampled
ordinary chance, boundary sample 1, no noise, visit acting/targets:

| Position / priors | Clear first evaluated | Stable giveaway avoidance through 8,192 visits |
| --- | --- | --- |
| Give9 / learned | 415–507 | 1,551–1,710 |
| Give9 / uniform | 36–124 | 162–207 |
| Give12 / learned | 884–1,006 | 1,671–1,869 |
| Give12 / uniform | 67–172 | 38–178 |

At 256, learned priors fail avoidance in all four seeds on both positions; uniform
avoids in all four. Avoidance can precede discovering the clear, especially on
Give12, so it must not be treated as tactical understanding. The original 119
AlphaZero-preset screen also favored uniform priors, but used different backup/FPU
settings; its larger thresholds are not a matched training-progress comparison.

- **Chance enumeration was not the main fix.** Enumerating every ordinary
  afterstate outcome barely changed learned-prior avoidance thresholds, while
  requiring roughly 5–6× as many evaluated states at recovery. It evaluates
  outcomes of a selected action, not every subsequent decision; a batch yields
  one expectation, not an independent visit per outcome.
- **Allocation and cumulative targets both matter.** Interior-only uniform
  priors discovered the clear earlier but still failed avoidance at 256. Historical
  visits remain in both ancestor value averages and final visit targets. Give9's
  learned 256 target increased giveaway mass from raw 68.4% to 78.9%; Give12's target
  already assigned 69.3% across alternatives despite selecting the giveaway.
- **The evaluator recognizes completed clears better than opportunities.**
  Candidate 12's clearing-player value rose 0.4392→0.6261 for clear9 and
  0.4885→0.6533 for clear12. Completing a whole TAKE/placement turn with the same
  evaluator favored TAKE where one-action lookahead favored DRAW. These are
  neural estimates, not calibrated outcome oracles. Actual fresh 256 searches at
  acquisition still chose DRAW; fresh searches after TAKE chose CLEAR.
- **No demonstrated backup arithmetic defect.** Under sampled chance and one
  boundary sample per visit, recursive mixture values matched flat means of fresh
  leaf evaluations, including a root pseudocount (maximum error 1.99e-7 across 22
  positions). This equivalence does not cover all enumeration/boundary settings.
  In Give9 seed 0, the next 20 placement visits after discovery all chose CLEAR,
  and all 320 subsequent acquisition visits chose TAKE. Local PUCT did respond;
  late discovery and retained earlier samples explain more than the first snapshot.
- **More search can make better targets.** On 17 exact endgames, candidate 12's
  policy regret was 0.08548, search 32 target regret 0.03264 and search 256 0.01707.
  Uniform 256 was worse at 0.03788. Thus uniform is not a general remedy, and the
  claim that deeper search never improves targets is unsupported.
- **Small-set fitting works; generalization remains open.** Policy-only fits on
  14/15 improved exact-case targets reached KL 0.00020/0.000058. These were
  same-example capacity checks. Combined-loss replay fits could overfit value
  outcomes and absorb policy targets less effectively; neither result diagnoses
  the full training loop by itself.

## Handoff and recovery

The full pre-cleanup investigation is preserved remotely at
[`archive/search-clearing-cards-20261019`](https://github.com/noahbkim/skyjo/tree/archive/search-clearing-cards-20261019),
commit `753a284d3f6db965c916b057b127afc200bed26b`. Its
[original findings](https://github.com/noahbkim/skyjo/blob/753a284d3f6db965c916b057b127afc200bed26b/docs/findings/search-budget.md)
link every detailed evidence file, method, checkpoint hash and experiment result.
The tag preserves them even if this branch is later squashed.

Removed from the active tree: six one-off runners (randomized pilot, learning-loop,
followups, target, value-propagation and chance-discovery probes), their four
support modules and two dedicated test files, two dated overnight recipes, and
14 bulky evidence JSON files. They encoded particular checkpoint names, fixed
case selections or intervention recipes; they overlapped existing diagnostics,
training and faceoff entrypoints. The pilot-specific checkpoint-adaptation test
was removed with its helper. Core feature tests remain.

Use existing tools for future work:

- `run_search_diagnostics.py`: fixture-based discovery, values, visits and budget
  traces; extend the TOML plan instead of adding another runner.
- `run_search_strategy_benchmark.py`: exact-endgame regret and frozen controls.
- `run_checkpoint_comparison.py`: paired full games or `--first-round`; reuse the
  same checkpoint with different contestant settings to isolate search changes.
- `run_train_epoch.py`: fixed-replay training and fitting diagnostics.

Local runs and checkpoints were not deleted. Final candidate:
`.runs/overnight-group256-value01-from4/20261018T002135Z-overnight-group256-value01-6fb23c2b2c/checkpoints/checkpoint_000012.pth`
(SHA256 `21159b7fec97cd3a46d76234429dfa3e6f5b493bc97e02d4d7275c046c6f1aac`).
`.runs/value01-faceoffs-20261018/plan.json` locates checkpoint 4/historical 87 and records
hashes and time matching; results are alongside it. Archived evidence preserves
that plan too. These local artifacts are not part of Git: retain or relocate them
before removing the worktree. Old ignored `configs/local/` overnight overrides
refer to archived recipes; recover those recipes from the tag before reusing them.

No new run is queued. If investigation resumes, the unresolved question is whether
useful supervision is absent, is not absorbed, or fails to generalize. Audit a
real training iteration with fixed positions/targets and a held-out comparison
before buying substantially more self-play. Residual column-to-head connections,
turn-completion search and sharper acting temperatures remain untested ideas,
not changes or recommendations established by this branch.

## Earlier inference and training evidence

**Conclusion, through October 9, 2026:** inference search gains depend on the
checkpoint. Higher-search self-play has not established better weights per hour.

At one frozen checkpoint, MCTS128 beat MCTS32 with 75% win credit and a
+26.50-point margin over 32 games. This small screen does not measure training
efficiency.

Matched-start continuations trained with 32 or 128 visits. Below: 64 games per
row, both seats, temperature zero, no root noise, matched evaluation visits.
The variant trained with 128 visits.

| Training checkpoint pair | Evaluation visits | Variant win credit | Margin, points |
| --- | ---: | ---: | ---: |
| Equal fresh games, iteration 41 | 32 | 37.50% | −21.06 |
| Equal fresh games, iteration 41 | 128 | 73.44% | +18.64 |
| Equal fresh games, iteration 42 | 32 | 39.06% | −17.34 |
| Equal fresh games, iteration 42 | 128 | 39.06% | −21.81 |

Iteration 41 reverses with inference budget; iteration 42 loses at both.
Better generated-game scores did not imply a robust training advantage. These
small repeated screens cover one paired trajectory.

At checkpoint 119, MCTS32 versus policy argmax scored 53.91% in 64 games (paired
95% interval 42.19%–65.63%). MCTS128 scored 58.98% in 128 games (50.00%–67.97%);
its score-margin interval was −1.22 to +15.10 points. Different seeds and unequal
compute preclude a direct 32-versus-128 comparison; neither establishes a clear
strength gain.

[Evidence](evidence/search-budget.json): settings, checkpoints, run IDs,
provenance and paired-bootstrap summaries for training and inference comparisons.
