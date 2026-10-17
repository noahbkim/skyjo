# Is more search worth its cost?

**Current direction, October 16, 2026:** isolate search strategy changes using
frozen weights and valuation. On two selected clearing-card positions, uniform
priors substantially reduce the iterations needed to stop handing the opponent
a clearing card. Highest-Q final selection also helps the learned-prior
AlphaZero preset. Playing-strength gains from those frozen-search changes remain
untested. A separate group-aware/randomized-budget training pilot below found no
clear strength improvement.

The frozen iteration 119 variant checkpoint is shared across all methods.
Here “AlphaZero” means the named preset: PUCT, mean-return backups, zero FPU,
sampled ordinary chance, one first-visit boundary sample, grouped symmetry,
no root noise, and greedy final selection. It differs from the recording's
historical mixture-backup search. Uniform means equal probability per symmetry
group at every decision node. Final Q selection changes no allocation or backup.

Ranges below cover seeds 0–3. “Stable from” is one plus the last giveaway
iteration: the selected action then avoids the labeled card through the tested
horizon. It is not an asymptotic convergence claim.

| Priors / final selection | 9-card position: stable from | 12-card position: stable from | Tested horizon |
| --- | ---: | ---: | ---: |
| Learned / visits | 4,514–5,171 | 11,305–13,196 | 32,768 |
| Learned / highest Q | 1,275–1,467 | 6,211–6,687 | 32,768 |
| Uniform / visits | 873–1,260 | 1–188 | 8,192 |
| Uniform / highest Q | 1,083–1,360 | 39–181 | 8,192 |

The initial 8,192-iteration screen had 4/4 final avoidances in every cell except
learned-prior visit selection on the 12-card position (0/4). That failure prompted
an exploratory extension of learned-prior methods to 32,768. The longer runs
retained identical prefixes and recovered in all four seeds. Paired visit/Q runs
had exactly identical root statistics, evaluator work, and witness discovery;
the improvement therefore comes from final selection on the same search.

Uniform priors discover the GIVE → TAKE → CLEAR line in 107–213 iterations for
the 9-card position and 126–503 for the 12-card position, versus 620–644 and
3,735–4,280 with learned priors. The model assigns the opponent's clearing action
only 3.33% and 1.65% prior probability, respectively. Discovery alone does not
immediately change the final choice. Conversely, avoiding the card at iteration
one does not establish that search understood the tactic. Q selection does not
improve median recovery over visits in the uniform-prior runs.

**Next action:** run paired-seat frozen-checkpoint faceoffs using the matching
[contestant recipes](../../configs/search-strategies/), first changing only
priors and then only final selection. Keep mean-return valuation and boundary
evaluation fixed; report win credit, score margin, paired uncertainty, and work.
No faceoffs or training were run in that tactical screen. Two selected positions and four
seeds cannot establish general playing strength or model accuracy.

**Implemented next step:** new training defaults to group-normalized learned
policies and group-level Dirichlet noise. Symmetric logits are averaged before
normalization, and group mass is split across the existing action interface.
The same normalization is used in the training loss, inference, search priors,
and policy-only selection; visit-target splitting already preserves group mass.
`model.policy_normalization = "actions"` and `search.group_root_noise = false`
retain the previous behavior. Old checkpoints load with their original policy
semantics, so the measurements above remain a legacy-checkpoint baseline. This
implementation is used in the continuation pilot below; that small pilot does
not establish a playing-strength gain from normalization.
See [configuration and continuation rules](../experiments.md#shared-and-local-configurations).

[Reproduction and faceoff commands](../search-benchmarks.md#clearing-card-alphazero-baseline).
[Compact evidence](evidence/clearing-alphazero.json) retains all 48 measured trials,
per-seed thresholds, hashes, plans, model probes, and local run references.
Detailed traces and source snapshots remain under `.runs/clearing-*` in the
`codex/search-clearing-cards` workspace. Recipes are committed in `d9f4010`.
Validation: 45 focused tests passed, and the four faceoff configurations were
checked against the measured diagnostic configurations.

## Randomized-budget continuation pilot

**Result, October 16:** both continuations and all six faceoffs completed in about
1 h 55 m. Neither group-aware fixed32 nor randomized fast16/full 128 demonstrated an
improvement over its starting model. The randomized method did not demonstrate
an advantage over fixed32. This is one training seed and a small evaluation screen.

Each row has 64 games (32 environment seeds, both seats), common search 32,
mixture backups, no root noise and greedy acting. Win credit includes half a win
for ties. Intervals bootstrap environment-seed pairs 20,000 times; they are
conditional on these checkpoints and unadjusted for multiple comparisons.
Positive score margin favors the first-named model.

| Comparison | Win credit (95% interval) | Score margin (95% interval) |
| --- | ---: | ---: |
| Group fixed32 vs historical | 56.25% (43.75–68.75) | +6.88 (−3.69–17.33) |
| Group randomized vs historical | 50.00% (39.06–60.94) | −1.19 (−9.48–7.12) |
| Group randomized vs group fixed32 | 45.31% (34.38–57.81) | −8.38 (−18.11–1.19) |
| Historical vs original 119 | 49.22% (37.50–60.94) | +0.34 (−11.66–13.08) |
| Group fixed32 vs group-normalized 119 | 43.75% (29.69–57.81) | −7.59 (−18.80–3.27) |
| Group randomized vs group-normalized 119 | 42.97% (32.03–54.69) | −5.48 (−14.77–4.53) |

Both new arms retained learned priors, starting from 119 weights, Adam moments,
the same 2,097,152-position replay capacity and explicitly initialized replay
sampler. New training used group normalization and group-level root noise,
LR 0.001, replay ratio 4, inherited auxiliaries and 256 games per iteration.
The random arm selected full 128 searches independently with probability 0.25;
other moves used greedy, noise-free search 16. Only full-search rows entered replay.

| Arm | Actual training time | Fresh games | Retained positions | Optimizer updates |
| --- | ---: | ---: | ---: | ---: |
| Historical | 62.5 min | 2,048 | 534,918 | 8,359 |
| Group fixed32 | 54.6 min | 2,560 | 693,606 | 10,842 |
| Group randomized | 52.7 min | 1,792 | 138,549 | 2,169 |

Randomized play generated 551,560 decisions and retained 25.12%. It produced
fewer games than fixed32: its expected cap was 44 rather than 32, and games averaged
307.8 rather than 270.9 decisions. This pilot therefore does not show the intended
game-throughput advantage relative to this cheaper fixed baseline. Only 6.6% of
its final replay was fresh versus 33.1% for fixed32, so most training still sampled
inherited targets. The short trial confounds deeper target quality with slower
replay refresh and fewer updates as components of the randomized method.

The historical checkpoint 121 was selected nearest the new arms' mean elapsed
time before faceoffs. It used older code, a different replay sampler and 1024-game
updates, and had more elapsed training time. Its contrast cannot isolate grouping.
Initial comparisons preserve each arm's normalization, separating training from
the immediate policy transformation of the same 119 weights.

**Decision:** do not select this randomized schedule for a large run based on
these results. A follow-up should account for retained-sample rate when sizing
replay and optimizer exposure, and use a fresh contemporaneous control if the
goal is to isolate normalization. More evaluation games can narrow match
uncertainty but cannot replace independent training seeds.

[Compact evidence](evidence/randomized-search-budget.json) includes report hashes,
checkpoint hashes, resolved settings, paired-seed statistics and bootstrap details.
Full artifacts are in `.runs/randomized-budget-20261016`; code revision `5d79800`.

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
