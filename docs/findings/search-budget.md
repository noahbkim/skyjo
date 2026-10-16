# Is more search worth its cost?

**Current direction, October 16, 2026:** isolate search strategy changes using
frozen weights and valuation. On two selected clearing-card positions, uniform
priors substantially reduce the iterations needed to stop handing the opponent
a clearing card. Highest-Q final selection also helps the learned-prior
AlphaZero preset. Playing-strength gains remain untested.

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
No faceoffs or training were run here. Two selected tactical positions and four
seeds cannot establish general playing strength or model accuracy.

[Reproduction and faceoff commands](../search-benchmarks.md#clearing-card-alphazero-baseline).
[Compact evidence](evidence/clearing-alphazero.json) retains all 48 measured trials,
per-seed thresholds, hashes, plans, model probes, and local run references.
Detailed traces and source snapshots remain under `.runs/clearing-*` in the
`codex/search-clearing-cards` workspace. Recipes are committed in `d9f4010`.
Validation: 45 focused tests passed, and the four faceoff configurations were
checked against the measured diagnostic configurations.

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

**Deferred training question:** compare matched-start training under explicit budgets,
tracking fresh games and optimizer work. Evaluate prespecified checkpoints with
matched inference; one policy/search screen cannot explain a training plateau.

[Evidence](evidence/search-budget.json): settings, checkpoints, run IDs,
provenance and paired-bootstrap summaries for training and inference comparisons.
