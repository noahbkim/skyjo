# Why are predictable scores underlearned?

**Conclusion, through October 9, 2026:** the model can learn score arithmetic,
but ordinary training underlearns rare, already determined scores. More width
or fixed-data updates did not resolve this.

At original checkpoint 41, 5,842 fully revealed examples were 0.557% of replay
player examples. Labels matched visible sums, yet head MAE was 8.83 points.
A linear probe achieved 0.00059-point test MAE from early card embeddings versus
7.34 from final global features: arithmetic became less accessible, without
proving irreversible information loss.

An arithmetic-only fit reached 0.45-point test MAE from trained weights and 0.98
from initial weights: 1,200 Adam updates, batch 128, no competing objectives or
weight decay. These bundled changes establish capacity, not a training remedy.

Three paired seeds on fixed replay, with a game-disjoint validation split:

| Updates | Raw-score weight | Known-board MAE, points | Overall raw-score MAE, points |
| ---: | ---: | ---: | ---: |
| 2,000 | 0.1 | 8.127 | 11.551 |
| 2,000 | 1.0 | 6.254 | 11.153 |
| 10,000 | 0.1 | 8.460 | 11.487 |
| 10,000 | 1.0 | 6.316 | 11.315 |

Higher weight helped; five times as many updates did not. Doubling embedding
widths at 2,000 updates gave negligible gains across two paired seeds. Long-run
training improved known-board MAE by 2.82 points versus the original model on
fixed diagnostics, but final-minus-iteration-10 was +0.115 (game-bootstrap 95%
interval −0.014 to +0.250). The subset remains unresolved.

**Next question:** can an isolated emphasis or representation change improve
known-score fit under the full objective—and playing strength? Normalized loss,
probes, and capacity fits alone cannot explain the failure.

[Evidence](evidence/score-learning.json): original run, weight/width studies,
probes, capacity fit and long-run diagnostic. Missing historical execution
revisions are marked unknown.
