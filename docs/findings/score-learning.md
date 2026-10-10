# Why are predictable scores underlearned?

**Conclusion, evidence through October 9, 2026:** the model can represent a narrow
score-arithmetic task, but ordinary training leaves substantial error on rare,
already determined scores. More embedding width or more fixed-data updates did
not resolve it in the tested comparisons.

At original checkpoint 41, 5,842 fully revealed player examples were just 0.557%
of replay player examples. Their final raw-score labels equaled visible sums to
numerical precision, yet the trained head had 8.83-point MAE. A linear probe
recovered these sums from early card embeddings with 0.00059-point test MAE,
versus 7.34 from final global features. This diagnoses accessibility of arithmetic
information; it does not prove irreversible information loss.

An end-to-end arithmetic-only fit of the same architecture reached 0.45-point
test MAE from trained weights and 0.98 from initial weights: 1,200 Adam updates,
batch 128, no competing objectives or weight decay. Those bundled changes do
not establish a remedy for ordinary gameplay training.

Three paired seeds on fixed replay, with a game-disjoint validation split:

| Updates | Raw-score weight | Known-board MAE, points | Overall raw-score MAE, points |
| ---: | ---: | ---: | ---: |
| 2,000 | 0.1 | 8.127 | 11.551 |
| 2,000 | 1.0 | 6.254 | 11.153 |
| 10,000 | 0.1 | 8.460 | 11.487 |
| 10,000 | 1.0 | 6.316 | 11.315 |

Higher weight helped this subset; five times as many updates did not. Doubling
embedding widths at 2,000 updates produced negligible score gains over the two
available paired seeds. Later long-run training improved known-board MAE by
2.82 points versus the original model on a fixed diagnostic dataset, but the
final-minus-iteration-10 difference was +0.115 points (game-bootstrap 95%
interval −0.014 to +0.250). Score learning improved without solving the subset.

**Next question:** can an isolated change to emphasis or representation improve
known-score fit under the full objective, and does that improve play? Avoid
treating a small normalized loss, a frozen probe, or this capacity fit as a
complete causal explanation.

[Selected evidence and provenance](evidence/score-learning.json) include the
original `20261002T205756Z-…-520eb7ede1` run, the 2k/10k weight studies,
representation/capacity reports, and long-run paired diagnostic. Execution
revisions missing from old diagnostic reports are marked unknown.
