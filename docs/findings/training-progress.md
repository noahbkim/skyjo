# Is self-play training still improving?

**Conclusion, evidence through October 9, 2026:** checkpoint matches showed both
substantial learning and later regression. Broader replay helped one controlled
continuation; later capacity, replay-ratio, and learning-rate changes did not
establish continued gains. Flat online loss or self-play score cannot settle
this question.

The comparisons below use MCTS32, no root noise, temperature zero, and both seats.
Ties receive half credit. Each row has 64 full games; positive score margin
favors the named variant.

| Variant versus control | Variant win credit | Control-minus-variant score, points |
| --- | ---: | ---: |
| Long-run 39 versus 19 | 82.81% | +49.16 |
| Long-run 39 versus 30 | 67.19% | +21.86 |
| Small-FIFO overnight 63 versus starting 50 | 23.44% | −39.33 |
| Small-FIFO 60 versus 50 | 35.16% | −13.08 |
| Four-times-capacity FIFO 60 versus 50 | 76.56% | +24.30 |
| Enlarged-replay 109 versus 66 | 63.28% | +13.63 |
| Enlarged-replay 109 versus 82 | 48.44% | −4.47 |

The iteration-63 regression is substantial: paired-seed 95% win-credit interval
14.06%–34.38%. For the two iteration-60 continuations, each had 10,240 fresh games;
the larger buffer's common-opponent advantage was 41.41 percentage points
(95% interval 25.00–57.03). Capacity changed from 524,288 to 2,097,152 positions;
replay ratio stayed four. Worker count changed, so elapsed time is not a fair
matching variable. This is one paired trajectory, not independent training-seed
replication or proof of permanent stability.

Later evidence is inconclusive. Iteration 109 versus 82 had a 37.50%–59.38%
win-credit interval. A subsequent larger-buffer/higher-ratio continuation from
119 scored 44.53% against its start in 256 games (39.06%–50.00%). A lower learning
rate, 0.0003 versus 0.001, scored 52.54% in 256 games (46.29%–58.79%). These are
distinct comparisons; the former bundles several changes and isolates none.

**Decision:** keep fixed-opponent evaluations alongside fit and throughput
metrics. Preserve complete-game history and compare interventions from matched
starting state. The later unresolved results supersede an earlier blanket
interpretation that simply continuing training remained reliably beneficial.

[Selected reports, intervals, checkpoint identities, and provenance](evidence/training-progress.json)
cover the long run, overnight regression, replay4x check-in/endpoint,
replay-growth endpoint, and lower-learning-rate faceoff. They preserve useful
negative evidence without requiring the original replay archives.
