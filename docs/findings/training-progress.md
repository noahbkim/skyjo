# Is self-play training still improving?

**Conclusion, through October 9, 2026:** training produced substantial learning
and later regression. Broader replay helped one continuation; later capacity,
replay-ratio and learning-rate changes did not establish continued gains.
Online loss and self-play scores cannot settle playing strength.

Each row: 64 games, both seats, MCTS32, no root noise, temperature zero, half
credit for ties. Positive margins favor the variant.

| Variant versus control | Variant win credit | Control-minus-variant score, points |
| --- | ---: | ---: |
| Long-run 39 versus 19 | 82.81% | +49.16 |
| Long-run 39 versus 30 | 67.19% | +21.86 |
| Small-FIFO overnight 63 versus starting 50 | 23.44% | −39.33 |
| Small-FIFO 60 versus 50 | 35.16% | −13.08 |
| Four-times-capacity FIFO 60 versus 50 | 76.56% | +24.30 |
| Enlarged-replay 109 versus 66 | 63.28% | +13.63 |
| Enlarged-replay 109 versus 82 | 48.44% | −4.47 |

Iteration 63's paired-seed 95% win-credit interval was 14.06%–34.38%. Both
iteration-60 continuations had 10,240 fresh games; the larger buffer's
common-opponent advantage was 41.41 percentage points (95% interval
25.00–57.03). Capacity rose from 524,288 to 2,097,152 positions at replay ratio
four. Worker count changed, preventing elapsed-time matching. One paired
trajectory does not establish stability across training seeds.

Later results were inconclusive: 109 versus 82 had a 37.50%–59.38% interval.
A larger-buffer/higher-ratio continuation from 119 scored 44.53% against its
start over 256 games (39.06%–50.00%), bundling changes that cannot be isolated.
Separately, learning rate 0.0003 versus 0.001 scored 52.54% over 256 games
(46.29%–58.79%).

**Decision:** keep fixed-opponent evaluations, complete-game history and
matched-start comparisons. Later inconclusive results supersede the earlier
assumption that continued training was reliably beneficial.

[Evidence](evidence/training-progress.json): long-run, regression, replay-capacity,
replay-growth and learning-rate comparisons, with intervals and provenance.
