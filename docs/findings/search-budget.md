# Is more search worth its cost?

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

**Next question:** compare matched-start training under explicit budgets,
tracking fresh games and optimizer work. Evaluate prespecified checkpoints with
matched inference; one policy/search screen cannot explain a training plateau.

[Evidence](evidence/search-budget.json): settings, checkpoints, run IDs,
provenance and paired-bootstrap summaries for training and inference comparisons.
