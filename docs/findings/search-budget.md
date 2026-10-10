# Is more search worth its cost?

**Conclusion, evidence through October 9, 2026:** more inference search helped
some frozen models, but the effect depends on checkpoint and comparison.
Higher-search self-play has not established better learned weights per hour.

At the same original frozen checkpoint, MCTS128 beat MCTS32 with 75% win credit
and a +26.50-point margin over 32 full games. That small inference screen says
nothing directly about training efficiency.

Matched-start continuations then trained with 32 or 128 visits. Each faceoff
below has 64 games, both seats, temperature zero and no root noise; both sides
use the listed evaluation budget. The variant is the search128-trained model.

| Training checkpoint pair | Evaluation visits | Variant win credit | Margin, points |
| --- | ---: | ---: | ---: |
| Equal fresh games, iteration 41 | 32 | 37.50% | −21.06 |
| Equal fresh games, iteration 41 | 128 | 73.44% | +18.64 |
| Equal fresh games, iteration 42 | 32 | 39.06% | −17.34 |
| Equal fresh games, iteration 42 | 128 | 39.06% | −21.81 |

The iteration-41 result reverses with inference budget; iteration 42 loses at
both. Better scores in generated games therefore did not establish a robust
training advantage. These are small repeated screens from one paired trajectory,
not a universal search-budget optimum.

At later checkpoint 119, MCTS32 versus the same weights' policy argmax scored
53.91% in 64 games (paired 95% interval 42.19%–65.63%). MCTS128 scored 58.98% in
128 games (50.00%–67.97%), with score-margin interval −1.22 to +15.10 points.
Those tests used different seeds and did not directly compare 32 versus 128.
Neither provides a clear full-game strength conclusion; compute was deliberately
unequal.

**Next question:** compare matched-start training under an explicit budget,
then evaluate several prespecified checkpoints under matched inference settings.
Track fresh-game diversity and optimizer work along with search cost. Do not
infer a cause of a training plateau from a single policy/search screen.

[Selected comparison records](evidence/search-budget.json) preserve budgets,
checkpoint identities, historical run IDs, code provenance where recorded, and
the later paired-bootstrap summaries. Search-training and inference comparisons
remain separate evidence.
