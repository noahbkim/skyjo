# Do better boundary estimates improve play?

**Conclusion, through October 9, 2026:** terminal averaging and pre-deal score
values improved local estimates. Neither established stronger full-game play.

On 1,344 held-out boundaries from 911 checkpoint-50 games, value MSE was
0.195429/0.193897/0.193941 for 1/10/100 next-round deals and 0.191582 for the score
MLP. MLP-minus-single-deal: −0.003848 (game-bootstrap 95% interval
−0.006037 to −0.001530). Different model training histories limit causal inference.

Against exact references for 389 actions across 32 one-hidden-card terminal
states, 1/10/100 ending samples gave mean regret of 3.847/0.309/0.035 percentage
points over 32 repetitions. This measures action ranking, not MCTS strength.

Full-game tests used both seats, MCTS32, temperature zero, no root noise.
K10 combines ten ending samples with a frozen score evaluator; baseline uses
one ending and a next-round deal.

| Question | Games | K10-side win credit | Paired interval |
| --- | ---: | ---: | --- |
| K10-trained 119 versus unchanged-training 119, same baseline inference | 768 fresh | 51.50% | 47.46%–55.47% (97.5%) |
| K10-trained 119 versus starting 109, same baseline inference | 768 fresh | 51.56% | 47.59%–55.47% (97.5%) |
| K10 versus baseline inference, identical 119 weights | 1,024 | 51.12% | 48.29%–54.00% (95%) |

Training confirmations were prespecified after checkpoint selection; 97.5%
intervals address two comparisons. Inference used fresh seeds and a fixed
1,024-game budget; score advantage was +0.74 points (95% interval −1.62 to +3.12).
These intervals permit gains or losses. Each arm has one training trajectory;
equal visits do not mean equal runtime.

**Decision:** keep boundary estimators optional. Test sample count and evaluator
separately under explicit compute budgets before attributing training stagnation
to boundary error.

[Evidence](evidence/boundary-evaluation.json): boundary/terminal reports, training
confirmations, inference comparison, source hashes and provenance.
