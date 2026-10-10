# Do better boundary estimates improve play?

**Conclusion, evidence through October 9, 2026:** averaging terminal outcomes
and fitting pre-deal score values improved local estimates. Neither the tested
training change nor the complete inference change established stronger full-game
play. Keep those conclusions separate.

On 1,344 held-out nonterminal boundaries from 911 games generated after
checkpoint 50's training, value MSE was 0.195429 for one next-round deal,
0.193897 for ten, 0.193941 for 100, and 0.191582 for the score MLP. The
MLP-minus-single-deal difference was −0.003848 (whole-game-bootstrap 95%
interval −0.006037 to −0.001530). The models had different training histories;
this measures frozen predictive quality on that policy's games.

Separately, exact enumeration of 389 actions across 32 one-hidden-card terminal
states gave mean action regret of 3.847, 0.309, and 0.035 percentage points for
1/10/100 ending samples over 32 repetitions. This evaluates sampled action
rankings against exact references, not complete MCTS play.

Full-game evaluations used both seats, MCTS32, temperature zero and no root
noise. K10 combines ten ending samples with a frozen pre-deal score evaluator;
the baseline uses one ending plus a next-round deal.

| Question | Games | K10-side win credit | Paired interval |
| --- | ---: | ---: | --- |
| K10-trained 119 versus unchanged-training 119, same baseline inference | 768 fresh | 51.50% | 47.46%–55.47% (97.5%) |
| K10-trained 119 versus starting 109, same baseline inference | 768 fresh | 51.56% | 47.59%–55.47% (97.5%) |
| K10 versus baseline inference, identical 119 weights | 1,024 | 51.12% | 48.29%–54.00% (95%) |

Fresh training comparisons were prespecified confirmations after earlier
checkpoint selection; their 97.5% marginal intervals address two comparisons.
The inference test had a fixed 1,024-game budget and fresh seeds. Its score
advantage was +0.74 points (95% interval −1.62 to +3.12). These intervals permit
modest gains or losses and do not establish equivalence. They condition on one
training trajectory per arm; matching search visits does not match runtime.

**Decision:** retain boundary estimators as explicit experiment choices. Require
a measured full-game benefit before claiming the improved local estimates solve
training stagnation. Further work should separate sample count from continuation
evaluator and state the compute budget.

[Selected evidence](evidence/boundary-evaluation.json) includes the completed
boundary-methods/terminal reports and the extended-training and same-checkpoint
inference summaries, with source hashes and recorded provenance.
