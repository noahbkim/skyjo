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

**Next run:** twelve-hour weights-only bootstrap from the long-selfplay run's
iteration 19 (19,456 games; checkpoint `checkpoint_20261003_125640_381463.pth`).
Use fresh replay and Adam state, learned group-normalized priors, group-level root
noise, fixed 256 search iterations, mixture backups and sampled chance. Preserve
the 16/32-dimensional model and score/doubling auxiliaries. Generate 1,024 games per
training iteration, replay ratio 4, LR 0.001, capacity 1,048,576 positions, eight CPU
workers. Save every iteration and finish the active batch after the 12-hour soft
cap. This tests sustained learning from deeper fresh targets; it is not an
equal-compute ablation establishing the optimal search budget. Compare against
the initial group-aware snapshot with matched inference settings before claiming
progress. Recipe: `configs/overnight-group256.toml`; source checkpoint is supplied
through a local override.

**Overnight result and tactical check, October 17:** training completed four
1,024-game iterations in 12h16m, generating 1,337,598 fresh targets and 20,902
optimizer updates. Final checkpoint is `checkpoint_000004.pth` in
`.runs/overnight-group256-from19/20261017T004030Z-overnight-group256-early-bootstrap-41f56ef9a6`.

The two giveaway cases regressed during this run. Below are avoidance counts out
of four paired search seeds, using identical PUCT mixture backups, c_puct 1,
parent FPU with zero reduction, sampled chance, boundary 1, no noise, visit-based
greedy acting and symmetry merging. These settings differ from the earlier
AlphaZero screen; do not directly compare its thresholds.

| Case / budget | Original19 | Group-normalized19 | Trained group model |
| --- | ---: | ---: | ---: |
| Give9 /32 | 0/4 | 0/4 | 0/4 |
| Give9 /256 | 0/4 | 0/4 | 0/4 |
| Give9 /2048 | 4/4 | 4/4 | 0/4 |
| Give12 /32 | 0/4 | 0/4 | 0/4 |
| Give12 /256 | 2/4 | 4/4 | 1/4 |
| Give12 /2048 | 4/4 | 4/4 | 4/4 |

For Give9, the new model's median first clearing-witness evaluation was 113.5
iterations, versus 108 for the group-normalized start and 220 for the original
start. It still selected the giveaway in all four searches at 2048. Its raw
giveaway prior fell from 39.1% to 30.9% versus the original start; failure therefore
is not explained solely by a larger root prior or never exploring the reply.
Avoidance labels are tactical hypotheses, not oracle values, and two positions
cannot establish broader strength. [Evidence](evidence/overnight-group256-giveaways.json)
preserves checkpoint/source hashes, raw outcomes, neural probes and summaries.

Completed faceoffs used 256 games each (128 paired environment seeds), eight workers,
new model search256 versus opponent search32. Opponents are the actual
group-normalized starting snapshot and historical iteration 44, the nearest saved
checkpoint to matched additional wall time after 19: 12h28m versus 12h16m. Historical
training used eight workers through 40 then four; code, replay and optimizer
history also differ. These compare model-plus-search setups, not isolated weight
quality or equal CPU work. Outputs: `.runs/overnight-group256-evaluation/`.

The new model scored 54.10% win credit against historical44 (paired-seed bootstrap
95% interval 48.05–59.96%), with a control-minus-new score margin of −5.47 points
(−11.36 to +0.37). Against its group-aware initial19 snapshot it scored 78.13%
(72.66–83.40%) and +23.77 points (+17.62 to +29.71). Thus the requested setup
clearly beats its start but has not established better progress per hour than
historical44, despite using eight times as many search iterations at evaluation.
[Faceoff evidence](evidence/overnight-group256-faceoffs.json) retains the paired
outcomes and checkpoint/settings provenance.

Completed follow-up: 1,024 fresh first-round games (512 paired deals, seats swapped),
historical44/search32 versus trained/search256, eight workers and one thread each.
`run_checkpoint_comparison.py --first-round` stops at the first round boundary
from zero cumulative scores; it retains normal scoring, including doubling, and
leaves search's full-game value objective unchanged. Positive control-minus-new
round score favors the new model. This measures opening-round performance without
later-round score contexts or differing numbers of rounds. Outputs:
`.runs/overnight-group256-evaluation/first-round/`.

The first-round comparison favored historical44: mean round scores 18.585 versus
20.056 for the new model, a new-model disadvantage of 1.471 points (paired-deal
bootstrap 95% interval 0.146–2.810). New-model win credit was 42.33%
(39.16–45.46%), with 433 wins, 590 losses and one draw. These are opening-round
scores, whereas the principal training/search value objective is full-game wins.

**Learning-loop diagnostic, October 17:** before another self-play run, compare
initial group-aware19, historical44, and overnight4 on matched search budgets
32/256/2048 with four search seeds. Reuse six tactical states and 17 exact endgames
plus eight old general controls; generate up to 32 fresh decisions from eight
new first rounds using frozen initial19/search32. Old controls have been inspected
before and are not a held-out validation set. Probe uniform priors and mean-return
backups on tactical states at 2048; measure training-noise targets at 256 for the
new model. Record root priors, visits, Q, highest-Q alternatives, witness values,
raw policy and symmetrized visit targets. Exact action values distinguish target
quality from merely changing a decision.

Use a saved final-generation replay subset split into 48 fitting and 16 validation
games, capped at 4096/2048 positions. Compare original generator3 and successor4,
then fit disposable copies of generator3 with policy-only versus combined losses,
fresh Adam at LR .001, batch256, at 16/64/256 steps. The split is held out only
from this diagnostic fitting, not from the original training. Separately fit
search targets that improve exact expected win credit by >.01; this deliberately
tests fitting capacity on the same examples, not generalization or playing strength.

Finally evaluate up to 16 disagreement positions, prioritizing the two giveaways,
with 16 paired continuation seeds and frozen historical44/search32 in both seats.
Force candidate actions, record remaining-round and full-game score margins and
full-game win credit. These are noisy, policy-dependent action estimates; selecting
positions where methods disagree is not a population performance estimate.
Eight workers, one thread each; source checkpoints and training replay remain intact.
Entrypoint: `run_learning_loop_diagnostic.py`; plan and outputs:
`.runs/learning-loop-20261017/`. All stages run sequentially in one detached job.

**Diagnostic result:** completed in 11m50s, with 2,664 searches and 592 forced-action
continuation games across 16 positions (two giveaways and 14 fresh disagreements).
On the 17 exact endgame controls, new-model probability-distribution regret was
0.10434 for raw policy, 0.04922 for search32 targets, 0.02383 for search256 targets,
and 0.00894 for search2048 targets. Both larger budgets improved expected target
value on all 17 cases, averaged over four seeds. Raw-policy regret was lower than
initial19 (0.12600) and historical44 (0.13438) on these controls. This is expected
win credit under a distribution, not greedy action error or a playing-strength rate.
At search256, adding self-play root noise raised target regret slightly to 0.02633.

The network can fit independently better exact-case targets: policy-only fitting
on 14 improved search256 cases reduced target KL from 0.23366 to 0.00020 and exact
expected regret from 0.12443 to 0.02700; on 15 search2048 cases, KL fell from 0.36154
to 0.000058 and regret from 0.11782 to 0.00994. These same-example fits establish
capacity/optimization on this small set, not unseen tactical generalization.

On the final-generation replay sample, the real generator3-to-successor4 update
barely changed policy KL (validation subset 0.11131 to 0.11048), while value MSE
fell from 0.21487 to 0.18991. In disposable fresh-Adam fits from generator3, at64
steps policy-only training reached train/validation KL 0.08495/0.10805; combined
losses reached 0.10253/0.11642. At256 steps, policy-only validation KL was 0.11273
and combined was 0.12487; combined value MSE overfit from train0.22393 to0.03333,
while validation increased from0.21487 to0.33002. This supports investigating loss
tradeoffs and overfitting on this subset, not a proven explanation of the full run.

Give9 also exposes acting/target allocation: trained search2048 chose the giveaway
in4/4 seeds even though another action had higher final Q in4/4. For seed0 the
giveaway had711 visits/Q0.52315 versus action24's269 visits/Q0.53624. Mean-return
backups still failed4/4; uniform priors avoided3/4. The highest-Q alternative had
better continuation point estimates than the giveaway, but the paired16-seed
intervals include zero. Avoidance itself is not proof of optimality: the initial
model's avoidance move had worse point estimates here; the deep Give12 alternative
also failed to establish a benefit. On the14 selected fresh disagreements, changes
from raw policy had mean win-credit deltas +1.56pp/search32, −3.35pp/search256,
−2.46pp/search2048. These correlated, selected, noisy comparisons do not establish
that deeper search helps general positions. Exact-case success and ordinary-state
value quality must remain separate conclusions.

Next useful isolated checks: fixed-replay loss-weight sweep with a held-out
validation criterion, and matched-weight/search-budget visit-versus-Q acting
comparisons before changing the self-play recipe. No new training run launched.
[Diagnostic evidence](evidence/learning-loop-20261017.json) preserves summaries,
critical tactical traces, and raw-artifact hashes.

**Next isolated comparisons, October 17:** run value-loss weights 0/.1/.3/1,
with score/doubling auxiliaries off/on (weights1/.1 when enabled), policy weight1,
and three paired minibatch seeds101/202/303. Every arm starts from generator3,
with fresh Adam at LR.001, batch256, and checkpoints at0/64/256/1024 updates.
Sample final-generation replay into256 fitting,64 validation,64 test games,
capped at16384/8192/8192 positions; exclude all64 games from the earlier diagnostic.
The source generator3 predates these final-generation targets. Select a step using
validation policy KL only, including unchanged step0; evaluate the test set only
after step selection, and report value MSE alongside policy KL. This is one data
split and three minibatch seeds, not independent self-play training replications.

Then compare visit-based acting against highest estimated Q using the same
frozen overnight4 checkpoint, PUCT mixture search and256 iterations per player.
Only `action_strategy` differs. Run256 full games and1024 fresh first-round games,
paired seats, no root noise, eight workers with one thread each. The first-round
seeds overlap the full-game starts, so these are complementary metrics rather
than independent replications. These faceoffs use the existing frozen checkpoint;
they do not evaluate the loss-sweep fitted copies. All work runs sequentially,
with the eight-worker limit per active stage. Smoke tests completed four tiny loss
ablations and both faceoff modes; six focused tests passed. Entrypoint:
`run_search_followups.py`; plan/status/results: `.runs/search-followups-20261017/`.

**Follow-up results, October 17:** all24 fits and both faceoffs completed.
With auxiliaries retained, value weight0.1 achieved test policy KL0.10867 and
value MSE0.18538, versus0.11427/0.19074 at weight1 (unchanged0.11525/0.19213).
All nonzero-value arms selected64 updates using validation policy KL; longer fits
were worse. Policy-only gave KL0.10798 but worsened value MSE to0.19588. This
supports testing value weight0.1 in self-play; it does not establish playing
strength or the ideal full-replay update count. The sweep used fresh Adam on a
small fixed split; the next continuation retains existing Adam moments.

Highest-Q acting lost to visits at identical frozen weights and256 search visits:
93 wins/163 losses in256 full games (36.33% win credit, paired95% CI30.08–42.58%),
and389 wins/633 losses/2 draws in1024 first rounds (38.09%, CI35.11–41.02%).
Q scored7.54 more points per first round (worse), paired95% CI5.87–9.23.
Keep visit acting and targets. [Evidence](evidence/search-followups-20261017.json)
records all sweep arms, faceoff settings, checkpoint hashes and paired intervals.

**Next run, October 17:** one eight-worker continuation from overnight4, fixed256
search,1024 games per iteration, group-normalized learned priors and group root
noise. Change only value weight1 to0.1; retain score/doubling auxiliaries1/.1,
LR.001, replay ratio4, weights, Adam state, replay, RNG and progress counters.
Recipe: `configs/overnight-group256-value01.toml`; local source configuration:
`configs/local/overnight-group256-value01-from4.toml`. A22h soft budget stops
after a complete batch (~3h), aiming for22–25h through October18 evening.
Artifacts: `.runs/overnight-group256-value01-from4/`.

Use all workers for this candidate rather than training another control. Match
the original baseline lineage by cumulative training elapsed after shared
iteration19: candidate's existing44150.45334387501 seconds plus this continuation's
actual elapsed time. Follow historical checkpoint ancestry and sum recorded
training intervals, excluding gaps between runs; select the closest checkpoint
after the candidate finishes. Preserve the prior comparison's baseline32 versus
candidate256 inference settings and report both full-game and first-round scores.
Historical worker counts and recipes differ, so this compares complete training
approaches per elapsed time and cannot isolate value weighting causally.

Disk cleanup losslessly archived the completed fixed/randomized pilot replay
datasets under `.runs/randomized-budget-20261016/`, verifying every file hash
before removing the unpacked copies. Reclaimed8.12GiB; archives and inventories
remain beside each run's data. Preserve the active source replay and all
checkpoints. Compact faceoff artifacts remain (only a few MB).

**Continuation completed, October 18:** overnight value0.1 finished normally at
19:00 Eastern after81538.1s (22h39m), reaching checkpoint12. Eight additional
iterations generated8192 games and2475670 positions. Checkpoint12 SHA256:
`21159b7fec97cd3a46d76234429dfa3e6f5b493bc97e02d4d7275c046c6f1aac`.
Heuristic target probability rose0.4219 to0.4652; target matches remained6/10.
These checks do not establish stronger play.

**Approved evaluation queue:** checkpoint12 versus historical87 at256/256,
then versus pre-continuation4 at256/256, then historical87 at baseline32 versus
candidate256. Each comparison runs256 full games and1024 fresh first rounds,
eight workers, one thread each, balanced seats, temperature0, no root noise,
visit acting and shared PUCT settings. Preserve each checkpoint's policy
normalization (historical actions; candidate/pre-continuation groups).
Use common fresh seeds across comparisons but separate seed streams for full
games and first rounds. Report win credit, opponent-minus-candidate score margin
(positive favors candidate), and paired-seat seed-cluster bootstrap95% intervals.

Historical87 has126008.90 seconds (35.0025h) after shared iteration19, nearest
candidate125688.56 seconds (34.9135h), a320.34s difference. The match sums
checkpoint elapsed intervals along the actual ancestry, excluding idle gaps.
[Plan and ancestry](evidence/value01-faceoff-plan-20261018.json) pin checkpoints,
hashes, settings and selection. Artifacts and sequential queue status:
`.runs/value01-faceoffs-20261018/`. Prior256/256 faceoffs took3731s/full256 and
4764s/first-round1024; expect roughly6h total, allowing5–8h for game-length
variation. No contemporaneous value1 continuation exists, so these comparisons
cannot isolate the causal contribution of value weighting.

**Faceoff results, October 19:** all six completed at01:18 Eastern in2h59m,
faster than estimated. Each row has256 full games and1024 first rounds; positive
score advantage means opponent score minus candidate12 score (lower is better).

| Opponent / search budgets opponent:candidate | Full-game win credit | Full-game score advantage | First-round win credit | First-round score advantage |
| --- | ---: | ---: | ---: | ---: |
| Historical87 /256:256 | 53.52% | +3.16 | 51.76% | +0.79 |
| Pre-continuation4 /256:256 | 83.20% | +58.29 | 85.94% | +13.68 |
| Historical87 /32:256 | 57.03% | +5.18 | 50.39% | +2.09 |

At equal256 search, historical87 full-game win-credit95% CI47.46–59.38% and
score CI−1.45..+7.76 include parity; first-round score CI−0.18..+1.82 does too.
Thus candidate12 is competitive with the time-matched baseline, with no clear
superiority established. Against checkpoint4, full-game win CI78.13–87.89% and
first-round score advantage CI12.33–15.01 establish substantial improvement
along this continuation. Against historical87/search32, full-game win
CI50.78–63.09% and score CI0.18–10.29 suggest a modest advantage; first-round
win CI47.46–53.42% is inconclusive despite a score advantage CI1.05–3.15.

These are marginal paired-seat seed-cluster bootstrap95% intervals, unadjusted
for multiple comparisons; arms share seeds. Eight workers were verified in all
six raw reports. [Results and provenance](evidence/value01-faceoff-results-20261019.json)
preserve settings, checkpoint/report hashes and all intervals. The continuation
learned substantially, but extra training and value-weight reduction remain
confounded; these results alone do not prove value0.1 caused the improvement or
that search256 self-play is more efficient than the historical recipe.

**Tactical recheck, October 19:** run the exact same two giveaway fixtures and
PUCT mixture plan as October17, with seeds0–3 and32/256/2048 iterations, on
candidate12 and historical87. Reuse the preserved matching checkpoint4 results;
plans and fixture hashes agree across all three.

| Position / budget | Before4 avoids | Candidate12 avoids | Historical87 avoids |
| --- | ---: | ---: | ---: |
| Give9 /32 | 0/4 | 0/4 | 0/4 |
| Give9 /256 | 0/4 | 0/4 | 0/4 |
| Give9 /2048 | 0/4 | 4/4 | 4/4 |
| Give12 /32 | 0/4 | 0/4 | 0/4 |
| Give12 /256 | 1/4 | 0/4 | 0/4 |
| Give12 /2048 | 4/4 | 4/4 | 4/4 |

Candidate12's stable avoidance starts at1551–1710 iterations on Give9 and
1671–1869 on Give12, remaining stable through the2048 tested horizon. Historical87
starts at930–1139 and1022–1191 respectively. Before4 never stabilized on Give9
through2048, but stabilized much earlier on Give12 (459–487). Thus tactical
progress is mixed and neither case is solved at the deployed256 budget.

Candidate12 raw policy still chooses both giveaways. Its Give9 giveaway prior
rose30.9%→68.4% from checkpoint4 (historical87:52.7%), while the modeled opponent
clearing-action prior fell4.95%→2.38%. Give12 root giveaway prior remained31.6%
(previous31.5%). Deeper recovery therefore did not come from a policy that
learned to avoid these cards directly. These two selected positions and their
avoidance labels do not prove the alternatives are optimal or measure general
tactical strength. [Evidence](evidence/value01-tactics-20261019.json) preserves
all72 trial summaries, model probes and source hashes.

**Own-clear recognition screen, October 19:** separate acquiring a clearing card
from placing it. Use `explain.create_obvious_clear_position` (visible10/10/? with
discard10), its deterministic TAKE successor, and `create_almost_clear_position`
(discard11). Probe raw policy and four search seeds0–3 at32/256 with the same
PUCT mixture configuration as tactical checks.

| Model | Raw P(TAKE clearing10) | Raw P(clearing placement after TAKE) | TAKE choices at32 /256 | Clearing placement choices at32 /256 |
| --- | ---: | ---: | ---: | ---: |
| Before4 | 47.95%, DRAW argmax | 17.12%, nonclear argmax | 4/4 /4/4 | 4/4 /4/4 |
| Candidate12 | 39.70%, DRAW argmax | 21.54%, clear argmax | 0/4 /4/4 | 4/4 /4/4 |
| Historical87 | 45.92%, DRAW argmax | 8.28%, nonclear argmax | 0/4 /0/4 | 0/4 /4/4 |

Candidate12's search256 average visit targets assign only51.37% to TAKE and
58.20% to the clearing placement. Before4 targets were88.96% and85.25%. Thus
search can choose the clear while supplying a relatively weak policy target;
greedy success alone hides this distinction. Candidate12 raw TAKE probability
is39.70% with the clearing10 versus38.69% with nonclearing11.

Construct four positive-rank variants3/6/9/12, each with a matched no-clear
control changing the second matching visible card from rank to rank−1. Probe
both before and after TAKE; validate states and confirm the intended placement
clears only in the matching condition using the engine. These are policy-only
probes, not additional search trials. All three models choose DRAW on all four
matching examples. Candidate12 chooses the clearing slot after TAKE in4/4,
before4 in2/4, historical87 in0/4. But candidate12 also chooses that slot in all
four nonmatching controls: its behavior is partly a general placement preference.
Matching increases its placement probability by3.05–3.89 percentage points,
while changing TAKE probability by at most0.56pp in either direction.

This screen supports weak policy sensitivity to the two-step clearing opportunity
and search reliance, with some placement sensitivity; it does not establish
general clear recognition. Heuristic examples are not optimal full-game oracles.
Next useful coverage: varied board/score contexts, matched clearing/nonclearing
controls, negative-card cases where clearing can be undesirable, and both own
opportunities and opponent threats. Keep acquisition, placement, policy mass,
search choices and visit targets separate. [Evidence](evidence/clear-recognition-20261019.json)
contains72 searches, all raw policy probes, recipes and checkpoint/source hashes.

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
