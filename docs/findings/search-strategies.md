# Search strategies on frozen tactical and endgame positions

October 12, 2026. **Best-response backups avoided the first recorded giveaway in
8/8 searches at 2,048 traversals; the control needed 8,192.** The second giveaway
still needed 8,192 under both strategies. Canonical Gumbel planning never explored
the critical clearing reply in these tests, and splitting the budget into eight
searches did not repair that discovery failure. These results support keeping
production defaults while continuing targeted exploration experiments.

This is an inference-only comparison using the existing iteration-119 checkpoint,
which was trained under the previous search procedure. It does not measure the
strength of a model trained with Gumbel targets, nor establish full-game strength.
No retraining or full-game faceoffs were performed.

## Protocol and provenance

The comparison completed **3,264 primary searches**:

- 1,216 tactical trials: two recorded positions; PUCT/Gumbel across three backups;
  Gumbel noise scales 0/1; and eight-replica uncertainty selectors with z=0/1/2.
- 2,000 controls: 17 existing exact endgames plus eight existing stratified replay
  positions, five strategies, budgets 256/2,048, and eight seeds.
- 48 follow-up trials: Gumbel root allocation with PUCT interior allocation at
  2,048, all three backups, both tactical positions, and eight seeds.

Tactical single-tree budgets were 256/2,048/8,192, with eight search seeds per
condition. Main runs enumerated in-round chance; tactical conditions also ran
sampled chance at 2,048. Replica conditions shared a **total** 2,048-traversal budget
(eight trees of 256 each), including initial root-action coverage.

All conditions retained the recording's c_puct=1.5, boundary initial batch=10,
no Dirichlet noise, symmetry safety checks, and zero acting temperature. The
checkpoint SHA256 is
`dcbe5ec51a529460de5c8f6ad91d37fcdef5e0e83f183474832976c3be5d976f`.

The [frozen fixtures](../../configs/benchmarks/search_positions.json) retain
source hashes and zero-based frame indices 53 and 112. GIVE and TAKE are
deterministic; replacing the opponent's final hidden cell consumes randomness,
but always clears the 9/9/? or 12/12/? column. The visible score falls 22→4 or
50→26 regardless of the revealed card. Engine tests verify that distinction.

The control uses parent-value-minus-0.1 FPU. The PUCT baselines use the planned
zero FPU, including best response. Values are win probabilities in [0,1], so
zero is pessimistic rather than neutral. This matters: at 256 the control finds
the first clear in 8/8 trials; zero-FPU PUCT finds it in 0/8. Comparisons against
the control therefore include this exploration difference.

## Tactical action selection

Each entry counts avoidance of the specific recorded giveaway, out of eight
seeds, with enumerated chance. Avoidance is a diagnostic, not an optimal-action
label; an alternative hidden replacement can still reveal the clearing rank.

| Strategy | R1 / 256 | R1 / 2,048 | R1 / 8,192 | R2 / 256 | R2 / 2,048 | R2 / 8,192 |
|---|---:|---:|---:|---:|---:|---:|
| Control: current mixture, parent FPU | 0/8 | 0/8 | 8/8 | 0/8 | 0/8 | 8/8 |
| AlphaZero-style PUCT: mean return | 0/8 | 0/8 | 8/8 | 0/8 | 0/8 | 0/8 |
| Gumbel: mean return, noise 0 | 0/8 | 0/8 | 2/8 | 0/8 | 0/8 | 0/8 |
| Gumbel: mean return, noise 1 | 0/8 | 0/8 | 4/8 | 1/8 | 0/8 | 0/8 |
| PUCT: best-response backup | 0/8 | 8/8 | 8/8 | 0/8 | 0/8 | 8/8 |

At 2,048, replacing visit-based selection by highest estimated value on the
**same control trees** avoids the first giveaway in 8/8 searches and the second
in 5/8. No extra evaluations are involved. On the AlphaZero-style trees at 8,192,
highest-value selection avoids both in 8/8, despite visits still choosing the
second giveaway. This confirms that allocation, value propagation, and final
selection can fail separately.

The original saved diagnostics show the same distinction: frame 53's highest-Q
choice is action 21 rather than the played action 18, with estimated advantage
0.01339. Frame 112's played action 27 also has the highest recorded Q; changing
that original selection rule alone cannot fix it.

## Gumbel and uncertainty ablations

The canonical Gumbel interior rule never visits the clearing reply in **384/384
trials**, including the 8,192 budget. Some runs avoid a giveaway without finding
its refutation. This is not evidence of tactical understanding, and a root
winner or improved training target cannot correct an unexamined reply.

Switching only the interior allocation to PUCT exposes the first clear in all
24 hybrid trials at 2,048. With best-response backup, the hybrid then avoids that
giveaway in 8/8; current-mixture and mean-return backups still choose it in 8/8.
The second clear remains undiscovered in all 24 hybrid trials at this budget.
This isolates a useful benefit of independently composing root allocation,
interior allocation, and backup.

Eight-replica selectors discover neither clear. Giveaway avoidance is 0/288
with enumerated chance and 15/288 with sampled chance; the latter choices occur
without examining the clear. Increasing z does not consistently help. Dividing
the total budget leaves each search too shallow, and between-search variation
does not estimate the shared error from a missed reply or a biased model.

## Exact endgames and replay controls

Each cell covers 17 frozen exact endgames × eight search seeds. References are
independently re-enumerated through the engine, including optimal final placement
after DRAW or TAKE. Regret is lost terminal win probability in percentage points.

| Strategy | Errors / 136 at 256 | Mean regret at 256 (pp) | Errors / 136 at 2,048 |
|---|---:|---:|---:|
| Control: current mixture, parent FPU | 0 | 0.000 | 0 |
| AlphaZero-style PUCT: mean return | 0 | 0.000 | 0 |
| Gumbel: mean return, noise 0 | 1 | 0.053 | 0 |
| Gumbel: mean return, noise 1 | 5 | 0.370 | 0 |
| PUCT: best-response backup | 0 | 0.000 | 0 |

Highest-Q selection is not uniformly safer: on the control's 256-traversal
trees it introduces 7/136 errors and 0.412 pp mean regret, versus zero under
visit selection. AlphaZero-style PUCT and best response each have 1/136
highest-Q errors at 256. All tested selectors have zero exact regret at 2,048.

The eight earlier-game controls have no oracle. Their selected actions, root
values, evaluator work, and elapsed times are retained in the evidence, but
these small stratified samples do not establish population-level improvement.

## Cost and verification

The protocol abstraction has a measurable overhead when evaluator work is
cheap. In a pure-NumPy uniform-evaluator opening-state benchmark (2,048
traversals, 2,049 evaluated states, four repeats), median process CPU time rose
from **0.446 s to 0.631 s**, about **41.5%**. The baseline is clean commit
`fe7d4674c1b08462c59e3d259750881824de93b8`. This isolates search overhead; it is
not a neural-throughput measurement. Immutable snapshots provide encapsulation
at a cost. Sweep wall times were collected under overlapping process workloads,
so evaluator-state counts are the preferred work comparison; no speedup is claimed.

The main runs began before final immutability and encapsulation cleanups. A
56-trial replay spanning every initial strategy exactly matches all original
numerical fields: choices, values, visits, priors, acting and learning policies,
uncertainty, evaluator work, and reply coverage. Only elapsed time is excluded.
Focused fixture/benchmark tests passed (13); the final full suite passed (384).

[Machine-readable evidence](evidence/search-strategies.json) includes aggregates,
source manifests, raw-output hashes, the CPU measurements, and replay verification.
[Reproduction instructions](../search-benchmarks.md) describe the complete matrix
and optional hybrid variants. Local raw outputs remain available as
[tactical trials](../../.runs/search-strategy-tactical/searches.jsonl),
[control trials](../../.runs/search-strategy-controls/searches.jsonl), and
[hybrid trials](../../.runs/search-strategy-hybrid/searches.jsonl), with manifests
and benchmark-source snapshots beside them. The
[final-source replay check](../../.runs/search-strategy-final-smoke/replay-verification.json)
records the exact numerical agreement.
