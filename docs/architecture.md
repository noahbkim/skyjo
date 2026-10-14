# Architecture decisions

Keep game mechanics, search, and simulation independent of Torch so they can be
profiled and optimized without choosing a model or training backend.

## Boundaries that matter

Simulation returns histories rather than training labels. This lets objectives
change without changing gameplay, and lets diagnostics analyze the same observed
games. Networks operate only on tensors; inference adapters handle encoding,
legal-policy normalization, batching, and player perspective.

Search values use **fixed player order**, while network outputs and training
targets use **current-player order**. Convert at the inference boundary: otherwise
values accumulated across turns refer to different players. The primary target
is eventual game-win credit, shared on ties. Auxiliary round scores supplement
that target; resampling a round ending must not change the observed game winner.

Replay retains and evicts complete games and splits by game to avoid leaking
correlated positions across training and validation. Persistence imports data
through replay's public interface so file formats do not dictate storage layout.

## Search correctness and cost

Treat state arrays as immutable: transitions copy changed data so search branches
can share predecessors safely.

Search accepts a game state and constructs a fresh tree for each call. Callers
cannot supply an existing node; nodes and their statistics persist only across
iterations within that search. The returned tree supports policy extraction and
diagnostics. This keeps evaluator bindings, chance modes, and symmetry safety
local to one search. Iteration budgets and action temperature govern search effort
and action selection separately.

Allocation, valuation, action selection, and learning targets are independently
composable. The runner exposes immutable statistics to strategy sessions and
completed-result policies; it owns transitions, chance handling, evaluator calls,
and budgets. Algorithm selection stays at the configuration boundary. See
[search strategies](search-strategies.md) for the contracts, presets, and
independent-replica uncertainty. The following mixture description specifies the
compatibility default; other valuations deliberately use different backups.

Visits count traversals, not evaluator calls or initialization samples. Root
evaluation initializes priors and FPU without a visit; outgoing action children
start at zero visits. PUCT allocates initial exploration using those priors and
FPU, without a forced sweep of every action. Values use fixed player order,
without sign flips between players. Each evaluated decision node retains one
neural contribution, including the root and pre-evaluated chance outcomes:

`V = (neural_value + sum(child.visits * child.value)) / (1 + sum(child.visits))`.

The neural contribution adds no traversal visit. Backup increments each node on
the traversed path once, then refreshes cached mixtures bottom-up using the
children's current estimates. It does not accumulate successive versions of a
child's running mean as fresh returns. Refreshing a mixture reads only immediate
children; it does not recursively recompute the whole tree.

Mean-return valuation instead averages fresh simulation evidence without an
additional neural contribution. Its initial estimate is the neural prediction
until the first return arrives. Best-response valuation propagates the evaluated
child that maximizes the acting player's component, preserving the full fixed-seat
vector for its ancestors. Neither valuation selects a best chance outcome.

Sampled chance nodes include their first realized decision child in the path,
just like later new outcomes. Each realized outcome increments its frequency
once, and the chance value mixes current conditional child values using those
frequencies. Exact chance nodes instead initialize the probability-weighted
expectation over all evaluated children, leaving those children at zero visits.
Subsequent traversals still sample one continuation, but the expectation always
uses fixed outcome probabilities, regardless of how much work each child receives.
Both modes propagate their updated mixtures through the same backup mechanism.

The engine enumerates ordinary single-card chance outcomes. Round endings can
reveal many hidden cards, so their sampling stays explicit. A round boundary
remains a traversal leaf: its first visit evaluates `boundary_samples` outcomes,
and each revisit evaluates one fresh outcome from the original pre-terminal state.
Each outcome is evaluated before averaging, including exact completed-game values
and fractional tie credit. PUCT and ancestor mixtures use the pooled mean with
equal weight per outcome. Boundary sample counts and ancestor visit counts
intentionally differ: initial outcomes 0, 0 followed by 1 give a boundary value
of 1/3 after two visits. A parent with neural value 0.5 and only this action
visited has value `(0.5 + 2 * (1/3)) / 3 = 7/18`. It retains one neural
contribution but no historical versions of the boundary estimate. Later iterations
continue sampling without repeating the initial batch.

For mean-return backup, that boundary supplies the fresh batch mean as one
simulation return: batches `[0, 0]` then `[1]` give the decision ancestor a mean
of `1/2`, while the boundary's pooled outcome estimate remains `1/3`. The two
quantities intentionally measure different evidence. A refreshed historical
estimate is never counted as a fresh return.

Even a deterministic round ending may lead to an uncertain next deal: its
completed state can be reused, but the evaluator must run on each revisit. Only
provably deterministic game-terminal boundaries cache their exact result without
further samples. A sampled terminal result alone does not prove determinism.
Symmetry grouping must respect deck-recycling safety, not just visible board
equivalence. Search checks the unseen-card reserve at each expanded decision
state instead of disabling grouping solely because the requested budget is large.
If an explored path loses the reserve needed to avoid recycling during final
reveals, it discards the grouped tree, restores the search RNG, and rebuilds the
full requested budget without grouping. No values from the discarded tree are
reused. Evaluators are expected to predict from their inputs without depending
on call history. Replica searches retain the conservative whole-budget check
to guarantee identical action groups across all replicas; single-tree searches
use the explored-path fallback. Allocation sessions are recreated on retry.

Transitions perform cheap precondition checks; deep invariant validation is
explicit. This keeps diagnostic cost out of the hot path while preserving a way
to check state integrity. See the [timing evidence](findings/architecture-performance.md).

## Reproducibility and execution

Derive randomness from stable run/game identities, with separate streams for
environment outcomes, action selection, search, auxiliary resampling, and replay
sampling. Changing worker counts or task grouping must not change a game's
trajectory; changing search effort must not consume environment randomness.

Keep requested settings immutable and track progress and schedule changes as
runtime state. Save requests before substituting run-local artifact paths so
continuation comparisons describe actual setting changes.

Online and offline recipes share the learner, but keep explicit execution loops:
self-play workers refresh evolving weights, while evaluation workers retain
frozen contestants. Explicit loops make those different lifecycles visible.
Keep exact restoration separate from changing run settings so reproducibility
remains testable.
