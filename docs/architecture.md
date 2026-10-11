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

Visits count traversals, not evaluator calls or initialization samples. Root
evaluation initializes priors and FPU without a visit; outgoing action children
start at zero visits. PUCT allocates initial exploration using those priors and
FPU, without a forced sweep of every action. Decision values average one return
per traversal in fixed player order, without sign flips between players.

Sampled chance nodes include their first realized decision child in the path,
just like later new outcomes. Both receive one backup of the child's model value;
the chance total starts at zero so initialization is not counted twice. Exact
chance nodes instead initialize the probability-weighted expectation over all
evaluated children, leaving those children at zero visits. Subsequent traversals
update the selected child's estimate, adjust the chance expectation by its
probability-weighted delta, and propagate that updated expectation upward.

The engine enumerates ordinary single-card chance outcomes. Round endings can
reveal many hidden cards, so their sampling stays explicit. A round boundary
remains a traversal leaf: its first visit evaluates `boundary_samples` outcomes,
and each revisit evaluates one fresh outcome from the original pre-terminal state.
Each outcome is evaluated before averaging, including exact completed-game values
and fractional tie credit. PUCT uses the pooled mean with equal weight per outcome;
the initial batch mean is backed up once, followed by each fresh singleton return.
Thus boundary sample counts and ancestor visit counts intentionally differ. For
example, initial outcomes 0, 0 followed by 1 give a boundary estimate of 1/3 and a
direct ancestor's per-visit estimate of 1/2. Later iterations within that search
continue sampling without repeating the initial batch.

Even a deterministic round ending may lead to an uncertain next deal: its
completed state can be reused, but the evaluator must run on each revisit. Only
provably deterministic game-terminal boundaries cache their exact result without
further samples. A sampled terminal result alone does not prove determinism.
Symmetry grouping must respect deck-recycling safety, not just visible board
equivalence.

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
