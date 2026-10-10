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

Trees bind to evaluator instances and immutable search settings. Validate reuse
before mutation: mixing exact-chance and sampled-chance nodes previously
corrupted their weights. Iteration budgets and action temperature remain separate
because they govern work and action selection, not evaluator identity.

The engine enumerates ordinary single-card chance outcomes. Round endings can
reveal many hidden cards, so their sampling stays explicit. Even a deterministic
round ending may lead to an uncertain next deal; it must still honor the requested
boundary sample budget. Completed games use exact outcomes. Symmetry grouping
must respect deck-recycling safety, not just visible board equivalence.

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
