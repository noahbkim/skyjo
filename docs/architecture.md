# Architecture

The engine describes legal play; search chooses promising actions; simulation
executes games; learning fits models to their histories. Experiments connect
these pieces and analytics interprets their results.

## Dependencies and ownership

```text
experiments ──► analytics, learning, simulation, search, engine
analytics ───► learning (model diagnostics), engine
learning ────► simulation, search contracts, engine
simulation ──► engine
search ──────► engine
engine ──────► NumPy
```

Engine, search, and simulation have no Torch dependency. In learning, tensor
models remain independent of replay storage and experiment orchestration. Keep
package initializers small so importing a low-level component cannot load the
training stack indirectly.

Simulation accepts any object implementing the player protocol; it does not
import search. Statistical reducers and report rendering consume recorded data.
The separate `analytics.explain` concept probes import learning and Torch because
they inspect a model directly; their heuristic labels never become replay targets.

| Area | Owns | Receives or returns |
| --- | --- | --- |
| Engine | Copy-on-write NumPy state, legal actions, transitions, scoring, symmetries, validation | States, actions, chance outcomes |
| Search | MCTS state, immutable search settings, action grouping, diagnostics | Supplied evaluator; action probabilities |
| Simulation | Player protocol, complete games and rounds, job identities, random streams | Supplied players; histories and outcomes |
| Learning | Encoding, tensor model, inference adapter, targets, replay, losses, optimizer, checkpoint I/O | Histories → prepared batches → model updates |
| Analytics | Statistical reducers and reports | Histories, metrics, match and training results |
| Experiments | Config parsing, component construction, workers, schedules, run artifacts | Requested settings → recorded results |

## Data and value contracts

`Skyjo` is a frozen dataclass containing NumPy arrays. Transitions create new
state instead of editing their input. Preserve action numbering and distinguish
ordinary single-card chance outcomes from final-action/hidden-card round-ending
sampling. Cheap preconditions protect transitions; full state validation belongs
in explicit diagnostics and invariant tests.

Search's batched evaluator accepts states and returns NumPy predictions:
legal-action probabilities and game-win values in **fixed player order**. The
neural adapter owns encoding, device movement, softmax, and conversion from
current-player-relative outputs. Never silently mix those two player orders.
Tied winners share credit. Search utility is eventual game-win probability;
auxiliary round scores do not replace it.

The model operates on tensors and returns value, policy logits, and named
auxiliary outputs. Learning owns the target normalization and losses for
`round_score`, `round_raw_score`, and `round_doubled`. Simulation records the
chosen action and its policy with the game history; it does not construct labels.
Every position receives its observed full-game outcome, while round objectives
use the corresponding round. Resampling auxiliary round endings does not
resample the observed full-game winner.

Replay receives prepared batches with game provenance. It retains and evicts
whole games, samples training positions, and keeps games disjoint across splits.
Dataset I/O validates and imports data through explicit operations; recipes do
not repair replay internals.

## Search and execution

Trees belong to their search configuration and evaluator instances. Reject
incompatible reuse before changing a tree. Iteration budget and action-selection
temperature are separate from the tree's search settings. Symmetric action
grouping must retain the deck-recycling safety check.

Boundary evaluators are supplied to search. The baseline samples an ending and
evaluates a next-round deal; the optional score evaluator averages sampled
endings before a new deal. Completed games always use their exact outcome.
Experiments load and snapshot evaluator checkpoints, keeping file access out of
search.

Runtime settings use frozen dataclasses; counters and active schedule state
belong to `TrainingState`. The online loop stays explicit:

```text
generate games → construct labels/encode → append replay → optimize → evaluate/save
```

Online and offline recipes share model, optimizer, loss, and learner operations.
Self-play workers refresh changing weights; evaluation workers keep frozen
contestants. Share job/result contracts without concealing those different
lifecycles in a generic execution framework.

Own random generators at the operation boundary. Derive game streams from stable
run/game identities, with separate streams for environment outcomes, action
sampling, search, auxiliary resampling, and replay sampling. Worker count and
task grouping must not change a game's identity. Offline exact resume restores
checkpoint state under matching settings and data. Online continuation starts a
new recorded run from a checkpoint and matching replay, preserves inherited
progress, and permits explicit setting changes. Historical seeded trajectories
and obsolete artifacts are not a compatibility contract.

## Where to extend

| Change | Start here | Verify |
| --- | --- | --- |
| Rule or transition optimization | Engine | State immutability, card conservation, scoring, exact chance probabilities |
| Search strategy or boundary method | Search contract and evaluator implementation | Fake-evaluator tests, perspective conversion, safe tree reuse |
| Network architecture | Learning model and construction | Shapes, legal policies, equivariance |
| Auxiliary objective | Learning head, target, normalization, loss | Known examples and objective composition |
| Training/evaluation recipe | Experiments using shared components | Small real run, saved artifacts, reproducibility |
| Metric or interpretation | Analytics | Aggregate agrees with primitive records |

Profile transition, search, inference, and optimizer work separately before
changing storage or execution backends. A faster microbenchmark does not imply
stronger play or faster training to a fixed strength.
