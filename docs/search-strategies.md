# Composable search strategies

Search separates simulation allocation, value estimation, the action played, and
the policy used for learning. A preset assembles these choices; it does not add
another branch to the tree traversal loop. The default composition uses PUCT and
mixture backups; root exploration noise now defaults to effective action groups.

## Ownership and extension points

The runner owns game transitions, legal action groups, tree expansion, chance
outcomes, evaluator calls, and traversal budgets. Strategies receive immutable
snapshots rather than mutable tree nodes. Values always use fixed player seats;
the player taking a decision determines which component is maximized.

An exploration factory starts a fresh session for each search. The session owns
allocation state and receives a search-local random generator. Root allocation
and interior allocation are independent choices. A sequential schedule, such as
Gumbel halving, belongs in a session; it must not move simulation orchestration
into an action policy. Each snapshot includes a stable search-local `node_id`, so
an interior session can maintain separate state for each decision node.

Valuation determines how newly acquired evidence updates an estimate. Current
child estimates, fresh simulation returns, and fresh boundary samples are
different kinds of evidence. Treating successive versions of an accumulated
average as new independent samples gives incorrect values and uncertainty.
Chance estimates are probability-weighted expectations, never best-response
choices.

Action and target policies consume the completed search summary. They do not
evaluate new states, change visits, or consume search randomness. An exploration
session may return an opaque immutable context for a paired policy, while generic
consumers continue to use shared priors, visits, and values. New compatible
strategies should require an implementation and configuration registration, with
no edits to the runner, player adapter, or training pipeline.

The public contracts live in `skyjo.search.strategies`. `SearchComponents`
contains `root_exploration`, `interior_exploration`, `valuation`, `action_policy`,
and `target_policy`. An exploration implements `start(budget, rng)` returning a
session with `select(view)`, `observe(view)`, and `finish(view)`. A valuation
implements `initialize(model_value)` returning an accumulator with a `value`
property and `update(view, evidence)`. Both output policies implement
`probabilities(policy_input, temperature)` and return a legal engine-action
distribution. Configuration resolves names once; custom callers can pass
`components=` directly to `runner.search` or `mcts.run_mcts`.
Sessions that do not need completed-traversal callbacks set `observe = None`;
the runner then avoids constructing unused observation snapshots.

Symmetry grouping remains centralized: exploration visits representatives;
stochastic output distributes group mass among equivalent actions; deterministic
selection uses the representative. All policy distributions contain only legal
engine actions.

## Baselines and experimental choices

| Choice | Behavior |
| --- | --- |
| Existing PUCT | Search-value first-play estimate and existing exploration scaling |
| AlphaZero PUCT | Mean-return valuation, zero first-play value, configurable PUCT and optional root Dirichlet noise |
| Gumbel | Root sequential halving, completed-Q interior allocation, winner action, improved-policy target |
| Current mixture | One neural contribution plus visit-weighted current child estimates |
| Mean return | Average fresh simulation evidence, using the neural value before evidence arrives |
| Best response | Propagate the evaluated child's vector that maximizes the acting player's component |
| Visit action | Group probability proportional to visit count raised to inverse temperature |
| Value action | Maximize estimated value, or soften values with a temperature |
| Lower-confidence action | Penalize uncertainty across independent searches before selecting |

Acting temperature does not implicitly change learning temperature. The
AlphaZero preset learns from normalized visits at temperature 1. The Gumbel
preset learns from `softmax(log_prior + transformed_completed_Q)` without its
Gumbel perturbations. Legacy callers can retain the played-action distribution as
their learning target.

Gumbel follows the root and interior allocation rules in
[DeepMind's mctx implementation](https://github.com/google-deepmind/mctx).
The adapted sequential-halving schedule retains its [Apache 2.0 license](mctx-license.txt).
Its default candidate limit is 16, capped by legal groups and the available
budget. Completed-Q scaling uses visit offset 50, value scale 0.1, and value
rescaling. Gumbel noise defaults to scale 1; scale 0 permits deterministic
comparisons. Dirichlet noise is a separate PUCT exploration option and is off by
default for evaluation.

Use `skyjo.search.configuration.preset` to construct `legacy`, `alphazero`,
`gumbel`, or `lower_confidence` configurations, with field overrides:

```python
from skyjo.search.configuration import preset
from skyjo.search.runner import search

config = preset("gumbel", gumbel_scale=0, after_state_evaluate_all_children=True)
result = search(state, evaluator, 256, config=config, rng=rng)
action_distribution = result.action_probabilities
learning_target = result.training_policy
```

For serializable configurations, the main fields are `root_strategy`,
`interior_strategy`, `valuation`, `action_strategy`, and `target_strategy`.
`target_strategy="legacy"` keeps the historical action-distribution target.
`mcts.run_mcts` continues returning its tree for existing diagnostics and visit
extraction; `runner.search` returns both distributions and immutable statistics.

## Repeated search and budgets

Repeated search partitions one total traversal budget across independent trees.
Every replica first visits every root action group, charging that coverage to the
budget. Insufficient budgets are rejected. Work totals distinguish traversal
visits from evaluator states and boundary samples, since a single traversal may
evaluate several chance outcomes or initial boundary samples.

For each action, the lower-confidence selector uses
`mean(Q) - z * sample_standard_deviation(Q) / sqrt(replicas)`. The default is
eight replicas and `z = 1`; `z = 0` isolates selection by the replica mean.
These defaults are assembled by the `lower_confidence` preset. Other presets use
one tree unless `replicas` is set explicitly.
This uncertainty measures variation between searches under the same model. It
does not measure a shared model error or establish a probability of correctness.

## Evaluation

The tactical fixtures extracted from recording frames 53 and 112 test whether
the search gives the next player a visible clearing card. Each preserves the
recording settings and source hashes. Avoiding that giveaway is useful evidence
about those positions, not proof of stronger full-game play.

Compare allocation and valuation at equal total budgets before comparing action
selectors. Include exact-endgame fixtures and a fixed representative sample as
controls. Report action probabilities, estimated values, opponent-response
coverage, evaluator work, elapsed time, and seed variability. Keep production
defaults unchanged until these comparisons support a change.


## Adding a strategy

For a Python experiment, implement the relevant protocol and replace just that
component. For example, a new target does not need a player or training change:

```python
from dataclasses import replace
from skyjo.search.configuration import preset, resolve_components
from skyjo.search.runner import search

config = preset("alphazero")
components = replace(resolve_components(config), target_policy=MyTargetPolicy())
result = search(state, evaluator, 256, config=config, components=components, rng=rng)
```

A serializable strategy additionally gets a validated name and factory entry in
`configuration.py`. The UI, experiment parser, and player do not interpret those
names. For stateful interior allocation, use the snapshot's search-local
`node_id`; create fresh state in `start`, never on a reusable factory.

A target receives an immutable `PolicyInput`. Replica summaries are available
through `replicas`; `AveragedTarget` composes a per-tree target when averaging is
appropriate. This keeps aggregation policy outside the search runner. Mutable
trees are exposed only on the compatibility result's `roots`, never to strategy
callbacks.
