"""Search orchestration and independent-replica aggregation.

Policies consume a completed immutable summary; neither policy knows how to run
searches. The tree runner knows nothing about concrete output policies.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from skyjo.engine import game as sj
from skyjo.simulation.decisions import validate_probabilities

from .configuration import DEFAULT_SEARCH_CONFIG, SearchConfig, resolve_components
from .evaluator import BoundaryEvaluator, Evaluator
from .mcts import DecisionStateNode, run_mcts
from .strategies import NodeView, PolicyInput, SearchComponents, frozen_array
from .symmetry import ActionGroups, safe_to_merge_actions


@dataclass(frozen=True, slots=True)
class SearchResult:
    summary: NodeView
    action_probabilities: np.ndarray
    training_policy: np.ndarray
    roots: tuple[DecisionStateNode, ...]
    context: object = None


def search(
    state: sj.Skyjo,
    evaluator: Evaluator,
    iterations: int,
    *,
    config: SearchConfig = DEFAULT_SEARCH_CONFIG,
    temperature: float = 0.0,
    boundary_evaluator: BoundaryEvaluator | None = None,
    rng: np.random.Generator | None = None,
    components: SearchComponents | None = None,
) -> SearchResult:
    """Run a total traversal budget and construct independently chosen policies."""
    if type(iterations) is not int or iterations < 1:
        raise ValueError("iterations must be positive")
    if not np.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and nonnegative")
    rng = rng if rng is not None else np.random.default_rng()
    components = components if components is not None else resolve_components(config)
    replicas = config.replicas
    budgets = [
        iterations // replicas + int(i < iterations % replicas) for i in range(replicas)
    ]
    if replicas > 1:
        merge = config.merge_symmetric_actions and safe_to_merge_actions(
            state, iterations
        )
        coverage = len(ActionGroups.from_state(state, merge=merge).members)
        if min(budgets) < coverage:
            raise ValueError(
                "Budget is too small to cover every root action group in every replica"
            )
    # Preserve the legacy stream for single-tree callers; replicas get distinct,
    # reproducible streams that never share process-global RNG state.
    generators = (
        [rng]
        if replicas == 1
        else [
            np.random.default_rng(int(rng.integers(0, 2**63))) for _ in range(replicas)
        ]
    )
    roots = tuple(
        run_mcts(
            state,
            evaluator,
            budget,
            config=config,
            boundary_evaluator=boundary_evaluator,
            rng=generator,
            components=components,
            cover_root=replicas > 1,
            symmetry_budget=iterations,
        )
        for budget, generator in zip(budgets, generators, strict=True)
    )
    views = [root.view() for root in roots]
    summary = views[0]
    contexts = tuple(
        root.context.root_session.finish(view)
        for root, view in zip(roots, views, strict=True)
    )
    context = contexts[0] if replicas == 1 else contexts
    if replicas > 1:
        values = np.stack([view.values for view in views])
        summary = replace(
            summary,
            values=frozen_array(values.mean(axis=0)),
            standard_error=frozen_array(values.std(axis=0, ddof=1) / np.sqrt(replicas)),
            visits=frozen_array(np.sum([view.visits for view in views], axis=0)),
            available=frozen_array(np.all([view.available for view in views], axis=0)),
            state_value=frozen_array(
                np.mean([view.state_value for view in views], axis=0)
            ),
            priors=frozen_array(np.mean([view.priors for view in views], axis=0)),
            visit_count=iterations,
        )
    probabilities = components.action_policy.probabilities(
        PolicyInput(summary, context), temperature
    )
    validate_probabilities(state, probabilities)
    probabilities = frozen_array(probabilities)
    replica_inputs = tuple(
        PolicyInput(view, ctx, probabilities)
        for view, ctx in zip(views, contexts, strict=True)
    )
    target = components.target_policy.probabilities(
        PolicyInput(
            summary, context, probabilities, replica_inputs if replicas > 1 else ()
        ),
        1.0,
    )
    validate_probabilities(state, target)
    return SearchResult(summary, probabilities, frozen_array(target), roots, context)
