"""Player adapter for composable search; model construction belongs to the caller."""

import numpy as np

from skyjo.engine import game as sj
from skyjo.simulation.decisions import DiagnosticSection, PlayerDecision

from .configuration import DEFAULT_SEARCH_CONFIG, SearchConfig
from .evaluator import BoundaryEvaluator, Evaluator
from .mcts import run_mcts
from .runner import search


class SearchPlayer:
    def __init__(
        self,
        evaluator: Evaluator,
        iterations: int,
        *,
        config: SearchConfig = DEFAULT_SEARCH_CONFIG,
        temperature: float = 1.0,
        boundary_evaluator: BoundaryEvaluator | None = None,
        rng: np.random.Generator | None = None,
        components=None,
    ):
        self.evaluator = evaluator
        self.iterations = iterations
        self.config = config
        self.temperature = temperature
        self.boundary_evaluator = boundary_evaluator
        self.rng = rng if rng is not None else np.random.default_rng()
        self.components = components

    def search(self, game_state):
        return search(
            game_state,
            self.evaluator,
            self.iterations,
            config=self.config,
            temperature=self.temperature,
            boundary_evaluator=self.boundary_evaluator,
            rng=self.rng,
            components=self.components,
        )

    def get_action_probabilities(self, game_state):
        return self.search(game_state).action_probabilities

    def get_decision(self, state, *, diagnostics=True):
        result = self.search(state)
        probabilities = result.action_probabilities
        if not diagnostics:
            return PlayerDecision(probabilities, training_policy=result.training_policy)
        view = result.summary
        sections = (
            action_section(result, self.iterations, self.temperature),
            DiagnosticSection(
                "Root win estimates",
                text="Fixed seat order.",
                columns=("Seat", "Initial model", "Search"),
                rows=tuple(
                    (i + 1, float(model), float(value))
                    for i, (model, value) in enumerate(
                        zip(view.model_value, view.state_value, strict=True)
                    )
                ),
            ),
        )
        return PlayerDecision(probabilities, sections, result.training_policy)

    def run_mcts(self, game_state):
        """Compatibility entry point returning one tree, without output policies."""
        return run_mcts(
            game_state,
            self.evaluator,
            self.iterations,
            config=self.config,
            boundary_evaluator=self.boundary_evaluator,
            rng=self.rng,
            components=self.components,
        )


def action_section(result, iterations, temperature):
    """Present group-owned statistics and distinct acting/training policies once."""
    view = result.summary
    root = result.roots[0]
    config = root.context.config
    total_visits = view.visits.sum()
    columns = (
        "Action",
        "Model prior",
        "Search prior",
        "Visits",
        "Visit share",
        "Selection probability",
        "Search value",
        "Training probability",
    )
    if view.standard_error is not None:
        columns += ("Standard error",)
    rows = []
    row_actions = []
    for index, group in enumerate(view.groups.members):
        singleton = len(group) == 1
        row = (
            sj.get_action_name(group[0])
            if singleton
            else f"Group {index + 1} ({len(group)} actions)",
            float(view.model_priors[index]),
            float(view.priors[index]),
            int(view.visits[index]),
            float(view.visits[index] / total_visits),
            float(result.action_probabilities[list(group)].sum()),
            float(view.q[index]) if view.available[index] else None,
            float(result.training_policy[list(group)].sum()),
        )
        if view.standard_error is not None:
            row += (float(view.standard_error[index, view.player]),)
        rows.append(row)
        row_actions.append(group[0] if singleton else None)
        if not singleton:
            for action in group:
                row = (
                    f"  {sj.get_action_name(action)}",
                    float(root.model_prediction.policy[action]),
                    "",
                    "",
                    "",
                    float(result.action_probabilities[action]),
                    "",
                    float(result.training_policy[action]),
                )
                if view.standard_error is not None:
                    row += ("",)
                rows.append(row)
                row_actions.append(action)
    return DiagnosticSection(
        "Search actions",
        text="Visit share is raw; selection probability is the final action policy "
        "(temperature applies to visits by default). Training probability is the "
        "learning target. Search prior includes root noise. Visits and search values "
        "belong to groups; values estimate the acting seat's win probability "
        "(— = unevaluated). Replica standard errors measure search variability, "
        "not model bias.",
        metrics={
            "Iterations": iterations,
            "Temperature": temperature,
            "Symmetry merging": root.effective_merge_symmetric_actions,
            "Root exploration": config.root_strategy,
            "Interior exploration": config.interior_strategy,
            "Valuation": config.valuation,
            "Action selection": config.action_strategy,
            "Training target": config.target_strategy,
            "Replicas": config.replicas,
        },
        columns=columns,
        rows=tuple(rows),
        row_actions=tuple(row_actions),
    )
