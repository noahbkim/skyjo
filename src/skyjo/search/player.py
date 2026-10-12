"""Player adapter for composable search; model construction belongs to the caller."""

import numpy as np

from skyjo.engine import game as sj
from skyjo.simulation.decisions import DiagnosticSection, PlayerDecision, policy_section

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
        labels = tuple(
            "; ".join(sj.get_action_name(a) for a in group)
            for group in view.groups.members
        )
        sections = (
            DiagnosticSection(
                "MCTS raw visit distribution",
                text="Visit fractions are counts / total visits, before action temperature. Each row represents a whole symmetry group.",
                columns=("Actions", "Visits", "Fraction"),
                rows=tuple(
                    (label, int(n), float(n / view.visits.sum()))
                    for label, n in zip(labels, view.visits, strict=True)
                ),
            ),
            DiagnosticSection(
                "Win estimates (fixed seat order)",
                columns=("Seat", "Model", "Search"),
                rows=tuple(
                    (i + 1, float(model), float(value))
                    for i, (model, value) in enumerate(
                        zip(view.model_value, view.state_value, strict=True)
                    )
                ),
            ),
            policy_section(
                state, result.roots[0].model_prediction.policy, "Model policy"
            ),
            DiagnosticSection(
                "Search action groups",
                text="Visits and priors belong to each whole symmetry group. An unevaluated action has no value estimate. Replica standard errors measure search variability, not model bias.",
                metrics={
                    "Iterations": self.iterations,
                    "Temperature": self.temperature,
                    "Root exploration": self.config.root_strategy,
                    "Interior exploration": self.config.interior_strategy,
                    "Valuation": self.config.valuation,
                    "Action selection": self.config.action_strategy,
                    "Training target": self.config.target_strategy,
                    "Replicas": self.config.replicas,
                },
                columns=(
                    "Actions",
                    "Model prior",
                    "Search prior",
                    "Visits",
                    "Value",
                    "Standard error",
                    "Action probability",
                    "Target probability",
                ),
                rows=tuple(
                    (
                        labels[i],
                        float(view.model_priors[i]),
                        float(view.priors[i]),
                        int(view.visits[i]),
                        float(view.q[i]) if view.available[i] else None,
                        float(view.standard_error[i, view.player])
                        if view.standard_error is not None
                        else None,
                        float(probabilities[list(group)].sum()),
                        float(result.training_policy[list(group)].sum()),
                    )
                    for i, group in enumerate(view.groups.members)
                ),
            ),
            policy_section(state, probabilities, "Action probabilities"),
            policy_section(state, result.training_policy, "Training policy"),
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
