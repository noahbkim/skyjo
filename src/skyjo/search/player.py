"""Player adapter for search; model construction belongs to the caller."""

import numpy as np

from skyjo.engine import game as sj
from skyjo.simulation.decisions import DiagnosticSection, PlayerDecision, policy_section

from .evaluator import BoundaryEvaluator, Evaluator
from .mcts import DEFAULT_SEARCH_CONFIG, SearchConfig, run_mcts


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
    ):
        self.evaluator = evaluator
        self.iterations = iterations
        self.config = config
        self.temperature = temperature
        self.boundary_evaluator = boundary_evaluator
        self.rng = rng if rng is not None else np.random.default_rng()

    def get_action_probabilities(self, game_state):
        return self.run_mcts(game_state).policy_targets(self.temperature)

    def get_decision(self, state, *, diagnostics=True):
        root = self.run_mcts(state)
        probabilities = root.policy_targets(self.temperature)
        if not diagnostics:
            return PlayerDecision(probabilities)
        total_visits = sum(child.visit_count for child in root.children.values())
        sections = (
            DiagnosticSection(
                "MCTS raw visit distribution",
                text="Visit fractions are counts / total visits, before action temperature. "
                "Each row represents a whole symmetry group.",
                columns=("Actions", "Visits", "Fraction"),
                rows=tuple(
                    (
                        "; ".join(sj.get_action_name(a) for a in group),
                        root.children[group[0]].visit_count,
                        root.children[group[0]].visit_count / total_visits,
                    )
                    for group in root.action_groups.members
                ),
            ),
            DiagnosticSection(
                "Win estimates (fixed seat order)",
                columns=("Seat", "Model", "Search"),
                rows=tuple(
                    (i + 1, float(model), float(search))
                    for i, (model, search) in enumerate(
                        zip(root.model_prediction.value, root.state_value, strict=True)
                    )
                ),
            ),
            policy_section(state, root.model_prediction.policy, "Model policy"),
            DiagnosticSection(
                "Search action groups",
                text="Visits and priors belong to each whole symmetry group. "
                "An unevaluated action has no value estimate.",
                metrics={
                    "Iterations": self.iterations,
                    "Temperature": self.temperature,
                },
                columns=("Actions", "Model prior", "Search prior", "Visits", "Value"),
                rows=tuple(
                    (
                        "; ".join(sj.get_action_name(a) for a in group),
                        float(root.model_action_priors[group[0]]),
                        float(root.action_priors[group[0]]),
                        root.children[group[0]].visit_count,
                        float(root.children[group[0]].state_value[sj.get_player(state)])
                        if root.children[group[0]].has_value_estimate
                        else None,
                    )
                    for group in root.action_groups.members
                ),
            ),
        )
        return PlayerDecision(probabilities, sections)

    def run_mcts(self, game_state):
        return run_mcts(
            game_state,
            self.evaluator,
            self.iterations,
            config=self.config,
            boundary_evaluator=self.boundary_evaluator,
            rng=self.rng,
        )
