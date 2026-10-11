"""Player adapter for search; model construction belongs to the caller."""

import numpy as np

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

    def run_mcts(self, game_state):
        return run_mcts(
            game_state,
            self.evaluator,
            self.iterations,
            config=self.config,
            boundary_evaluator=self.boundary_evaluator,
            rng=self.rng,
        )
