"""Self-play-only randomized search caps and selection of training positions."""

from dataclasses import replace

import numpy as np

from .contestants import PreparedContestant

BUDGET_SEED_STREAM = 0x434150


class RandomizedBudgetPlayer:
    """Use full noisy searches for learning and cheap greedy searches otherwise.

    The budget RNG is independent of search, actions and environment outcomes.
    Full games are still observed; only full-search decisions enter replay.
    """

    def __init__(
        self,
        prepared: PreparedContestant,
        *,
        fast_iterations: int,
        full_probability: float,
        search_rng: np.random.Generator,
        budget_rng: np.random.Generator,
    ):
        self.full = prepared.player(search_rng)
        fast_config = replace(
            prepared.config,
            iterations=fast_iterations,
            temperature=0.0,
            search=replace(prepared.config.search, dirichlet_epsilon=0.0),
        )
        self.fast = replace(prepared, config=fast_config).player(search_rng)
        self.full_probability = full_probability
        self.budget_rng = budget_rng

    def get_decision(self, state, *, diagnostics=True):
        full = bool(self.budget_rng.random() < self.full_probability)
        decision = (self.full if full else self.fast).get_decision(
            state, diagnostics=diagnostics
        )
        return replace(decision, trainable=full)
