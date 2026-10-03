"""Round-local auxiliary labels. Observed full-game targets are never resampled."""

from __future__ import annotations

import dataclasses
import random

import numpy as np

from . import game as sj

AUXILIARY_SEED_STREAM = 0x415558


@dataclasses.dataclass(frozen=True)
class TerminalSummary:
    raw_scores: np.ndarray
    doubled: np.ndarray
    scores: np.ndarray


def terminal_summary(state: sj.Skyjo) -> TerminalSummary:
    raw, doubled = sj.get_round_score_components(state)
    shift = sj.get_player(state)
    return TerminalSummary(
        np.roll(raw, shift),
        np.roll(doubled, shift),
        np.roll(raw * (1 + doubled.astype(np.int16)), shift),
    )


def summarize_round(
    round_result, *, mode="observed", samples=32, seed=0, game_index=0, round_index=0
) -> TerminalSummary:
    if type(samples) is not int or samples < 1:
        raise ValueError("auxiliary_targets.samples must be a positive integer")
    if mode == "observed":
        return terminal_summary(round_result.history[-1].state)
    if mode != "resampled":
        raise ValueError("Unknown auxiliary target mode")
    decision = round_result.history[-2]
    sample_seed = int(
        np.random.SeedSequence(
            [seed, game_index, round_index, AUXILIARY_SEED_STREAM]
        ).generate_state(1, dtype=np.uint64)[0]
    )
    rng = random.Random(sample_seed)
    count = samples if sj.is_action_random(decision.action, decision.state) else 1
    totals = np.zeros((3, sj.get_player_count(decision.state)), dtype=np.float64)
    for _ in range(count):
        final = sj.apply_action(decision.state, decision.action, rng=rng)
        if not sj.get_round_over(final):
            raise ValueError(
                "Auxiliary resampling requires a round-terminal transition"
            )
        summary = terminal_summary(final)
        totals += (summary.raw_scores, summary.doubled, summary.scores)
    return TerminalSummary(*(totals / count).astype(np.float32))


CONTEXT_BUILDERS = {"terminal": summarize_round}
