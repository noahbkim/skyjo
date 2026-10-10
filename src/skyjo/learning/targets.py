"""Round-local auxiliary labels. Observed full-game targets are never resampled."""

from __future__ import annotations

import dataclasses
import random

import numpy as np

from skyjo.engine import game as sj
from skyjo.engine import symmetry, values
from skyjo.simulation.play import GameResult

from . import batches, objectives

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


ROUND_SCORE_MIN = -48.0
ROUND_SCORE_RANGE = 336.0


def normalize_round_scores(scores: np.ndarray) -> np.ndarray:
    return ((scores.astype(np.float32) - ROUND_SCORE_MIN) / ROUND_SCORE_RANGE).astype(
        np.float32
    )


def auxiliary_targets(
    summary: TerminalSummary, state: sj.Skyjo, resolved: objectives.ResolvedObjectives
) -> dict[str, np.ndarray]:
    """Convert a round summary into labels in the decision's current-seat order."""
    shift = -sj.get_player(state)
    result = {}
    for name, _ in resolved.entries:
        if name == "round_score":
            result[name] = normalize_round_scores(np.roll(summary.scores, shift))
        elif name == "round_raw_score":
            result[name] = (np.roll(summary.raw_scores, shift) / 144.0).astype(
                np.float32
            )
        elif name == "round_doubled":
            result[name] = np.roll(summary.doubled, shift).astype(np.float32)
    return result


def build_training_batch(
    result: GameResult,
    auxiliary_objectives: objectives.ObjectiveConfig = None,
    *,
    mode: str = "observed",
    samples: int = 32,
    seed: int = 0,
    game_index: int = 0,
) -> batches.TrainingBatch:
    """Encode observed decisions and label them with the final full-game outcome.

    Optional round targets may resample the final transition. Those samples
    never change observed decisions, policy targets, or the full-game outcome.
    """
    resolved = objectives.resolve(auxiliary_objectives)
    outcome = values.skyjo_to_game_state_value(result.rounds[-1].history[-1].state)
    states = []
    rows = []
    for round_index, round_result in enumerate(result.rounds):
        summary = (
            summarize_round(
                round_result,
                mode=mode,
                samples=samples,
                seed=seed,
                game_index=game_index,
                round_index=round_index,
            )
            if resolved.entries
            else None
        )
        for state, action, probabilities in round_result.history[:-1]:
            if action is None or probabilities is None:
                raise ValueError("Observed decisions require an action and policy")
            row = {
                "value": np.roll(outcome, -sj.get_player(state)),
                "policy": symmetry.symmetrize_policy_target(state, probabilities),
            }
            if summary is not None:
                row.update(auxiliary_targets(summary, state, resolved))
            states.append(state)
            rows.append(row)
    if not states:
        raise ValueError("Training requires at least one observed decision")
    inputs = batches.states_to_batch(states)
    return batches.TrainingBatch(
        inputs.spatial_inputs,
        inputs.non_spatial_inputs,
        inputs.action_masks,
        {name: np.stack([row[name] for row in rows]) for name in rows[0]},
    )
