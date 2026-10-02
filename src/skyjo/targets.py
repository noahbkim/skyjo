"""Learner-side construction of cached labels from completed self-play histories."""

from __future__ import annotations

import dataclasses
import random
from collections.abc import Callable
from typing import Any

import numpy as np

from . import game as sj
from . import objectives, play, skynet


@dataclasses.dataclass
class TerminalSummary:
    outcome: np.ndarray
    value: np.ndarray
    scores: np.ndarray
    cleared_columns: np.ndarray
    raw_scores: np.ndarray
    doubled: np.ndarray


def summarize_terminal(
    state: sj.Skyjo,
    action: sj.SkyjoAction,
    simulations: int,
    *,
    rng: sj.Random | None = None,
) -> TerminalSummary:
    if isinstance(simulations, bool) or not isinstance(simulations, int) or simulations < 1:
        raise ValueError("terminal_rollouts must be a positive integer")
    rng = rng if rng is not None else random.Random()
    players = sj.get_player_count(state)
    summary = TerminalSummary(
        *[np.zeros(players, dtype=np.float64) for _ in range(3)],
        np.zeros(players * sj.COLUMN_COUNT, dtype=np.float64),
        *[np.zeros(players, dtype=np.float64) for _ in range(2)],
    )
    count = simulations if sj.is_action_random(action, state) else 1
    for _ in range(count):
        final = sj.apply_action(state, action, rng=rng)
        if not sj.get_game_over(final):
            raise ValueError("Target construction requires a terminal transition")
        raw, doubled = sj.get_round_score_components(final)
        shift = sj.get_player(final)
        raw = np.roll(raw, shift)
        doubled = np.roll(doubled, shift)
        scores = raw * (1 + doubled.astype(np.int16))
        summary.outcome[sj.get_fixed_perspective_winner(final)] += 1
        summary.value += skynet.scores_to_score_differential_value(scores)
        summary.scores += scores
        summary.cleared_columns += sj.get_fixed_perspective_cleared_columns(
            final
        ).reshape(-1)
        summary.raw_scores += raw
        summary.doubled += doubled
    return TerminalSummary(
        *[
            (getattr(summary, field.name) / count).astype(np.float32)
            for field in dataclasses.fields(summary)
        ]
    )


def terminal_context(
    history: play.GameHistory, terminal_rollouts: int, rng: random.Random
) -> TerminalSummary:
    last_decision = history[-2]
    return summarize_terminal(
        last_decision.state, last_decision.action, terminal_rollouts, rng=rng
    )


# New objectives can register another history-derived context here. Dependencies
# are built once per history and shared by every objective that requests them.
CONTEXT_BUILDERS: dict[str, Callable[..., Any]] = {"terminal": terminal_context}


def build_targets(
    game_history: play.GameHistory,
    auxiliary_objectives: objectives.ObjectiveConfig = None,
    *,
    terminal_rollouts: int = 1,
    rng: random.Random | None = None,
) -> play.GeneratedEpisode:
    """Build labels once, before replay insertion; never performs policy search."""
    if len(game_history) < 2 or not sj.get_game_over(game_history[-1].state):
        raise ValueError("Target construction requires a completed history")
    if game_history[-2].action is None:
        raise ValueError("Completed history is missing its final action")
    resolved = objectives.resolve(auxiliary_objectives)
    rng = rng if rng is not None else random.Random()
    # Each context has a stream keyed by its dependency name. Adding objectives
    # cannot perturb core targets or consume extra draws from the learner's RNG.
    seed = rng.getrandbits(128)
    dependencies = {"terminal"} | {o.dependency for _, _, o in resolved.entries}
    contexts = {
        name: CONTEXT_BUILDERS[name](
            game_history, terminal_rollouts, random.Random(f"{seed}:{name}")
        )
        for name in sorted(dependencies)
    }
    summary = contexts["terminal"]
    training_data = []
    action_counts = np.zeros(sj.MASK_SIZE, dtype=np.float32)
    action_possibility_counts = np.zeros(sj.MASK_SIZE, dtype=np.float32)
    flip_count, flip_possibility_count = 0, 0
    replace_face_up_count, replace_face_down_count, replace_possibility_count = 0, 0, 0
    for game_state, action, mcts_probs in game_history[:-1]:
        action_mask = sj.actions(game_state).astype(np.float32)
        assert action is not None, "expected non-terminal action"
        assert mcts_probs is not None, "expected non-terminal action probabilities"
        training_data.append(
            play.GameDataPoint(
                game_state,  # game
                action,  # realized action
                {
                    "value": np.roll(summary.value, -sj.get_player(game_state)),
                    "policy": mcts_probs,
                    **{
                        name: objective.target(contexts[objective.dependency], game_state)
                        for name, _, objective in resolved.entries
                    },
                },
            )
        )
        action_counts[action] += 1
        action_possibility_counts += action_mask
        if action < sj.MASK_FLIP:
            continue

        # Always possible to replace a face up card or face down card
        replace_possibility_count += 1
        if np.any(action_mask[sj.MASK_FLIP : sj.MASK_FLIP + sj.FINGER_COUNT]):
            flip_possibility_count += 1

        if sj.MASK_FLIP <= action < sj.MASK_REPLACE:
            flip_count += 1
        else:
            row, col = divmod(action - sj.MASK_REPLACE, sj.COLUMN_COUNT)
            if sj.get_finger(game_state, row, col, 0) == sj.FINGER_HIDDEN:
                replace_face_down_count += 1
            else:
                replace_face_up_count += 1

    cleared_cards = (
        sj.get_table(game_history[-2].state)[:, :, :, sj.FINGER_CLEARED].sum()
    )
    assert cleared_cards % 3 == 0, (
        f"Cleared cards is not divisible by 3: {cleared_cards}"
    )
    clear_count = cleared_cards // 3
    game_stats = play.GameStats(
        game_length=len(game_history) - 1,
        outcome_state_value=summary.outcome,
        scores_state_value=summary.scores,
        action_counts=action_counts,
        action_possibility_counts=action_possibility_counts,
        clear_count=clear_count,
        flip_count=flip_count,
        flip_possibility_count=flip_possibility_count,
        replace_face_up_count=replace_face_up_count,
        replace_face_down_count=replace_face_down_count,
        replace_possibility_count=replace_possibility_count,
    )
    return training_data, game_stats
