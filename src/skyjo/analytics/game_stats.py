"""Pure statistics over observed game histories; no target construction or I/O."""

from __future__ import annotations

import dataclasses
import typing
from itertools import pairwise

import numpy as np

from skyjo.engine import game as sj
from skyjo.engine import values as game_values

if typing.TYPE_CHECKING:
    from skyjo.simulation.play import GameResult, RoundResult


@dataclasses.dataclass(slots=True)
class GameStats:
    game_length: int
    outcome_state_value: np.ndarray
    scores_state_value: np.ndarray
    action_counts: np.ndarray[tuple[int], np.float32]
    action_possibility_counts: np.ndarray[tuple[int], np.float32]
    clear_count: int
    flip_count: int
    flip_possibility_count: int
    replace_face_up_count: int
    replace_face_down_count: int
    replace_possibility_count: int

    rounds: tuple[RoundStats, ...] = ()

    def to_record_dict(self) -> dict[str, typing.Any]:
        record_dict = {
            "game_length": self.game_length,
            "clear_count": self.clear_count,
        }
        for i, outcome in enumerate(self.outcome_state_value):
            record_dict[f"outcome_{i}"] = outcome
        for i, score in enumerate(self.scores_state_value):
            record_dict[f"score_{i}"] = score

        flip_rate = self.flip_count / max(self.flip_possibility_count, 1)
        replace_face_up_rate = self.replace_face_up_count / max(
            self.replace_possibility_count, 1
        )
        replace_face_down_rate = self.replace_face_down_count / max(
            self.replace_possibility_count, 1
        )
        record_dict["flip_rate"] = flip_rate
        record_dict["replace_face_up_rate"] = replace_face_up_rate
        record_dict["replace_face_down_rate"] = replace_face_down_rate

        action_rates = self.action_counts / np.maximum(
            self.action_possibility_counts, 1
        )
        for i in range(sj.MASK_SIZE):
            record_dict[f"{sj.get_action_name(i)}"] = action_rates[i]
        return record_dict


@dataclasses.dataclass(frozen=True)
class RoundStats:
    decisions: int
    turns: int
    raw_scores: tuple[int, ...]
    scores: tuple[int, ...]
    cumulative_scores: tuple[int, ...]
    ending_player: int
    cleared_columns: tuple[int, ...]
    ending_reason: str
    partial_start: bool
    action_counts: dict[str, int]
    eligible_counts: dict[str, int]
    slot_counts: tuple[int, ...]
    slot_eligible_counts: tuple[int, ...]


def analyze_round(result: RoundResult) -> RoundStats:
    history = result.history
    if len(history) < 2 or not sj.get_round_over(history[-1].state):
        raise ValueError("Expected a completed round with decisions")
    categories = (
        "initial_reveal",
        "draw",
        "take",
        "flip",
        "replace_face_up",
        "replace_face_down",
    )
    counts = dict.fromkeys(categories, 0)
    eligible = dict.fromkeys(categories, 0)
    slots = np.zeros(sj.MASK_SIZE, dtype=int)
    slot_eligible = np.zeros(sj.MASK_SIZE, dtype=int)
    ending_reason = "unknown"  # A partial history may start after the countdown.
    for entry, following in pairwise(history):
        state, action, _ = entry
        assert action is not None
        mask = sj.actions(state).astype(bool)
        hidden = sj.get_table(state)[0, :, :, sj.FINGER_HIDDEN].astype(bool).ravel()
        replace_mask = mask[sj.MASK_REPLACE :]
        opportunities = {
            "initial_reveal": bool(mask[: sj.MASK_DRAW].any()),
            "draw": bool(mask[sj.MASK_DRAW]),
            "take": bool(mask[sj.MASK_TAKE]),
            "flip": bool(mask[sj.MASK_FLIP : sj.MASK_REPLACE].any()),
            "replace_face_up": bool((replace_mask & ~hidden).any()),
            "replace_face_down": bool((replace_mask & hidden).any()),
        }
        for name, possible in opportunities.items():
            eligible[name] += int(possible)
        if action < sj.MASK_DRAW:
            category = "initial_reveal"
        elif action == sj.MASK_DRAW:
            category = "draw"
        elif action == sj.MASK_TAKE:
            category = "take"
        elif action < sj.MASK_REPLACE:
            category = "flip"
        else:
            category = (
                "replace_face_down"
                if hidden[action - sj.MASK_REPLACE]
                else "replace_face_up"
            )
        counts[category] += 1
        slots[action] += 1
        slot_eligible += mask
        if (
            sj.get_countdown(state) is None
            and sj.get_countdown(following.state) is not None
        ):
            # The actor's board is now last in the rotated table. This snapshot
            # precedes the other players' final turns and automatic reveals.
            visible = sj.get_is_visible(following.state, sj.get_player_count(state) - 1)
            ending_reason = "natural" if visible else "no_progress"
    final = history[-1].state
    raw = np.roll(
        [sj.get_score(final, i) for i in range(sj.get_player_count(final))],
        sj.get_player(final),
    )
    return RoundStats(
        decisions=len(history) - 1,
        turns=sum(counts[k] for k in ("flip", "replace_face_up", "replace_face_down")),
        raw_scores=tuple(map(int, raw)),
        scores=result.round_scores,
        cumulative_scores=result.cumulative_scores,
        ending_player=result.ending_player,
        cleared_columns=tuple(
            map(int, sj.get_fixed_perspective_cleared_columns(final).sum(axis=1))
        ),
        ending_reason=ending_reason,
        partial_start=(counts["initial_reveal"] != sj.get_player_count(final)),
        action_counts=counts,
        eligible_counts=eligible,
        slot_counts=tuple(map(int, slots)),
        slot_eligible_counts=tuple(map(int, slot_eligible)),
    )


def analyze_game(result: GameResult) -> GameStats:
    if not result.rounds:
        raise ValueError("Expected a completed full game")
    rounds = tuple(analyze_round(r) for r in result.rounds)
    final = result.rounds[-1].history[-1].state
    return GameStats(
        game_length=sum(r.decisions for r in rounds),
        outcome_state_value=game_values.skyjo_to_game_state_value(final),
        scores_state_value=sj.get_fixed_perspective_game_scores(final),
        action_counts=np.sum([r.slot_counts for r in rounds], axis=0),
        action_possibility_counts=np.sum(
            [r.slot_eligible_counts for r in rounds], axis=0
        ),
        clear_count=sum(sum(r.cleared_columns) for r in rounds),
        flip_count=sum(r.action_counts["flip"] for r in rounds),
        flip_possibility_count=sum(r.eligible_counts["flip"] for r in rounds),
        replace_face_up_count=sum(r.action_counts["replace_face_up"] for r in rounds),
        replace_face_down_count=sum(
            r.action_counts["replace_face_down"] for r in rounds
        ),
        # Preserve the legacy game-level rate denominator (all placement decisions).
        replace_possibility_count=sum(r.turns for r in rounds),
        rounds=rounds,
    )


def summarize_games(games: list[GameStats]) -> dict[str, float | int]:
    """Use rounds/player-rounds as samples and pooled opportunities for rates."""
    rounds = [r for g in games for r in g.rounds]
    complete = [r for r in rounds if not r.partial_start]
    metrics: dict[str, float | int] = {}
    records = [game.to_record_dict() for game in games]
    if records:
        metrics.update(
            {
                f"game_mean/{key}": float(np.mean([record[key] for record in records]))
                for key in records[0]
            }
        )
    distributions = {
        "round/turns": [r.turns for r in complete],
        "round/decisions": [r.decisions for r in complete],
        "round/score": [s for r in rounds for s in r.scores],
        "round/clears": [c for r in rounds for c in r.cleared_columns],
        "game/rounds": [len(g.rounds) for g in games],
    }
    for name, values in distributions.items():
        metrics[f"{name}/count"] = len(values)
        if values:
            metrics.update(
                {
                    f"{name}/{key}": float(value)
                    for key, value in (
                        ("mean", np.mean(values)),
                        ("median", np.median(values)),
                        ("p90", np.percentile(values, 90)),
                    )
                }
            )
    metrics["round/count"] = len(rounds)
    metrics["round/partial_count"] = len(rounds) - len(complete)
    known = [r for r in rounds if r.ending_reason != "unknown"]
    metrics["round/known_ending_count"] = len(known)
    if known:
        metrics["round/no_progress_rate"] = sum(
            r.ending_reason == "no_progress" for r in known
        ) / len(known)
    scores = [(s, raw) for r in rounds for s, raw in zip(r.scores, r.raw_scores)]
    if scores:
        metrics["round/score_adjustment_rate"] = sum(
            s != raw for s, raw in scores
        ) / len(scores)
    if rounds:
        for name in rounds[0].action_counts:
            count = sum(r.action_counts[name] for r in rounds)
            opportunities = sum(r.eligible_counts[name] for r in rounds)
            metrics[f"action/{name}/count"] = count
            metrics[f"action/{name}/eligible"] = opportunities
            if opportunities:
                metrics[f"action/{name}/rate"] = count / opportunities
    return metrics


def format_summary(metrics: dict[str, float | int], *, detailed: bool = False) -> str:
    """Render saved aggregates as a compact line or detailed DEBUG summary."""
    if not detailed:
        parts = [f"{metrics['round/count']} rounds"]
        for name, label in (
            ("round/turns", "turns/round"),
            ("round/score", "points/player-round"),
            ("round/clears", "clears/player-round"),
        ):
            value = metrics.get(f"{name}/mean")
            parts.append(
                f"{value:.2f} {label}" if value is not None else f"{label}: n/a"
            )
        rate = metrics.get("round/no_progress_rate")
        parts.append(
            f"{100 * rate:.1f}% no-progress" if rate is not None else "no-progress: n/a"
        )
        return " | ".join(parts)
    lines = [
        f"{metrics['round/count']} rounds ({metrics['round/partial_count']} partial)"
    ]
    for name, label in (
        ("round/turns", "Turns/round"),
        ("round/decisions", "Decisions/round"),
        ("round/score", "Points/player-round"),
        ("round/clears", "Clears/player-round"),
        ("game/rounds", "Rounds/game"),
    ):
        count = metrics[f"{name}/count"]
        summary = (
            ", ".join(
                f"{stat}={metrics[f'{name}/{stat}']:.2f}"
                for stat in ("mean", "median", "p90")
            )
            if count
            else "n/a"
        )
        lines.append(f"{label}: {summary} (n={count})")
    for name, label in (
        ("no_progress_rate", "No-progress endings"),
        ("score_adjustment_rate", "Score adjustments"),
    ):
        value = metrics.get(f"round/{name}")
        lines.append(
            f"{label}: {100 * value:.1f}%" if value is not None else f"{label}: n/a"
        )
    final = {
        key.removeprefix("game_mean/"): round(value, 3)
        for key, value in metrics.items()
        if key.startswith(("game_mean/score_", "game_mean/outcome_"))
    }
    rates = {
        key.split("/")[1]: round(value, 3)
        for key, value in metrics.items()
        if key.startswith("action/") and key.endswith("/rate")
    }
    lines.extend((f"Final-game means: {final}", f"Action rates: {rates}"))
    return "\n  ".join(lines)
