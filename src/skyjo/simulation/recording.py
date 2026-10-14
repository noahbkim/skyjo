"""Versioned visual game recordings. Playback requires neither models nor RNGs."""

from __future__ import annotations

import dataclasses
import datetime
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from skyjo.engine import game as sj

from .decisions import DiagnosticSection, PlayerDecision, validate_decision

FORMAT = "skyjo.game-recording"
VERSION = 2


@dataclasses.dataclass(frozen=True)
class Frame:
    state: sj.Skyjo
    round_number: int
    event: str = "decision"
    action: int | None = None
    decision: PlayerDecision | None = None
    round_scores: tuple[int, ...] | None = None
    cumulative_scores: tuple[int, ...] | None = None
    winners: tuple[int, ...] | None = None


def make_frame(state: sj.Skyjo, round_number: int) -> Frame:
    if not sj.get_round_over(state):
        return Frame(state, round_number)
    scores = tuple(map(int, sj.get_fixed_perspective_game_scores(state)))
    complete = sj.get_game_over(state)
    return Frame(
        state,
        round_number,
        "game_end" if complete else "round_end",
        round_scores=tuple(map(int, sj.get_fixed_perspective_round_scores(state))),
        cumulative_scores=scores,
        winners=tuple(i for i, score in enumerate(scores) if score == min(scores))
        if complete
        else None,
    )


@dataclasses.dataclass
class Recording:
    players: list[dict]
    seed: int
    frames: list[Frame]
    created_at: str = dataclasses.field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC).isoformat()
    )

    @property
    def complete(self):
        return self.frames[-1].event == "game_end"


class RecordingError(ValueError):
    """An unsupported or malformed game recording."""


def _state_dict(state):
    return {
        field.name: value.tolist() if isinstance(value, np.ndarray) else value
        for field in dataclasses.fields(state)
        for value in (getattr(state, field.name),)
    }


def to_dict(recording: Recording) -> dict:
    frames = []
    for frame in recording.frames:
        frames.append(
            {
                "state": _state_dict(frame.state),
                "round_number": frame.round_number,
                "event": frame.event,
                "action": frame.action,
                "probabilities": frame.decision.probabilities.tolist()
                if frame.decision
                else None,
                "training_policy": frame.decision.training_policy.tolist()
                if frame.decision and frame.decision.training_policy is not None
                else None,
                "diagnostics": [
                    dataclasses.asdict(s) for s in frame.decision.diagnostics
                ]
                if frame.decision
                else [],
                **{
                    key: list(value) if value is not None else None
                    for key in ("round_scores", "cumulative_scores", "winners")
                    for value in (getattr(frame, key),)
                },
            }
        )
    return {
        "format": FORMAT,
        "version": VERSION,
        "created_at": recording.created_at,
        "players": recording.players,
        "seed": recording.seed,
        "complete": recording.complete,
        "frames": frames,
    }


def save(recording: Recording, path: Path) -> Path:
    """Replace atomically, leaving the previous file intact if serialization fails."""
    payload = json.dumps(to_dict(recording), allow_nan=False, separators=(",", ":"))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def _integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise RecordingError("Expected an integer within range")
    return value


def _state(data, players):
    arrays = {}
    for name, shape in (
        ("game", (sj.GAME_SIZE,)),
        ("table", (sj.PLAYER_COUNT, sj.ROW_COUNT, sj.COLUMN_COUNT, sj.FINGER_SIZE)),
        ("deck", (sj.CARD_SIZE,)),
    ):
        values = np.asarray(data[name])
        if (
            values.shape != shape
            or values.dtype.kind not in "iu"
            or np.any(values < -32768)
            or np.any(values > 32767)
        ):
            raise RecordingError(f"Invalid {name} state array")
        arrays[name] = values.astype(np.int16)
    if type(data["players"]) is not int or data["players"] != players:
        raise RecordingError("State and recording player counts differ")
    countdown = data["countdown"]
    pending = data["pending_card"]
    if countdown is not None:
        _integer(countdown)
    if pending is not None and not 0 <= _integer(pending) < sj.CARD_SIZE:
        raise RecordingError("Invalid pending card")
    state = sj.Skyjo(
        **arrays,
        players=players,
        turn=_integer(data["turn"]),
        countdown=countdown,
        pending_card=pending,
    )
    sj.validate(state)
    return state


def _section(data):
    section = DiagnosticSection(
        title=data["title"],
        text=data.get("text", ""),
        metrics=data.get("metrics", {}),
        columns=tuple(data.get("columns", ())),
        rows=tuple(tuple(row) for row in data.get("rows", ())),
    )
    if (
        not isinstance(section.title, str)
        or not isinstance(section.text, str)
        or not isinstance(section.metrics, dict)
        or not all(isinstance(k, str) for k in section.metrics)
        or not all(isinstance(k, str) for k in section.columns)
        or any(len(row) != len(section.columns) for row in section.rows)
    ):
        raise RecordingError("Invalid diagnostic section")
    for value in [*section.metrics.values(), *(v for r in section.rows for v in r)]:
        if value is not None and type(value) not in (str, int, float, bool):
            raise RecordingError("Diagnostic values must be JSON scalars")
        if isinstance(value, float) and not np.isfinite(value):
            raise RecordingError("Diagnostic values must be finite")
    return section


def from_dict(data: dict) -> Recording:
    """Read snapshots, not regenerated transitions. Unknown versions fail explicitly."""
    try:
        if data["format"] != FORMAT or data["version"] not in (1, VERSION):
            raise RecordingError("Unsupported game recording format or version")
        players = data["players"]
        if not 2 <= len(players) <= sj.PLAYER_COUNT or any(
            not isinstance(player, dict) or not isinstance(player.get("name"), str)
            for player in players
        ):
            raise RecordingError("Expected 2–8 named players")
        frames = []
        for raw in data["frames"]:
            state = _state(raw["state"], len(players))
            frame = make_frame(state, _integer(raw["round_number"], 1))
            if raw["event"] != frame.event:
                raise RecordingError("Frame event does not match state")
            for key in ("round_scores", "cumulative_scores", "winners"):
                expected = getattr(frame, key)
                if raw.get(key) != (list(expected) if expected is not None else None):
                    raise RecordingError(f"Invalid {key}")
            action = raw["action"]
            probabilities = raw["probabilities"]
            training_policy = raw.get("training_policy") if data["version"] >= 2 else None
            if (action is None) != (probabilities is None):
                raise RecordingError("Actions and probabilities must occur together")
            if action is not None:
                _integer(action)
                if (
                    frame.event != "decision"
                    or action >= sj.MASK_SIZE
                    or not sj.actions(state)[action]
                ):
                    raise RecordingError("Illegal recorded action")
                probabilities = np.asarray(probabilities, dtype=np.float32)
                decision = PlayerDecision(
                    probabilities,
                    tuple(_section(s) for s in raw.get("diagnostics", ())),
                    np.asarray(training_policy, dtype=np.float32)
                    if training_policy is not None
                    else None,
                )
                validate_decision(state, decision)
                if probabilities[action] <= 0:
                    raise RecordingError("Recorded action has zero probability")
                frame = dataclasses.replace(
                    frame,
                    action=action,
                    decision=decision,
                )
            elif raw.get("diagnostics") or training_policy is not None:
                raise RecordingError("Diagnostics and targets require a recorded decision")
            if frames:
                previous = frames[-1]
                expected_round = previous.round_number + (previous.event == "round_end")
                if previous.event == "game_end" or frame.round_number != expected_round:
                    raise RecordingError("Invalid round sequence")
                if previous.event == "decision" and previous.action is None:
                    raise RecordingError("Missing action before next snapshot")
            elif frame.round_number != 1:
                raise RecordingError("Recordings start at round one")
            frames.append(frame)
        if not frames or frames[-1].action is not None:
            raise RecordingError("Recording must finish with a resulting snapshot")
        result = Recording(players, _integer(data["seed"]), frames, data["created_at"])
        if data["complete"] is not result.complete:
            raise RecordingError("Completion flag does not match final snapshot")
        return result
    except (
        KeyError,
        TypeError,
        ValueError,
        AssertionError,
        IndexError,
        AttributeError,
    ) as error:
        raise RecordingError(f"Invalid game recording: {error}") from error


def load(path: Path) -> Recording:
    try:
        return from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
    except (OSError, ValueError) as error:
        raise RecordingError(f"Could not open {path}: {error}") from error
