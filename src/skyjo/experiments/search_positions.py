"""Portable, annotated public positions for tactical search diagnostics."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

from skyjo.engine import game
from skyjo.simulation import recording

FORMAT = "skyjo.search-position"


@dataclasses.dataclass(frozen=True)
class Expectation:
    kind: str  # avoid, prefer, or unscored
    actions: tuple[int, ...]
    rationale: str

    def accepts(self, action: int) -> bool | None:
        if self.kind == "unscored":
            return None
        return (action in self.actions) == (self.kind == "prefer")


@dataclasses.dataclass(frozen=True)
class Witness:
    name: str
    actions: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class Position:
    id: str
    description: str
    tags: tuple[str, ...]
    state: game.Skyjo
    expectation: Expectation
    witnesses: tuple[Witness, ...] = ()
    source: dict = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "format": FORMAT,
            "version": 1,
            "id": self.id,
            "description": self.description,
            "tags": list(self.tags),
            "state": recording.state_to_dict(self.state),
            "expectation": dataclasses.asdict(self.expectation),
            "witnesses": [dataclasses.asdict(line) for line in self.witnesses],
            "source": self.source,
        }


def witness_states(position: Position, line: Witness):
    """Return pre-action states and exact outcomes of the final ordinary action.

    Prefixes must be deterministic; no arbitrary sampled branch is frozen into
    a tactical line. Round-ending multi-card reveals need a separate oracle.
    """
    if not line.actions:
        raise ValueError("Witness must contain at least one action")
    state = position.state
    prefix = []
    for index, action in enumerate(line.actions):
        _legal_actions(state, (action,))
        if game.get_round_about_to_end(state):
            raise ValueError("Witness cannot cross a round boundary")
        prefix.append(state)
        if index == len(line.actions) - 1:
            return prefix, game.chance_outcomes(state, action)
        if game.is_action_random(action, state):
            raise ValueError("Witness prefix must be deterministic")
        state = game.apply_action(state, action)
    raise AssertionError("Empty witness")


def _legal_actions(state, actions):
    mask = game.actions(state)
    if len(set(actions)) != len(actions) or any(
        type(a) is not int or not 0 <= a < game.MASK_SIZE or not mask[a]
        for a in actions
    ):
        raise ValueError("Annotations must name unique legal action indices")


def from_dict(data: dict) -> Position:
    if data["format"] != FORMAT or data["version"] != 1:
        raise ValueError("Unsupported search position format/version")
    state = recording.state_from_dict(data["state"], data["state"]["players"])
    if game.get_round_over(state):
        raise ValueError("A search position must be an active decision")
    expected = data["expectation"]
    expectation = Expectation(
        expected["kind"], tuple(expected["actions"]), expected["rationale"]
    )
    if expectation.kind not in {"avoid", "prefer", "unscored"}:
        raise ValueError("Expectation must be avoid, prefer, or unscored")
    if bool(expectation.actions) != (expectation.kind != "unscored"):
        raise ValueError("Only scored positions need expected actions")
    _legal_actions(state, expectation.actions)
    position = Position(
        data["id"],
        data["description"],
        tuple(data["tags"]),
        state,
        expectation,
        tuple(Witness(w["name"], tuple(w["actions"])) for w in data["witnesses"]),
        data["source"],
    )
    if not position.id or not position.description or not expectation.rationale:
        raise ValueError("Position needs an id, description, and expectation rationale")
    names = [w.name for w in position.witnesses]
    if len(set(names)) != len(names):
        raise ValueError("Witness names must be unique")
    for line in position.witnesses:
        witness_states(position, line)
    return position


def load_suite(folder: Path) -> list[Position]:
    """Adding a JSON file adds a case; unrelated ordering never changes its seed."""
    positions = [
        from_dict(json.loads(p.read_text())) for p in sorted(folder.glob("*.json"))
    ]
    ids = [p.id for p in positions]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Suite must contain positions with unique IDs")
    return positions


def export_position(
    source: Path,
    frame: int,
    destination: Path,
    *,
    id: str,
    description: str,
    expectation: Expectation,
    tags: tuple[str, ...] = (),
    witnesses: tuple[Witness, ...] = (),
) -> Path:
    """Extract a self-contained case without copying model files or later frames."""
    saved = recording.load(source)
    if not 0 <= frame < len(saved.frames):
        raise ValueError("Frame is outside recording")
    snapshot = saved.frames[frame]
    position = Position(
        id,
        description,
        tags,
        snapshot.state,
        expectation,
        witnesses,
        {
            "recording": source.name,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "frame": frame,
            "round_number": snapshot.round_number,
            "recorded_action": snapshot.action,
        },
    )
    from_dict(position.to_dict())
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as stream:
        json.dump(position.to_dict(), stream, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    return destination
