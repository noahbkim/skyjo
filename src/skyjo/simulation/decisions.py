"""Optional, serializable player diagnostics, independent of any renderer."""

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from skyjo.engine import game as sj

type Scalar = str | int | float | bool | None


@dataclass(frozen=True)
class DiagnosticSection:
    title: str
    text: str = ""
    metrics: dict[str, Scalar] = field(default_factory=dict)
    columns: tuple[str, ...] = ()
    rows: tuple[tuple[Scalar, ...], ...] = ()


@dataclass(frozen=True)
class PlayerDecision:
    probabilities: np.ndarray
    diagnostics: tuple[DiagnosticSection, ...] = ()


@runtime_checkable
class DiagnosticPlayer(Protocol):
    def get_decision(
        self, state: sj.Skyjo, *, diagnostics: bool = True
    ) -> PlayerDecision: ...


def validate_probabilities(state: sj.Skyjo, probabilities: np.ndarray) -> None:
    values = np.asarray(probabilities)
    if (
        values.shape != (sj.MASK_SIZE,)
        or not np.isfinite(values).all()
        or np.any(values < 0)
        or not np.isclose(values.sum(), 1, atol=1e-6)
        or np.any(values[~sj.actions(state).astype(bool)] != 0)
    ):
        raise ValueError(
            "Player must return a normalized distribution on legal actions"
        )


def get_decision(player, state: sj.Skyjo, *, diagnostics=True) -> PlayerDecision:
    """Support existing probability-only players without a second evaluation."""
    result = (
        player.get_decision(state, diagnostics=diagnostics)
        if isinstance(player, DiagnosticPlayer)
        else PlayerDecision(player.get_action_probabilities(state))
    )
    validate_probabilities(state, result.probabilities)
    return result


def policy_section(state, probabilities, title="Action probabilities"):
    return DiagnosticSection(
        title,
        columns=("Action", "Probability"),
        rows=tuple(
            (sj.get_action_name(int(action)), float(probabilities[action]))
            for action in np.flatnonzero(sj.actions(state))
        ),
    )
