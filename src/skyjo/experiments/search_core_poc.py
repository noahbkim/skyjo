"""Experimental search storage alternatives; never enabled by production callers.

The array variant is limited to built-in PUCT/current-mixture search. Its private
selection view borrows mutable storage; public ``view()`` still returns an owned,
immutable snapshot. Use one context per process, never concurrent contexts.
"""

from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from skyjo.engine import game
from skyjo.search import mcts
from skyjo.search.strategies import PUCT, CurrentMixture, NodeView, frozen_array

VARIANTS = ("baseline", "arrays", "lazy", "combined")
_Decision = mcts.DecisionStateNode


@dataclass(slots=True)
class _Unvisited:
    """Zero statistics for a legal edge whose child has not been constructed."""

    state_value: np.ndarray
    visit_count: int = 0
    has_value_estimate: bool = False
    is_expanded: bool = False


class _LazyChildren(dict):
    __slots__ = ("owner",)

    def __init__(self, owner, children):
        super().__init__(children)
        self.owner = owner

    def __getitem__(self, action):
        child = super().__getitem__(action)
        if isinstance(child, _Unvisited):
            child = _Decision.create_child_node(self.owner, action)
            self[action] = child
        return child


class _SearchArrays:
    """Private, ephemeral-to-callers PUCT inputs, updated on completed backups."""

    __slots__ = (
        "actions",
        "available",
        "index",
        "node_id",
        "player",
        "priors",
        "state_value",
        "values",
        "visit_count",
        "visits",
    )

    def __init__(self, node):
        self.actions = tuple(node.children)
        self.index = {action: i for i, action in enumerate(self.actions)}
        self.priors = node.action_priors[list(self.actions)].copy()
        self.visits = np.zeros(len(self.actions), dtype=np.int64)
        # Match np.array([child.state_value, ...]) precision exactly. Decision
        # backups promote to float64; chance/boundary values remain float32.
        self.values = np.zeros((len(self.actions), node.state.players), np.float32)
        self.available = np.zeros(len(self.actions), dtype=bool)
        self.player = game.get_player(node.state)
        self.state_value = node.state_value
        self.visit_count = node.visit_count
        self.node_id = node._node_id

    @property
    def q(self):
        return self.values[:, self.player]

    def update_child(self, child):
        i = self.index[child.action]
        value = child.state_value
        dtype = np.result_type(self.values.dtype, value.dtype)
        if dtype != self.values.dtype:
            self.values = self.values.astype(dtype)
        self.visits[i] = child.visit_count
        self.values[i] = value
        self.available[i] = child.has_value_estimate


class _ExperimentalDecision(_Decision):
    __slots__ = ("_data", "_defer", "_empty")
    use_arrays = False
    use_lazy = False

    def __post_init__(self):
        super().__post_init__()
        self._defer = False
        self._empty = (
            _Unvisited(np.zeros(self.state.players, np.float32))
            if self.use_lazy
            else None
        )
        self._data = None

    def create_child_node(self, action):
        return self._empty if self._defer else super().create_child_node(action)

    def expand(self, prediction):
        if self.use_arrays:
            components = self.context.components
            assert type(components.root_exploration) is PUCT
            assert type(components.interior_exploration) is PUCT
            assert type(components.valuation) is CurrentMixture
        self._defer = self.use_lazy
        try:
            super().expand(prediction)
        finally:
            self._defer = False
        if self.use_lazy:
            self.children = _LazyChildren(self, self.children)
        if self.use_arrays:
            self._data = _SearchArrays(self)

    def refresh_root_noise(self):
        super().refresh_root_noise()
        if self.use_arrays:
            self._data.priors[:] = self.action_priors[list(self._data.actions)]

    def select_child(self, **kwargs):
        if not self.use_arrays:
            return super().select_child(**kwargs)
        data = self._data
        data.state_value = self.state_value
        data.visit_count = self.visit_count
        session = (
            self.context.root_session
            if self.parent is None
            else self.context.interior_session
        )
        # Built-in PUCT sessions (and the benchmark's pending-visit adapter)
        # consume these arrays synchronously and do not retain the view.
        return self.children[session.select(data)]

    def refresh_value(self, evidence=None):
        if not self.use_arrays:
            return super().refresh_value(evidence)
        data = self._data
        # Preserve both operation order and dtype of _MixtureValue.update.
        self._state_value = (
            self.model_prediction.value
            + (data.visits[:, None] * data.values).sum(axis=0)
        ) / (1 + data.visits.sum())
        self._valuation_state.value = self._state_value

    def view(self):
        if not self.use_lazy:
            return super().view()
        # Read unvisited zero edges without materializing them. As in the
        # production API, every snapshot owns immutable copies of its arrays.
        actions = tuple(self.children)
        children = tuple(self.children.values())
        return NodeView(
            self.action_groups,
            actions,
            frozen_array([self.action_priors[a] for a in actions]),
            frozen_array([self.model_action_priors[a] for a in actions]),
            frozen_array([child.visit_count for child in children]),
            frozen_array([child.state_value for child in children]),
            frozen_array([child.has_value_estimate for child in children]),
            game.get_player(self.state),
            frozen_array(self.state_value),
            frozen_array(self.model_prediction.value),
            self.visit_count,
            self.parent is None,
            node_id=self._node_id,
            base_priors=frozen_array([self.base_action_priors[a] for a in actions]),
        )


class ArrayDecision(_ExperimentalDecision):
    __slots__ = ()
    use_arrays = True


class LazyDecision(_ExperimentalDecision):
    __slots__ = ()
    use_lazy = True


class CombinedDecision(_ExperimentalDecision):
    __slots__ = ()
    use_arrays = True
    use_lazy = True


def _array_backpropagate(path, evidence=None):
    for node in reversed(path):
        node.visit_count += 1
        if isinstance(node, _Decision):
            node.refresh_value(evidence)
        elif isinstance(node, mcts.AfterStateNode):
            node.refresh_value()
        parent = node.parent
        if isinstance(parent, _ExperimentalDecision) and parent.use_arrays:
            parent._data.update_child(node)


@contextmanager
def search_variant(variant):
    """Install a process-local prototype, restoring production even on failure."""
    variants = {
        "baseline": _Decision,
        "arrays": ArrayDecision,
        "lazy": LazyDecision,
        "combined": CombinedDecision,
    }
    if variant not in variants:
        raise ValueError(variant)
    assert mcts.DecisionStateNode is _Decision, "Nested variant contexts unsupported"
    original_backup = mcts.backpropagate
    try:
        mcts.DecisionStateNode = variants[variant]
        if variant in {"arrays", "combined"}:
            mcts.backpropagate = _array_backpropagate
        yield
    finally:
        mcts.DecisionStateNode = _Decision
        mcts.backpropagate = original_backup
