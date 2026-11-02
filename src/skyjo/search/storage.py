"""Private edge storage for built-in PUCT/current-mixture search."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from skyjo.engine import game


@dataclass(frozen=True, slots=True)
class _Unvisited:
    state_value: np.ndarray
    visit_count: int = 0
    has_value_estimate: bool = False


class LazyChildren(Mapping):
    """All legal edges are keys; accessing a child materializes that child.

    Public items()/values() expose real nodes, including unvisited ones. Search
    snapshots use statistics() so diagnostics do not allocate child states.
    """

    __slots__ = ("_empty", "_nodes", "_owner")

    def __init__(self, owner, actions):
        self._owner = owner
        self._nodes = dict.fromkeys(actions)
        self._empty = _Unvisited(np.zeros(owner.state.players, np.float32))

    def __iter__(self):
        return iter(self._nodes)

    def __len__(self):
        return len(self._nodes)

    def __contains__(self, action):
        return action in self._nodes

    def __getitem__(self, action):
        child = self._nodes[action]
        if child is None:
            child = self._owner.create_child_node(action)
            self._nodes[action] = child
        return child

    def statistics(self):
        return tuple(self._empty if n is None else n for n in self._nodes.values())


class EdgeStatistics:
    """Mutable inputs borrowed only by the synchronous built-in PUCT session."""

    __slots__ = (
        "actions",
        "available",
        "index",
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
        self.visits = np.zeros(len(self.actions), np.int64)
        self.values = np.zeros((len(self.actions), node.state.players), np.float32)
        self.available = np.zeros(len(self.actions), bool)
        self.player = game.get_player(node.state)
        self.state_value = node.state_value
        self.visit_count = node.visit_count

    @property
    def q(self):
        return self.values[:, self.player]

    def update_child(self, child):
        index = self.index[child.action]
        value = child.state_value
        # Match the eager snapshot's precision: decision mixtures promote to
        # float64, while chance and boundary estimates stay float32.
        dtype = np.result_type(self.values.dtype, value.dtype)
        if dtype != self.values.dtype:
            self.values = self.values.astype(dtype)
        self.visits[index] = child.visit_count
        self.values[index] = value
        self.available[index] = child.has_value_estimate
