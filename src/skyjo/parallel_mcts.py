"""Batched scheduling adapter for the shared MCTS search core.

All tree rules and node implementations live in :mod:`skyjo.mcts`.  Predictor
clients remain responsible for coalescing queued evaluations; this module only
retains the historical batched configuration and call surface.
"""

from __future__ import annotations

import dataclasses
import typing

from . import config, mcts, predictor, skynet
from . import game as sj


@dataclasses.dataclass(slots=True)
class BatchedMCTSConfig(config.Config):
    iterations: int
    dirichlet_epsilon: float
    after_state_evaluate_all_children: bool
    batched_leaf_count: int
    virtual_loss: float
    c_puct: float = 1.5
    fpu_reduction: float = 0.0
    score_utility_weight: float = 0.0


# Compatibility names now point at the single tested node implementation.
DecisionStateNode = mcts.DecisionStateNode
AfterStateNode = mcts.AfterStateNode
RoundBoundaryNode = mcts.RoundBoundaryNode
MCTSNode = mcts.MCTSNode
Config = BatchedMCTSConfig
ucb_score = mcts.ucb_score
find_leaf = mcts.find_leaf
backpropagate = mcts.backpropagate


def run_mcts(
    game_state: sj.Skyjo,
    predictor_client: predictor.AbstractPredictorClient,
    iterations: int,
    *,
    dirichlet_epsilon: float = 0.0,
    after_state_evaluate_all_children: bool = False,
    batched_leaf_count: int = 1,
    virtual_loss: float = 0.5,
    c_puct: float = 1.5,
    fpu_reduction: float = 0.0,
    score_utility_weight: float = 0.0,
    root_node: MCTSNode | None = None,
) -> MCTSNode:
    """Run shared tree semantics while batching pending leaf evaluations."""
    if batched_leaf_count < 1:
        raise ValueError("batched_leaf_count must be at least one")
    if virtual_loss < 0:
        raise ValueError("virtual_loss cannot be negative")
    root = mcts.run_mcts(
        game_state,
        predictor_client,
        0,
        dirichlet_epsilon=dirichlet_epsilon,
        after_state_evaluate_all_children=after_state_evaluate_all_children,
        c_puct=c_puct,
        fpu_reduction=fpu_reduction,
        score_utility_weight=score_utility_weight,
        root_node=root_node,
    )

    completed = 0
    while completed < iterations:
        batch_paths: list[list[MCTSNode]] = []
        seen_leaves: set[int] = set()
        for _ in range(min(batched_leaf_count, iterations - completed)):
            path = mcts.find_leaf(
                root,
                update_after_state_child_weights=not after_state_evaluate_all_children,
            )
            leaf_identity = id(path[-1])
            if leaf_identity in seen_leaves:
                break
            seen_leaves.add(leaf_identity)
            for node in path[1:]:
                if hasattr(node, "virtual_loss_total"):
                    node.virtual_loss_total += virtual_loss
                else:
                    node.virtual_loss += virtual_loss
            batch_paths.append(path)

        pending: dict[int, tuple[str, list[MCTSNode], MCTSNode | None]] = {}
        after_paths: dict[int, tuple[list[MCTSNode], AfterStateNode]] = {}
        backup_values: dict[int, skynet.StateValue] = {}
        for path in batch_paths:
            leaf = path[-1]
            if isinstance(leaf, mcts.RoundBoundaryNode):
                next_round = leaf.prepare()
                if next_round is not None:
                    prediction_id = predictor_client.put(next_round)
                    pending[prediction_id] = ("boundary", path, None)
                else:
                    backup_values[id(path)] = leaf.state_value.copy()
            elif isinstance(leaf, mcts.DecisionStateNode):
                prediction_id = predictor_client.put(leaf.state)
                pending[prediction_id] = ("decision", path, None)
            else:
                leaf.discover(
                    discover_all_children=after_state_evaluate_all_children
                )
                after_paths[id(leaf)] = (path, leaf)
                for child in leaf.children.values():
                    prediction_id = predictor_client.put(child.state)
                    pending[prediction_id] = ("after", path, child)

        if pending:
            predictor_client.send()
            results = predictor_client.get_all()
            if len(results) != len(pending):
                raise RuntimeError(
                    f"predictor returned {len(results)} of {len(pending)} batched results"
                )
            for prediction_id, prediction in results:
                kind, path, child = pending[prediction_id]
                if kind == "boundary":
                    leaf = typing.cast(mcts.RoundBoundaryNode, path[-1])
                    leaf.set_prediction(prediction)
                elif kind == "decision":
                    leaf = typing.cast(mcts.DecisionStateNode, path[-1])
                    leaf.expand(prediction)
                else:
                    decision_child = typing.cast(mcts.DecisionStateNode, child)
                    decision_child.expand(prediction)

        for path, after_leaf in after_paths.values():
            after_leaf.expand()

        for path in batch_paths:
            for node in path[1:]:
                if hasattr(node, "virtual_loss_total"):
                    node.virtual_loss_total -= virtual_loss
                else:
                    node.virtual_loss -= virtual_loss
            mcts.backpropagate(
                path,
                backup_values.get(id(path), path[-1].state_value.copy()),
            )
            completed += 1

        if not batch_paths:
            raise RuntimeError("batched search could not schedule a leaf")

    return root


def visualize_children(node: MCTSNode) -> None:
    mcts.visualize_children(node)
