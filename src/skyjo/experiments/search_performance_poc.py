"""Experimental CPU-only alternatives; never installed by production callers.

Predictors own a private frozen model. Snapshot substitution is process-local,
not thread-safe, and restricted to the experiment's context manager.
"""

from contextlib import contextmanager
from copy import deepcopy

import numpy as np
import torch

from skyjo.engine import game
from skyjo.learning import batches, observations, predictor
from skyjo.search import mcts
from skyjo.search.evaluator import Prediction
from skyjo.search.strategies import NodeView, frozen_array

VARIANTS = (
    "baseline",
    "mode",
    "packing",
    "sharing",
    "inference",
    "snapshots",
    "combined",
)


class PackedInputs:
    """Reusable CPU buffers. Tensor views must be consumed before the next fill."""

    def __init__(self, capacity, players):
        self.spatial = np.empty(
            (capacity, *observations.spatial_input_shape(players)), np.float32
        )
        self.non_spatial = np.empty(
            (capacity, *observations.get_non_spatial_input_shape(players)), np.float32
        )
        self.masks = np.empty((capacity, game.MASK_SIZE), np.float32)

    def fill(self, states):
        for i, state in enumerate(states):
            self.spatial[i] = game.get_table(state)
            row = self.non_spatial[i]
            row[: game.GAME_SIZE] = game.get_game(state)
            row[game.GAME_SIZE : game.GAME_SIZE + game.CARD_SIZE] = game.get_deck(state)
            offset = game.GAME_SIZE + game.CARD_SIZE
            row[offset] = game.get_turn(state)
            countdown = game.get_countdown(state)
            row[offset + 1] = -1 if countdown is None else countdown
            row[offset + 2 :] = game.get_turn(state) - game.get_last_revealed_turns(
                state
            )
            self.masks[i] = game.actions(state)
        n = len(states)
        return batches.TrainingBatch(
            self.spatial[:n], self.non_spatial[:n], self.masks[:n]
        )


class ExperimentalPredictor:
    def __init__(
        self, model, *, mode=False, packing=False, sharing=False, max_batch_size=64
    ):
        if str(model.device) != "cpu":
            raise ValueError("This experiment is CPU-only")
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        self.model = deepcopy(model).eval()
        self.model.requires_grad_(False)
        self.mode, self.packing, self.sharing = mode, packing, sharing
        self.max_batch_size = max_batch_size
        self.inputs = PackedInputs(max_batch_size, model.players) if packing else None

    @torch.inference_mode()
    def evaluate(self, states):
        if not states:
            return []
        if not self.mode:
            self.model.eval()
        predictions = []
        for start in range(0, len(states), self.max_batch_size):
            selected = states[start : start + self.max_batch_size]
            batch = (
                self.inputs.fill(selected)
                if self.packing
                else batches.states_to_batch(selected)
            )
            if self.sharing:
                spatial, non_spatial, masks = (
                    torch.from_numpy(array)
                    for array in (
                        batch.spatial_inputs,
                        batch.non_spatial_inputs,
                        batch.action_masks,
                    )
                )
            else:
                tensors = batches.to_tensors(batch, device=self.model.device)
                spatial, non_spatial, masks = (
                    tensors.spatial_inputs,
                    tensors.non_spatial_inputs,
                    tensors.action_masks,
                )
            output = self.model(spatial, non_spatial, masks)
            values = output.value.cpu().numpy()
            policies = torch.softmax(output.policy_logits, dim=-1).cpu().numpy()
            predictions.extend(
                Prediction(np.roll(value, game.get_player(state)), policy)
                for state, value, policy in zip(selected, values, policies, strict=True)
            )
        return predictions


def make_predictor(model, variant, *, max_batch_size=64):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    if variant in {"baseline", "snapshots"}:
        return predictor.LocalPredictor(deepcopy(model).eval(), max_batch_size)
    all_changes = variant in {"inference", "combined"}
    return ExperimentalPredictor(
        model,
        mode=all_changes or variant == "mode",
        packing=all_changes or variant == "packing",
        sharing=all_changes or variant == "sharing",
        max_batch_size=max_batch_size,
    )


class CachedDecisionNode(mcts.DecisionStateNode):
    """Conservative content-keyed cache, including in-place array changes.

    Content signatures deliberately retain checking overhead. A versioned tree
    could avoid it but would require auditing every mutation site. Old snapshots
    own their dynamic arrays; cached static arrays are also immutable copies.
    """

    __slots__ = ("_cached_view", "_static", "_static_key", "_view_key")

    def view(self):
        actions = tuple(self.children)
        static_key = (
            actions,
            self.action_priors.tobytes(),
            self.model_action_priors.tobytes(),
            self.base_action_priors.tobytes(),
        )
        if static_key != getattr(self, "_static_key", None):
            self._static_key = static_key
            self._static = tuple(
                frozen_array([priors[a] for a in actions])
                for priors in (
                    self.action_priors,
                    self.model_action_priors,
                    self.base_action_priors,
                )
            )
            self._view_key = None
        children = tuple(self.children.values())
        visits = [child.visit_count for child in children]
        values = [child.state_value for child in children]
        available = [child.has_value_estimate for child in children]
        key = (
            tuple(visits),
            tuple(value.tobytes() for value in values),
            tuple(available),
            self.state_value.tobytes(),
            self.model_prediction.value.tobytes(),
            self.visit_count,
            game.get_player(self.state),
            self.parent is None,
            self.action_groups,
            self._node_id,
        )
        if key != getattr(self, "_view_key", None):
            self._cached_view = NodeView(
                self.action_groups,
                actions,
                self._static[0],
                self._static[1],
                frozen_array(visits),
                frozen_array(values),
                frozen_array(available),
                game.get_player(self.state),
                frozen_array(self.state_value),
                frozen_array(self.model_prediction.value),
                self.visit_count,
                self.parent is None,
                node_id=self._node_id,
                base_priors=self._static[2],
            )
            self._view_key = key
        return self._cached_view


@contextmanager
def snapshot_variant(variant):
    """Restore the production class even after a failed experimental search."""
    original = mcts.DecisionStateNode
    try:
        if variant in {"snapshots", "combined"}:
            mcts.DecisionStateNode = CachedDecisionNode
        yield
    finally:
        mcts.DecisionStateNode = original
