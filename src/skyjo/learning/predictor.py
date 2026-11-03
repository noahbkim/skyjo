"""Neural adapters for the NumPy search and simulation contracts."""

from collections.abc import Sequence

import numpy as np
import torch

from skyjo.engine import game as sj
from skyjo.search.batching import EvaluationRequest, evaluate
from skyjo.search.evaluator import Evaluator, Prediction
from skyjo.search.strategies import softmax
from skyjo.search.symmetry import ActionGroups
from skyjo.simulation.decisions import DiagnosticSection, PlayerDecision

from . import batches, skynet


class LocalPredictor:
    """Encode states and evaluate bounded tensor batches in input order."""

    def __init__(
        self, model: skynet.SkyNet, max_batch_size: int, *, fixed_eval_mode=False
    ):
        # Opt in only for an inference-owned model that will never switch to train.
        self.fixed_eval_mode = fixed_eval_mode
        if fixed_eval_mode:
            model.eval()
        if type(max_batch_size) is not int or max_batch_size <= 0:
            raise ValueError("max_batch_size must be a positive integer")
        self.model = model
        self.max_batch_size = max_batch_size

    @torch.inference_mode()
    def evaluate(self, states: Sequence[sj.Skyjo]) -> list[Prediction]:
        if not states:
            return []
        if not self.fixed_eval_mode:
            self.model.eval()
        predictions = []
        for start in range(0, len(states), self.max_batch_size):
            selected = states[start : start + self.max_batch_size]
            tensors = batches.to_tensors(
                batches.states_to_batch(selected), device=self.model.device
            )
            output = self.model(
                tensors.spatial_inputs, tensors.non_spatial_inputs, tensors.action_masks
            )
            values = output.value.cpu().numpy()
            policies = torch.softmax(output.policy_logits, dim=-1).cpu().numpy()
            predictions.extend(
                Prediction(np.roll(value, sj.get_player(state)), policy)
                for state, value, policy in zip(selected, values, policies, strict=True)
            )
        return predictions


class PolicyPlayer:
    """Temperature-controlled play in the model's action/group probability space."""

    def __init__(
        self,
        evaluator: Evaluator,
        *,
        temperature: float = 0.0,
        group_actions: bool | None = None,
    ):
        if not np.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        self.evaluator = evaluator
        self.temperature = temperature
        if group_actions is None:
            group_actions = (
                isinstance(evaluator, LocalPredictor)
                and evaluator.model.policy_normalization == "groups"
            )
        self.group_actions = group_actions

    def get_action_probabilities(self, state: sj.Skyjo) -> np.ndarray:
        return self.get_decision(state, diagnostics=False).probabilities

    def get_decision(self, state: sj.Skyjo, *, diagnostics=True) -> PlayerDecision:
        return evaluate(self.decision_steps(state, diagnostics=diagnostics))

    def decision_steps(self, state, *, diagnostics=False):
        predictions = yield EvaluationRequest(self.evaluator, [state])
        prediction = predictions[0]
        legal = np.flatnonzero(sj.actions(state))
        groups = ActionGroups.from_state(state, merge=self.group_actions)
        priors = groups.aggregate(prediction.policy)[list(groups.representatives)]
        logits = np.full(len(priors), -np.inf)
        positive = priors > 0
        logits[positive] = np.log(priors[positive])
        probabilities = groups.distribute(
            softmax(logits, self.temperature), greedy=self.temperature == 0
        )
        return PlayerDecision(
            probabilities,
            (
                DiagnosticSection(
                    "Model policy",
                    text="Temperature applies to total probability per symmetry group."
                    if self.group_actions
                    else "Temperature applies to individual action probabilities.",
                    metrics={"Temperature": self.temperature},
                    columns=("Action", "Model prior", "Selection probability"),
                    rows=tuple(
                        (
                            sj.get_action_name(int(action)),
                            float(prediction.policy[action]),
                            float(probabilities[action]),
                        )
                        for action in legal
                    ),
                    row_actions=tuple(map(int, legal)),
                ),
                DiagnosticSection(
                    "Win estimates (fixed seat order)",
                    metrics={
                        f"Seat {i + 1}": float(value)
                        for i, value in enumerate(prediction.value)
                    },
                ),
            )
            if diagnostics
            else (),
        )
