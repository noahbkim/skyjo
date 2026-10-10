"""Neural adapters for the NumPy search and simulation contracts."""

from collections.abc import Sequence

import numpy as np
import torch

from skyjo.engine import game as sj
from skyjo.search.evaluator import Evaluator, Prediction

from . import batches, skynet


class LocalPredictor:
    """Encode states and evaluate bounded tensor batches in input order."""

    def __init__(self, model: skynet.SkyNet, max_batch_size: int):
        if type(max_batch_size) is not int or max_batch_size <= 0:
            raise ValueError("max_batch_size must be a positive integer")
        self.model = model
        self.max_batch_size = max_batch_size

    @torch.inference_mode()
    def evaluate(self, states: Sequence[sj.Skyjo]) -> list[Prediction]:
        if not states:
            return []
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
    """Choose the highest-probability legal action without search."""

    def __init__(self, evaluator: Evaluator):
        self.evaluator = evaluator

    def get_action_probabilities(self, state: sj.Skyjo) -> np.ndarray:
        prediction = self.evaluator.evaluate([state])[0]
        legal = np.flatnonzero(sj.actions(state))
        probabilities = np.zeros(sj.MASK_SIZE, dtype=np.float32)
        probabilities[legal[prediction.policy[legal].argmax()]] = 1
        return probabilities
