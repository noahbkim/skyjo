"""Model, optimizer and replay sampler shared by online and offline training."""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np
import torch

from . import losses, models, objectives, randomness, train


@dataclass
class Learner:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    loss_function: losses.LossFunction
    sampling_rng: np.random.Generator

    @classmethod
    def create(
        cls,
        model_settings: dict,
        *,
        players: int,
        device: torch.device | str,
        learn_rate: float,
        seed: int,
        auxiliary_objectives: objectives.ObjectiveConfig = None,
        value_scale: float = 1.0,
        policy_scale: float = 1.0,
    ) -> Learner:
        randomness.set_seed(seed)
        model = models.build(
            model_settings,
            players=players,
            device=device,
            auxiliary_objectives=auxiliary_objectives,
        )
        loss = functools.partial(
            losses.configured_loss,
            auxiliary_objectives=auxiliary_objectives,
            value_scale=value_scale,
            policy_scale=policy_scale,
        )
        sampling_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x53414D50]))
        randomness.set_seed(
            int(np.random.SeedSequence([seed, 0x54524149]).generate_state(1)[0])
        )
        return cls(model, train.make_optimizer(model, learn_rate), loss, sampling_rng)

    @classmethod
    def from_configuration(cls, configuration: dict, seed: int) -> Learner:
        """Adapt a resolved workflow configuration at the composition boundary."""
        settings = configuration["training"]
        return cls.create(
            configuration["model"],
            players=configuration["players"],
            device=configuration["execution"]["device"],
            seed=seed,
            auxiliary_objectives=configuration["auxiliary_objectives"],
            learn_rate=settings["learn_rate"],
            value_scale=settings["value_scale"],
            policy_scale=settings["policy_scale"],
        )

    def fit(self, replay, *, steps, batch_size, diagnostics=None, gradient_scales=None):
        return train.train_steps(
            self.model,
            replay,
            training_batch_size=batch_size,
            optimizer_steps=steps,
            optimizer=self.optimizer,
            loss_function=self.loss_function,
            diagnostics=diagnostics,
            gradient_scales=gradient_scales,
            sampling_rng=self.sampling_rng,
        )

    def evaluate(self, replay, *, batch_size, indices=None, diagnostics=False):
        return train.evaluate_loss(
            self.model,
            replay,
            batch_size,
            self.loss_function,
            indices=indices,
            diagnostics=diagnostics,
        )
