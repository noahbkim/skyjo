"""Offline execution shared by the single-run CLI and fixed-replay comparisons."""

import functools
import random
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch

from . import checkpoint, models, objectives, train


@contextmanager
def preserve_rng():
    state = checkpoint.capture_rng_state()
    try:
        yield
    finally:
        checkpoint.restore_rng_state(state)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class OfflineTrainer:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    loss_function: object
    sampling_rng: np.random.Generator | None = None

    @classmethod
    def from_configuration(cls, configuration, seed):
        set_seed(seed)
        model = models.build(
            configuration["model"],
            players=configuration["players"],
            device=configuration["execution"]["device"],
            auxiliary_objectives=configuration["auxiliary_objectives"],
        )
        settings = configuration["training"]
        loss = functools.partial(
            objectives.configured_loss,
            auxiliary_objectives=configuration["auxiliary_objectives"],
            value_scale=settings["value_scale"],
            policy_scale=settings["policy_scale"],
        )
        # Model construction and stochastic training never advance this stream.
        sampling_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x53414D50]))
        set_seed(int(np.random.SeedSequence([seed, 0x54524149]).generate_state(1)[0]))
        return cls(
            model,
            train.make_optimizer(model, settings["learn_rate"]),
            loss,
            sampling_rng,
        )

    def fit(self, replay, *, steps, batch_size, gradient_scales=None):
        return train.train_steps(
            self.model,
            replay,
            training_batch_size=batch_size,
            optimizer_steps=steps,
            optimizer=self.optimizer,
            loss_function=self.loss_function,
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
