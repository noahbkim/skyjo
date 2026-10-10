"""Global runtime RNG scopes; replay sampling uses a separate owned Generator."""

import random
from contextlib import contextmanager

import numpy as np
import torch

from skyjo.learning import checkpoint


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
