"""Tensor heads for the three supported optional training objectives."""

import torch
from torch import nn

from . import objectives


class NormalizedRoundScoreTail(nn.Sequential):
    """Predict a charged round score on its normalized [0, 1] scale."""

    def __init__(self, width: int, players: int):
        super().__init__(nn.Linear(width, players), nn.Sigmoid())


def make_heads(
    resolved: objectives.ResolvedObjectives, width: int, players: int
) -> nn.ModuleDict:
    heads = nn.ModuleDict()
    # Optional heads neither perturb core initialization nor one another.
    for name, _ in resolved.entries:
        with torch.random.fork_rng(devices=[]):
            heads[name] = (
                NormalizedRoundScoreTail(width, players)
                if name == "round_score"
                else nn.Linear(width, players)
            )
    return heads
