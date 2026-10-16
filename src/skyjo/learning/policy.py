"""Group-normalized policies with the existing per-action logit interface."""

import torch

from skyjo.engine import game


def group_normalized_logits(
    logits: torch.Tensor, spatial: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Mean each legal symmetry group's logits, softmax groups, split their mass.

    Returning mean_logit - log(group_size) at every member makes an ordinary
    action softmax exactly implement that operation. The same logits therefore
    work with inference and cross-entropy against uniformly split group targets.
    Slots match by finger state and their column's multiset, as in slot_orbits;
    flip and replace remain separate, and non-positional actions are singletons.
    All grouping stays on the input device and adds no learned parameters.
    """
    board = spatial[:, 0]
    cards = board.flatten(1, 2)
    columns = board.sum(dim=1).repeat(1, game.ROW_COUNT, 1)
    same = (cards[:, :, None] == cards[:, None, :]).all(dim=-1)
    same &= (columns[:, :, None] == columns[:, None, :]).all(dim=-1)
    parts = [logits[:, : game.MASK_FLIP]]
    for start in (game.MASK_FLIP, game.MASK_REPLACE):
        stop = start + game.ROW_COUNT * game.COLUMN_COUNT
        legal = mask[:, start:stop].bool()
        members = same & legal[:, :, None] & legal[:, None, :]
        weights = members.to(logits.dtype)
        sizes = weights.sum(dim=-1).clamp_min(1)
        # Exclude masked scores before pooling; their values are irrelevant.
        scores = logits[:, start:stop].masked_fill(~legal, 0)
        means = torch.bmm(weights, scores.unsqueeze(-1)).squeeze(-1) / sizes
        parts.append(means - sizes.log())
    return torch.cat(parts, dim=-1).masked_fill(~mask.bool(), -1e10)
