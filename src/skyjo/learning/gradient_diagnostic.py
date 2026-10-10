"""First-batch gradient scales, using the existing forward graph without .grad writes."""

import time

import torch

from skyjo.learning import losses
from skyjo.learning import objectives


def measure(
    model,
    output,
    targets,
    *,
    value_scale=1.0,
    policy_scale=1.0,
    auxiliary_objectives=None,
):
    started = time.perf_counter()
    parameters = [
        p
        for name, p in model.named_parameters()
        if p.requires_grad
        and not name.startswith(("value_tail.", "policy_tail.", "auxiliary_heads."))
    ]

    def norm(loss):
        gradients = torch.autograd.grad(
            loss, parameters, retain_graph=True, allow_unused=True
        )
        return (
            sum(
                g.detach().double().square().sum().item()
                for g in gradients
                if g is not None
            )
            ** 0.5
        )

    core, _ = losses.base_loss(output, targets, value_scale, policy_scale)
    core_norm = norm(core)
    metrics = {"core_weighted_norm": core_norm}
    for name, weight in objectives.resolve(auxiliary_objectives).entries:
        loss, _ = losses.auxiliary_loss(
            name, output.auxiliary_outputs[name], targets[name]
        )
        unweighted = norm(loss)
        metrics.update(
            {
                f"{name}/unweighted_norm": unweighted,
                f"{name}/weighted_norm": weight * unweighted,
                f"{name}/unweighted_to_core": unweighted / core_norm
                if core_norm
                else None,
                f"{name}/weighted_to_core": weight * unweighted / core_norm
                if core_norm
                else None,
            }
        )
    metrics["seconds"] = time.perf_counter() - started
    return metrics
