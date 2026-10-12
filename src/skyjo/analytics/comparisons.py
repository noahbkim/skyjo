"""Pure reductions of game and training comparisons."""

import numpy as np


def summarize_match(games):
    if not games:
        raise ValueError("A match must contain completed games")
    return {
        "variant_win_fraction": float(
            np.mean([g["variant_win_credit"] for g in games])
        ),
        "control_minus_variant_margin": float(
            np.mean([g["control_minus_variant"] for g in games])
        ),
    }


def summarize(completed, control):
    pairs = []
    for child in completed:
        if child["variant"] == control:
            continue
        baseline = next(
            c
            for c in completed
            if c["variant"] == control and c["seed"] == child["seed"]
        )
        # Only common, unweighted errors are comparable across loss definitions.
        shared = child["validation"].keys() & baseline["validation"].keys()
        keys = [
            k
            for k in sorted(shared)
            if k != "total_loss"
            and "weighted" not in k
            and (k.endswith(("_loss", "_mae_points")) or k == "policy_kl")
        ]
        delta = {
            k: child["validation"][k] - baseline["validation"][k]
            for k in keys
            if child["validation"][k] is not None
            and baseline["validation"][k] is not None
        }
        pairs.append(
            {
                "variant": child["variant"],
                "seed": child["seed"],
                "control_run_id": baseline["run_id"],
                "variant_run_id": child["run_id"],
                "variant_minus_control": delta,
            }
        )
    averages = {}
    for variant in dict.fromkeys(c["variant"] for c in completed):
        selected = [c for c in completed if c["variant"] == variant]
        metrics = {}
        for key in sorted(set().union(*(c["validation"].keys() for c in selected))):
            values = [
                c["validation"][key]
                for c in selected
                if c["validation"].get(key) is not None
            ]
            if values:
                metrics[key] = {
                    "mean": float(np.mean(values)),
                    "seed_count": len(values),
                }
        paired = [p["variant_minus_control"] for p in pairs if p["variant"] == variant]
        deltas = (
            {
                key: float(np.mean([p[key] for p in paired if key in p]))
                for key in set().union(*paired)
            }
            if paired
            else {}
        )
        averages[variant] = {
            "validation": metrics,
            "paired_differences": deltas,
            **{
                key: float(np.mean([c[key] for c in selected]))
                for key in (
                    "training_seconds",
                    "evaluation_seconds",
                    "elapsed_seconds",
                    "sampled_positions",
                    "parameter_count",
                    "optimizer_steps",
                )
            },
        }
    return {
        "runs": completed,
        "paired_differences": pairs,
        "averages": averages,
        "interpretation": "Fixed-replay learning, not playing strength. Steps are matched; exposure and compute may differ. Weighted totals across different objectives are not rankings.",
    }
