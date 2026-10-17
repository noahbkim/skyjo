"""Fixed-target fitting probes on disposable models; never alter source weights."""

from __future__ import annotations

import dataclasses
import functools
import json
from pathlib import Path

import numpy as np
import torch

from skyjo.experiments import contestants
from skyjo.learning import batches, losses, train


def replay_subset(path, seed, output):
    """Read only selected rows from the latest generation; split by whole game."""
    manifest = json.loads((path / "manifest.json").read_text())
    offsets = np.load(path / "game_offsets.npy")
    identities = np.load(path / "game_indices.npy")
    latest_count = manifest["generation_metadata"]["latest_generated_batch"][
        "game_count"
    ]
    eligible = np.flatnonzero(identities >= identities.max() - latest_count + 1)
    selected = np.random.default_rng(seed).choice(eligible, size=64, replace=False)
    rng = np.random.default_rng(seed)
    indices = []
    for games, count in ((selected[:48], 4096), (selected[48:], 2048)):
        rows = np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i in games])
        indices.append(np.sort(rng.choice(rows, min(count, len(rows)), replace=False)))
    np.savez(
        output / "replay-selection.npz",
        train_rows=indices[0],
        validation_rows=indices[1],
        train_games=identities[selected[:48]],
        validation_games=identities[selected[48:]],
    )
    arrays = {
        name: np.load(path / f"{name}.npy", mmap_mode="r")
        for name in ("spatial_inputs", "non_spatial_inputs", "action_masks")
    }
    targets = {
        spec["name"]: np.load(path / "targets" / f"{spec['name']}.npy", mmap_mode="r")
        for spec in manifest["target_specs"]
    }
    result = [
        batches.TrainingBatch(
            **{k: v[i] for k, v in arrays.items()},
            targets={k: v[i] for k, v in targets.items()},
        )
        for i in indices
    ]
    return result, {
        "dataset_id": manifest["dataset_id"],
        "replay_path": str(path),
        "generation": manifest["generation_metadata"]["latest_generated_batch"],
        "train_games": 48,
        "validation_games": 16,
        "train_positions": len(result[0]),
        "validation_positions": len(result[1]),
        "interpretation": "Game-disjoint for diagnostic fitting only; these rows were available during the original training run.",
    }


@torch.no_grad()
def metrics(model, batch):
    model.eval()
    diagnostics = train.TrainingDiagnostics()
    value_error = 0.0
    oracle_regret = 0.0
    for start in range(0, len(batch), 256):
        tensors = batches.to_tensors(batch[start : start + 256], device=model.device)
        predicted = model(
            tensors.spatial_inputs, tensors.non_spatial_inputs, tensors.action_masks
        )
        diagnostics.update(
            predicted.policy_logits, tensors.targets["policy"], tensors.action_masks
        )
        value_error += float(
            (predicted.value - tensors.targets["value"]).square().mean()
        ) * len(tensors.spatial_inputs)
        if "oracle_values" in tensors.targets:
            reference = tensors.targets["oracle_values"]
            probabilities = predicted.policy_logits.softmax(dim=-1)
            oracle_regret += float(
                (
                    reference.max(dim=-1).values
                    - (reference * probabilities).sum(dim=-1)
                ).sum()
            )
    return {
        **diagnostics.summary(),
        "value_mse": value_error / len(batch),
        **(
            {"oracle_policy_expected_regret": oracle_regret / len(batch)}
            if "oracle_values" in batch.targets
            else {}
        ),
    }


def policy_loss(output, targets):
    loss = losses.cross_entropy_policy_loss(output.policy_logits, targets["policy"])
    return loss, {"policy_loss": float(loss.detach())}


def fit_copy(checkpoint, training, validation, *, steps, seed, policy_only):
    torch.manual_seed(seed)
    model = contestants.load_model(Path(checkpoint))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = (
        policy_loss
        if policy_only
        else functools.partial(
            losses.configured_loss,
            auxiliary_objectives={"round_raw_score": 1.0, "round_doubled": 0.1},
        )
    )
    rng = np.random.default_rng(seed)
    snapshots = []
    for step in range(max(steps) + 1):
        if step in steps:
            snapshots.append(
                {
                    "step": step,
                    "sampled_positions": step * 256,
                    "train": metrics(model, training),
                    "validation": metrics(model, validation),
                }
            )
        if step < max(steps):
            batch = training[rng.integers(0, len(training), 256)]
            value, _ = train.train_step(model, batch, loss, optimizer)
            if not np.isfinite(value):
                raise ValueError("Non-finite diagnostic training loss")
    return snapshots


def fit_diagnostics(plan, output, positions, references, rows):
    from skyjo.experiments.learning_loop_diagnostic import write_json

    torch.set_num_threads(1)
    replay, provenance = replay_subset(Path(plan["replay"]), plan["seed"], output)
    result = {
        "replay": provenance,
        "before_after_original_training": {},
        "replay_fit": [],
        "exact_target_fit": [],
    }
    for name in ("initial", "generator", "trained"):
        model = contestants.load_model(Path(plan["checkpoints"][name]))
        result["before_after_original_training"][name] = {
            "train": metrics(model, replay[0]),
            "validation": metrics(model, replay[1]),
        }
    for policy_only in (True, False):
        snapshots = fit_copy(
            plan["checkpoints"]["generator"],
            *replay,
            steps=plan["fit_steps"],
            seed=plan["seed"],
            policy_only=policy_only,
        )
        result["replay_fit"].append(
            {"start": "generator", "policy_only": policy_only, "snapshots": snapshots}
        )
        write_json(output / "fitting.json", result)
    by_id = {p.id: p for p in positions}
    for budget in (256, max(plan["budgets"])):
        selected = []
        for case in references:
            examples = [
                r
                for r in rows
                if r["case"] == case
                and r["model"] == "trained"
                and r["method"] == "base"
                and r["budget"] == budget
            ]
            if (
                examples
                and np.mean(
                    [
                        r["oracle"]["policy_expected_regret"]
                        - r["oracle"]["target_expected_regret"]
                        for r in examples
                    ]
                )
                > 0.01
            ):
                selected.append(
                    (case, np.mean([r["target"] for r in examples], axis=0))
                )
        if not selected:
            result["exact_target_fit"].append(
                {
                    "budget": budget,
                    "cases": [],
                    "status": "no targets improve exact expected win credit by >0.01",
                }
            )
            continue
        inputs = batches.states_to_batch([by_id[c].state for c, _ in selected])
        model = contestants.load_model(Path(plan["checkpoints"]["trained"]))
        tensors = batches.to_tensors(inputs, device=model.device)
        with torch.no_grad():
            values = model(
                tensors.spatial_inputs, tensors.non_spatial_inputs, tensors.action_masks
            ).value.numpy()
        batch = dataclasses.replace(
            inputs,
            targets={
                "policy": np.asarray([p for _, p in selected], dtype=np.float32),
                "value": values,
                "oracle_values": np.asarray(
                    [
                        [references[c].get(a, 0.0) for a in range(28)]
                        for c, _ in selected
                    ],
                    dtype=np.float32,
                ),
            },
        )
        snapshots = fit_copy(
            plan["checkpoints"]["trained"],
            batch,
            batch,
            steps=plan["fit_steps"],
            seed=plan["seed"],
            policy_only=True,
        )
        result["exact_target_fit"].append(
            {
                "budget": budget,
                "cases": [c for c, _ in selected],
                "interpretation": "Intentional same-example fitting test, not held-out generalization; targets independently improve exact expected win credit.",
                "snapshots": snapshots,
            }
        )
    write_json(output / "fitting.json", result)
