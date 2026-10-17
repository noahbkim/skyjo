"""Paired fixed-replay loss ablations with game-disjoint selection and testing."""

from __future__ import annotations

import copy
import functools
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from skyjo.experiments import contestants
from skyjo.experiments.learning_loop_fitting import metrics
from skyjo.learning import batches, losses, train

SPLITS = ("train", "validation", "test")


def choose_split(
    offsets, identities, eligible, excluded, game_counts, row_counts, seed
):
    """Choose disjoint whole games, then sample rows only inside each split."""
    excluded = set(excluded)
    eligible = np.asarray([i for i in eligible if identities[i] not in excluded])
    rng = np.random.default_rng(seed)
    selected = rng.choice(eligible, size=sum(game_counts), replace=False)
    result = {}
    start = 0
    for name, count, rows in zip(SPLITS, game_counts, row_counts, strict=True):
        chosen = selected[start : start + count]
        start += count
        candidates = np.concatenate(
            [np.arange(offsets[i], offsets[i + 1]) for i in chosen]
        )
        result[f"{name}_games"] = identities[chosen]
        result[f"{name}_rows"] = np.sort(
            rng.choice(candidates, size=min(rows, len(candidates)), replace=False)
        )
    return result


def prepare_split(plan, output):
    path = Path(plan["replay"])
    manifest = json.loads((path / "manifest.json").read_text())
    generation = manifest["generation_metadata"]["latest_generated_batch"]
    if (
        Path(generation["checkpoint_path"]).resolve()
        != Path(plan["fit_checkpoint"]).resolve()
    ):
        raise ValueError(
            "Start from the checkpoint that generated the selected fresh targets"
        )
    identities = np.load(path / "game_indices.npy")
    offsets = np.load(path / "game_offsets.npy")
    eligible = np.flatnonzero(
        identities >= identities.max() - generation["game_count"] + 1
    )
    with np.load(plan["previous_selection"]) as previous:
        excluded = np.concatenate(
            [previous[f"{name}_games"] for name in ("train", "validation")]
        )
    selection = choose_split(
        offsets,
        identities,
        eligible,
        excluded,
        plan["split_games"],
        plan["split_positions"],
        plan["split_seed"],
    )
    np.savez(output / "selection.npz", **selection)
    return {
        "dataset_id": manifest["dataset_id"],
        "generation": generation,
        "excluded_prior_games": excluded.tolist(),
        "counts": {
            name: {
                "games": len(selection[f"{name}_games"]),
                "positions": len(selection[f"{name}_rows"]),
            }
            for name in SPLITS
        },
    }


def load_split(replay, selection_path):
    path = Path(replay)
    manifest = json.loads((path / "manifest.json").read_text())
    arrays = {
        name: np.load(path / f"{name}.npy", mmap_mode="r")
        for name in ("spatial_inputs", "non_spatial_inputs", "action_masks")
    }
    targets = {
        spec["name"]: np.load(path / "targets" / f"{spec['name']}.npy", mmap_mode="r")
        for spec in manifest["target_specs"]
    }
    result = {}
    with np.load(selection_path) as selection:
        for name in SPLITS:
            rows = selection[f"{name}_rows"]
            result[name] = batches.TrainingBatch(
                **{key: value[rows] for key, value in arrays.items()},
                targets={key: value[rows] for key, value in targets.items()},
            )
    return result


def fit_arm(plan, selection_path, value_scale, auxiliaries, seed):
    torch.manual_seed(seed)
    data = load_split(plan["replay"], selection_path)
    model = contestants.load_model(Path(plan["fit_checkpoint"]))
    initial = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = functools.partial(
        losses.configured_loss,
        value_scale=value_scale,
        auxiliary_objectives={"round_raw_score": 1.0, "round_doubled": 0.1}
        if auxiliaries
        else {},
    )
    rng = np.random.default_rng(seed)
    snapshots = []
    best = None
    for step in range(max(plan["fit_steps"]) + 1):
        if step in plan["fit_steps"]:
            row = {
                "step": step,
                "sampled_positions": step * 256,
                "train": metrics(model, data["train"]),
                "validation": metrics(model, data["validation"]),
            }
            snapshots.append(row)
            if (
                best is None
                or row["validation"]["policy/all/target_kl"]
                < best["validation"]["policy/all/target_kl"]
            ):
                best = row
                best_state = copy.deepcopy(model.state_dict())
        if step < max(plan["fit_steps"]):
            batch = data["train"][rng.integers(0, len(data["train"]), 256)]
            value, _ = train.train_step(model, batch, loss, optimizer)
            if not np.isfinite(value):
                raise ValueError("Non-finite sweep loss")
    # The test partition is first evaluated after validation has selected a step.
    model.load_state_dict(initial)
    test_before = metrics(model, data["test"])
    model.load_state_dict(best_state)
    return {
        "value_scale": value_scale,
        "auxiliaries": auxiliaries,
        "seed": seed,
        "snapshots": snapshots,
        "selected_step": best["step"],
        "selected_validation": best["validation"],
        "test_before": test_before,
        "test_after": metrics(model, data["test"]),
    }


def initialize():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def run_sweep(plan, output):
    output.mkdir(exist_ok=False)
    provenance = prepare_split(plan, output)
    (output / "split.json").write_text(json.dumps(provenance, indent=2) + "\n")
    rows = []
    with ProcessPoolExecutor(
        max_workers=plan["workers"],
        mp_context=mp.get_context("spawn"),
        initializer=initialize,
    ) as pool:
        pending = [
            pool.submit(fit_arm, plan, output / "selection.npz", weight, aux, seed)
            for weight in plan["value_scales"]
            for aux in (False, True)
            for seed in plan["fit_seeds"]
        ]
        with (output / "arms.jsonl").open("x") as stream:
            for future in as_completed(pending):
                row = future.result()
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                print(f"Loss sweep: {len(rows)}/{len(pending)}", flush=True)
    summary = []
    for weight in plan["value_scales"]:
        for aux in (False, True):
            selected = [
                r
                for r in rows
                if r["value_scale"] == weight and r["auxiliaries"] == aux
            ]
            summary.append(
                {
                    "value_scale": weight,
                    "auxiliaries": aux,
                    "seeds": len(selected),
                    "selected_steps": [
                        r["selected_step"]
                        for r in sorted(selected, key=lambda r: r["seed"])
                    ],
                    "means": {
                        partition: {
                            metric: float(
                                np.mean([r[partition][metric] for r in selected])
                            )
                            for metric in ("policy/all/target_kl", "value_mse")
                        }
                        for partition in (
                            "selected_validation",
                            "test_before",
                            "test_after",
                        )
                    },
                }
            )
    report = {
        "split": provenance,
        "selection": "Minimum validation policy KL over recorded steps, including unchanged step0. Always inspect value MSE alongside policy KL. Test results did not select a step.",
        "interpretation": "Three minibatch seeds on one fixed game split, not independent self-play training runs. Test games excluded from this fitting and from the previous diagnostic subset.",
        "arms": summary,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
