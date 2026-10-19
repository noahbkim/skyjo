"""Status: experimental
Purpose: Diagnose exploration, value ranking and accumulated visit targets.
Promote when: These frozen-position probes become a recurring validation suite.
"""

import dataclasses
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import typer

from skyjo.analytics import explain
from skyjo.engine import game
from skyjo.experiments import contestants, runs
from skyjo.experiments import search_diagnostics as diagnostics
from skyjo.experiments.learning_loop_diagnostic import BASE
from skyjo.experiments.search_positions import Expectation, Position, Witness
from skyjo.experiments.search_strategy_benchmark import exact_action_values, load_cases
from skyjo.learning.predictor import LocalPredictor


def positions(fixtures):
    _, cases = load_cases(fixtures)
    result = []
    for case in cases:
        reference = case.metadata.get("exact_action_values")
        if case.cohort == "tactical":
            expected = Expectation(
                "avoid",
                tuple(case.metadata["giveaway_actions"]),
                "Labeled giveaway; not an optimal-action oracle",
            )
            witnesses = (Witness("clear", tuple(case.metadata["clear_sequence"])),)
        elif reference:
            actual = exact_action_values(case.state)
            assert all(np.isclose(actual[int(a)], v) for a, v in reference.items())
            expected = Expectation(
                "prefer",
                tuple(
                    int(a)
                    for a, v in reference.items()
                    if np.isclose(v, max(reference.values()))
                ),
                "Exact terminal win-credit maximizers",
            )
            witnesses = ()
        else:
            continue
        result.append(
            (
                Position(
                    case.name,
                    case.name,
                    (case.cohort,),
                    case.state,
                    expected,
                    witnesses,
                ),
                reference,
            )
        )
    clear = explain.create_obvious_clear_position()
    for name, state, target, line in [
        (
            "own_take_clear",
            clear,
            game.MASK_TAKE,
            (game.MASK_TAKE, game.MASK_REPLACE + 8),
        ),
        (
            "own_place_clear",
            game.take(clear),
            game.MASK_REPLACE + 8,
            (game.MASK_REPLACE + 8,),
        ),
        (
            "own_no_clear_control",
            explain.create_almost_clear_position(),
            game.MASK_DRAW,
            (),
        ),
    ]:
        result.append(
            (
                Position(
                    name,
                    name,
                    ("heuristic",),
                    state,
                    Expectation(
                        "prefer",
                        (target,),
                        "Handcrafted heuristic; no full-game oracle",
                    ),
                    (Witness("clear", line),) if line else (),
                ),
                None,
            )
        )
    return result


def measure_snapshot(snapshot, position, reference, previous=None):
    groups = snapshot["groups"]
    visits = np.array([g["visits"] for g in groups], dtype=float)
    target = visits / visits.sum()
    accepted = np.array(
        [
            np.mean([position.expectation.accepts(a) for a in g["actions"]])
            for g in groups
        ]
    )
    available = [g for g in groups if g["q"] is not None]
    q_best = max(available, key=lambda g: g["q"])
    good_q = [
        g["q"]
        for g, good in zip(groups, accepted, strict=True)
        if good == 1 and g["q"] is not None
    ]
    bad_q = [
        g["q"]
        for g, good in zip(groups, accepted, strict=True)
        if good == 0 and g["q"] is not None
    ]
    result = {
        "iteration": snapshot["iteration"],
        "choice": snapshot["action"],
        "accepted": position.expectation.accepts(snapshot["action"]),
        "accepted_target_mass": float(target @ accepted),
        "q_best_actions": q_best["actions"],
        "q_best_accepted": bool(position.expectation.accepts(q_best["actions"][0])),
        "good_minus_bad_q": max(good_q) - max(bad_q) if good_q and bad_q else None,
        "groups": groups,
    }
    if previous is not None:
        old = {tuple(g["actions"]): g["visits"] for g in previous["groups"]}
        delta = np.array([g["visits"] - old[tuple(g["actions"])] for g in groups])
        assert (delta >= 0).all()
        result["recent_accepted_mass"] = float(delta @ accepted / delta.sum())
    if reference:
        values = np.array(
            [np.mean([reference[str(a)] for a in g["actions"]]) for g in groups]
        )
        result["target_exact_regret"] = float(max(reference.values()) - target @ values)
        result["choice_exact_regret"] = (
            max(reference.values()) - reference[str(snapshot["action"])]
        )
        result["q_choice_exact_regret"] = (
            max(reference.values()) - reference[str(q_best["actions"][0])]
        )
    return result


_evaluators = {}


def initialize(checkpoints):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    for name, path in checkpoints.items():
        _evaluators[name] = LocalPredictor(
            contestants.load_model(Path(path)), max_batch_size=512
        )


def trial(task):
    model, position, reference, method, config, seed, budget = task
    evaluator = _evaluators[model]
    measured = diagnostics.run_search(position, evaluator, budget, seed, config=config)
    snapshots = []
    previous = None
    for snapshot in measured["trace"]:
        if snapshot["iteration"] < 32:
            continue
        snapshots.append(measure_snapshot(snapshot, position, reference, previous))
        previous = snapshot
    return {
        "model": model,
        "case": position.id,
        "exact": reference is not None,
        "method": method,
        "seed": seed,
        "budget": budget,
        "witnesses": measured["witnesses"],
        "seconds": measured["seconds"],
        "evaluator_states": measured["evaluator_states"],
        "snapshots": snapshots,
    }


def probe(checkpoint_plan: Path, output: Path, workers: int = 8, smoke: bool = False):
    root = Path(__file__).resolve().parent
    fixtures = root / "configs/benchmarks/search_positions.json"
    cases = positions(fixtures)
    plan = json.loads(checkpoint_plan.read_text())
    checkpoints = {
        name: plan["checkpoints"][name]["path"] for name in ["candidate12", "before4"]
    }
    methods = {
        "learned": BASE,
        "noisy": dataclasses.replace(BASE, dirichlet_epsilon=0.25),
        "uniform": dataclasses.replace(BASE, uniform_group_priors=True),
        "uniform_noisy": dataclasses.replace(
            BASE, uniform_group_priors=True, dirichlet_epsilon=0.25
        ),
        "mean_return": dataclasses.replace(BASE, valuation="mean_return"),
    }
    if smoke:
        cases = [
            next(c for c in cases if c[1]),
            next(c for c in cases if c[0].id == "own_take_clear"),
        ]
        methods = {k: methods[k] for k in ["learned", "noisy"]}
        checkpoints = {"candidate12": checkpoints["candidate12"]}
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running",
        "started_at": runs.utc_now(),
        "git": runs.git_provenance(root, allow_dirty=smoke),
        "script_sha256": runs.file_digest(Path(__file__)),
        "fixtures_sha256": runs.file_digest(fixtures),
        "checkpoints": {
            k: {"path": p, "sha256": runs.file_digest(Path(p))}
            for k, p in checkpoints.items()
        },
        "methods": {k: dataclasses.asdict(v) for k, v in methods.items()},
        "workers": workers,
        "threads": 1,
        "smoke": smoke,
    }
    tasks = [
        (
            model,
            p,
            reference,
            method,
            config,
            seed,
            32 if smoke else 256 if reference else 2048,
        )
        for model in checkpoints
        for p, reference in cases
        for method, config in methods.items()
        if model == "candidate12"
        or (reference is None and method in {"learned", "noisy"})
        for seed in range(1 if smoke else 2 if reference else 4)
    ]
    manifest["planned_trials"] = len(tasks)
    (output / "positions.json").write_text(
        json.dumps([p.to_dict() for p, _ in cases], indent=2) + "\n"
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        initialize(checkpoints)
        probes = []
        for model, evaluator in _evaluators.items():
            for p, reference in cases:
                raw = diagnostics.model_diagnostic(p, evaluator)
                if reference:
                    raw["raw_exact_regret"] = max(reference.values()) - sum(
                        raw["policy"][int(a)] * v for a, v in reference.items()
                    )
                probes.append({"model": model, **raw})
        (output / "policy.json").write_text(json.dumps(probes, indent=2) + "\n")
        with (
            ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=initialize,
                initargs=(checkpoints,),
            ) as pool,
            (output / "trials.jsonl").open("w") as stream,
        ):
            futures = [pool.submit(trial, task) for task in tasks]
            for count, future in enumerate(as_completed(futures), 1):
                stream.write(json.dumps(future.result(), allow_nan=False) + "\n")
                stream.flush()
                if count % 20 == 0 or count == len(tasks):
                    print(f"Completed {count}/{len(tasks)} probes", flush=True)
        manifest.update(status="completed", finished_at=runs.utc_now())
    except BaseException as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output


if __name__ == "__main__":
    typer.run(probe)
