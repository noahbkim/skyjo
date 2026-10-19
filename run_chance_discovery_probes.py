"""Status: experimental
Purpose: Measure giveaway discovery and recovery under sampled/enumerated chance.
Promote when: Chance-allocation comparisons become recurring search validation.
"""

import dataclasses
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import typer

import run_search_target_probes as targets
from skyjo.experiments import runs
from skyjo.experiments import search_diagnostics as diagnostics
from skyjo.experiments.learning_loop_diagnostic import BASE
from skyjo.search import mcts
from skyjo.search.evaluator import NextDealEvaluator


def trial(task):
    position, method, config, seed, budget = task
    counted = diagnostics.CountingEvaluator(targets._evaluators["candidate12"])
    boundary = diagnostics.CountingBoundary(NextDealEvaluator(counted))
    discovery = None
    first = {}
    streak = {}
    changes = []
    snapshots = []
    previous = None
    observed_root = None
    start = time.perf_counter()

    def observe(root):
        nonlocal discovery, previous, observed_root
        if root is not observed_root:
            observed_root = root
            discovery = previous = None
            first.clear()
            streak.clear()
            changes.clear()
            snapshots.clear()
        iteration = root.visit_count
        view = root.view()
        accepted = np.array([position.expectation.accepts(a) for a in view.actions])
        good = view.q[accepted & view.available]
        bad = view.q[~accepted & view.available]
        gap = float(good.max() - bad.max()) if len(good) and len(bad) else None
        choice = diagnostics.selected_action(root)
        avoids = position.expectation.accepts(choice)
        line = diagnostics.line_snapshot(root, position.witnesses[0].actions)
        assert line["status"] == "available"
        point = {
            "iteration": iteration,
            "evaluator_states": counted.states,
            "boundary_samples": boundary.states,
            "seconds": time.perf_counter() - start,
            "choice": choice,
            "good_minus_bad_q": gap,
            "avoidance_target_mass": float(view.visits[accepted].sum() / iteration),
        }
        newly_discovered = discovery is None and line["evaluated"]
        if newly_discovered:
            discovery = {**point, "witness": line}
        flags = {
            "avoid": avoids,
            "q_prefers_avoid": gap is not None and gap > 1e-8,
            "avoid_after_discovery": avoids and discovery is not None,
            "q_prefers_avoid_after_discovery": (
                gap is not None and gap > 1e-8 and discovery is not None
            ),
        }
        for name, enabled in flags.items():
            if enabled:
                first.setdefault(name, point)
                streak.setdefault(name, point)
            else:
                streak.pop(name, None)
        state = (avoids, flags["q_prefers_avoid"], discovery is not None)
        if state != previous:
            changes.append({**point, **flags})
            previous = state
        if iteration & (iteration - 1) == 0 or iteration == budget or newly_discovered:
            snapshots.append(
                {**point, **diagnostics.root_snapshot(root), "witness": line}
            )

    root = mcts.run_mcts(
        position.state,
        counted,
        budget,
        config=config,
        boundary_evaluator=boundary,
        rng=np.random.default_rng(seed),
        observer=observe,
    )
    assert root.visit_count == budget
    assert sum(c.visit_count for c in root.children.values()) == budget
    return {
        "case": position.id,
        "method": method,
        "seed": seed,
        "budget": budget,
        "discovery": discovery,
        "first": first,
        "stable_through_budget": streak,
        "transitions": changes,
        "snapshots": snapshots,
        "seconds": time.perf_counter() - start,
        "evaluator_calls": counted.calls,
        "evaluator_states": counted.states,
        "boundary_samples": boundary.states,
    }


def probe(
    checkpoint_plan: Path,
    output: Path,
    workers: int = 8,
    budget: int = 8192,
    seeds: int = 4,
    allow_dirty: bool = False,
):
    if min(workers, budget, seeds) < 1:
        raise ValueError("Workers, budget and seeds must be positive")
    repository = Path(__file__).resolve().parent
    fixtures = repository / "configs/benchmarks/search_positions.json"
    checkpoint = json.loads(checkpoint_plan.read_text())["checkpoints"]["candidate12"]
    assert runs.file_digest(Path(checkpoint["path"])) == checkpoint["sha256"]
    cases = [p for p, _ in targets.positions(fixtures) if p.id.startswith("giveaway_")]
    assert len(cases) == 2
    methods = {
        f"{chance}_{prior}": dataclasses.replace(
            BASE,
            after_state_evaluate_all_children=chance == "enumerated",
            uniform_group_priors=prior == "uniform",
        )
        for chance in ("sampled", "enumerated")
        for prior in ("learned", "uniform")
    }
    manifest = {
        "status": "running",
        "started_at": runs.utc_now(),
        "git": runs.git_provenance(repository, allow_dirty=allow_dirty),
        "script_sha256": runs.file_digest(Path(__file__)),
        "fixtures_sha256": runs.file_digest(fixtures),
        "checkpoint": checkpoint,
        "methods": {
            name: dataclasses.asdict(config) for name, config in methods.items()
        },
        "workers": workers,
        "threads_per_worker": 1,
        "budget": budget,
        "seeds": list(range(seeds)),
        "definitions": {
            "discovery": "First evaluation of the full GIVE/TAKE/CLEAR witness edge; enumerated mode evaluates every ordinary outcome on that edge.",
            "avoid": "Configured temperature-zero visit policy selects a non-giveaway action.",
            "q_prefers_avoid": "Best evaluated non-giveaway Q exceeds best evaluated giveaway Q by more than 1e-8; not a claim of true optimality.",
            "stable_through_budget": "Start of the final uninterrupted true interval, censored at this trial's budget; absent if false at the end.",
            "work": "Evaluated state count includes root and continuing-round network evaluations; time includes per-iteration diagnostics and concurrent-worker contention.",
        },
    }
    tasks = [
        (position, method, config, seed, budget)
        for position in cases
        for seed in range(seeds)
        for method, config in methods.items()
    ]
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        with (
            ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=targets.initialize,
                initargs=({"candidate12": checkpoint["path"]},),
            ) as pool,
            (output / "trials.jsonl").open("w") as stream,
        ):
            futures = [pool.submit(trial, task) for task in tasks]
            for number, future in enumerate(as_completed(futures), 1):
                stream.write(json.dumps(future.result(), allow_nan=False) + "\n")
                stream.flush()
                print(f"Completed {number}/{len(tasks)} chance probes", flush=True)
        manifest.update(status="completed", finished_at=runs.utc_now())
    except BaseException as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return output


if __name__ == "__main__":
    typer.run(probe)
