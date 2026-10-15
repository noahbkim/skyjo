"""Fresh-root, paired tactical diagnostics; labels are hypotheses, not oracles."""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import json
import time
import tomllib
from collections import defaultdict
from pathlib import Path

import numpy as np

from skyjo.engine import game
from skyjo.experiments import runs
from skyjo.experiments.search_positions import Position, load_suite, witness_states
from skyjo.search import mcts
from skyjo.search.evaluator import NextDealEvaluator
from skyjo.search.strategies import PolicyInput


class CountingEvaluator:
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.calls = self.states = 0

    def evaluate(self, states):
        self.calls += bool(states)
        self.states += len(states)
        return self.evaluator.evaluate(states)


class CountingBoundary:
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.calls = self.states = 0

    def evaluate(self, states, rng):
        self.calls += bool(states)
        self.states += len(states)
        return self.evaluator.evaluate(states, rng)


def model_diagnostic(position: Position, evaluator) -> dict:
    """Exact ordinary-outcome neural probe, outside search work/timing counters."""
    prediction = evaluator.evaluate([position.state])[0]
    action = int(prediction.policy.argmax())
    lines = []
    for line in position.witnesses:
        prefix, outcomes = witness_states(position, line)
        predictions = evaluator.evaluate(prefix + [s for s, _ in outcomes])
        lines.append(
            {
                "name": line.name,
                "actions": list(line.actions),
                "actors": [game.get_player(s) for s in prefix],
                "values": [p.value.tolist() for p in predictions[: len(prefix)]],
                "action_priors": [
                    float(p.policy[a]) for p, a in zip(predictions, line.actions)
                ],
                "after_value": sum(
                    weight * p.value
                    for (_, weight), p in zip(
                        outcomes, predictions[len(prefix) :], strict=True
                    )
                ).tolist(),
                "outcomes": len(outcomes),
            }
        )
    return {
        "case": position.id,
        "action": action,
        "accepted": position.expectation.accepts(action),
        "policy": prediction.policy.tolist(),
        "value": prediction.value.tolist(),
        "witnesses": lines,
    }


def selected_action(root) -> int:
    """Use the configured acting policy, independently of learning targets."""
    view = root.view()
    context = root.context.root_session.finish(view)
    probabilities = root.context.components.action_policy.probabilities(
        PolicyInput(view, context), 0.0
    )
    return int(probabilities.argmax())


def _validate_traced_config(config):
    if config.replicas != 1:
        raise ValueError(
            "Per-iteration diagnostics require replicas=1; use "
            "run_search_strategy_benchmark.py for replica comparisons"
        )


def root_snapshot(root) -> dict:
    seat = game.get_player(root.state)
    return {
        "action": selected_action(root),
        "value": root.state_value.tolist(),
        "groups": [
            {
                "actions": list(group),
                "visits": root.children[group[0]].visit_count,
                "prior": float(root.model_action_priors[group[0]]),
                "search_prior": float(root.action_priors[group[0]]),
                "q": float(root.children[group[0]].state_value[seat])
                if root.children[group[0]].has_value_estimate
                else None,
            }
            for group in root.action_groups.members
        ],
    }


def line_snapshot(root, actions) -> dict:
    """Inspect expanded/visited edges without treating allocation as discovery.

    A grouped *prefix* can change slot coordinates. Without a symmetry mapping
    such a witness is explicitly unavailable, never counted as undiscovered.
    """
    node = root
    edges = []
    for index, action in enumerate(actions):
        if not isinstance(node, mcts.DecisionStateNode) or not node.is_expanded:
            break
        group = next((g for g in node.action_groups.members if action in g), None)
        if group is None:
            return {"status": "unavailable", "edges": edges}
        if len(group) > 1 and index < len(actions) - 1:
            return {"status": "symmetry_ambiguous", "edges": edges}
        node = node.children[group[0]]
        edges.append(
            {
                "actions": list(group),
                "visits": node.visit_count,
                "evaluated": bool(node.has_value_estimate),
                "value": node.state_value.tolist() if node.has_value_estimate else None,
            }
        )
    complete = len(edges) == len(actions)
    return {
        "status": "available",
        "edges": edges,
        "evaluated": complete and edges[-1]["evaluated"],
        "visited": complete and edges[-1]["visits"] > 0,
    }


def run_search(
    position: Position,
    evaluator,
    iterations: int,
    seed: int,
    *,
    config: mcts.SearchConfig = mcts.DEFAULT_SEARCH_CONFIG,
    boundary_evaluator=None,
    search=mcts.run_mcts,
) -> dict:
    """Run an MCTS-compatible search callable with read-only per-iteration tracing."""
    _validate_traced_config(config)
    counted = CountingEvaluator(evaluator)
    boundary = CountingBoundary(boundary_evaluator or NextDealEvaluator(counted))
    first_evaluated = {line.name: None for line in position.witnesses}
    first_visited = dict(first_evaluated)
    first_accepted = None
    last_failure = 0
    observed_iterations = 0
    observed_root = None
    trace = []

    def observe(root):
        nonlocal first_accepted, last_failure, observed_iterations
        nonlocal observed_root
        if root is not observed_root:
            # A symmetry-safety fallback discards the earlier tree. Keep actual
            # evaluator work, but report discovery and recovery only for this tree.
            observed_root = root
            observed_iterations = 0
            first_accepted = None
            last_failure = 0
            first_evaluated.update(dict.fromkeys(first_evaluated))
            first_visited.update(dict.fromkeys(first_visited))
            trace.clear()
        iteration = root.visit_count
        if iteration != observed_iterations + 1:
            raise ValueError("Search observer must receive every iteration in order")
        observed_iterations = iteration
        selected = selected_action(root)
        accepted = position.expectation.accepts(selected)
        if accepted and first_accepted is None:
            first_accepted = iteration
        if accepted is False:
            last_failure = iteration
        for line in position.witnesses:
            snapshot = line_snapshot(root, line.actions)
            if snapshot.get("evaluated") and first_evaluated[line.name] is None:
                first_evaluated[line.name] = iteration
            if snapshot.get("visited") and first_visited[line.name] is None:
                first_visited[line.name] = iteration
        # Keep logarithmic snapshots; discovery and recovery still checked every visit.
        if iteration & (iteration - 1) == 0 or iteration == iterations:
            trace.append({"iteration": iteration, **root_snapshot(root)})

    start = time.perf_counter()
    root = search(
        position.state,
        counted,
        iterations,
        config=config,
        boundary_evaluator=boundary,
        rng=np.random.default_rng(seed),
        observer=observe,
    )
    elapsed = time.perf_counter() - start
    if (
        root.visit_count != iterations
        or not trace
        or trace[-1]["iteration"] != iterations
    ):
        raise ValueError(
            "Search adapter must notify observer after every completed iteration"
        )
    final = root_snapshot(root)
    accepted = position.expectation.accepts(final["action"])
    return {
        **final,
        "accepted": accepted,
        "first_accepted_iteration": first_accepted,
        "stable_accepted_from": last_failure + 1 if accepted else None,
        "seconds": elapsed,
        "evaluator_calls": counted.calls,
        "evaluator_states": counted.states,
        "boundary_calls": boundary.calls,
        "boundary_samples": boundary.states,
        "effective_merge_symmetric_actions": root.effective_merge_symmetric_actions,
        "trace": trace,
        "witnesses": [
            {
                "name": line.name,
                "first_evaluated_iteration": first_evaluated[line.name],
                "first_visited_iteration": first_visited[line.name],
                **line_snapshot(root, line.actions),
            }
            for line in position.witnesses
        ],
    }


def summarize(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["case"], row["method"], row["iterations"]].append(row)
    return [
        {
            "case": case,
            "method": method,
            "iterations": budget,
            "trials": len(group),
            "accepted": sum(r["accepted"] is True for r in group),
            "scored": sum(r["accepted"] is not None for r in group),
            "median_stable_accepted_from": _observed_median(
                r["stable_accepted_from"] for r in group
            ),
            "median_evaluator_states": float(
                np.median([r["evaluator_states"] for r in group])
            ),
            "median_seconds": float(np.median([r["seconds"] for r in group])),
            "witnesses": {
                name: {
                    "evaluated": sum(w.get("evaluated", False) for w in witnesses),
                    "visited": sum(w.get("visited", False) for w in witnesses),
                    "available": sum(w["status"] == "available" for w in witnesses),
                    "median_first_evaluated_iteration": _observed_median(
                        w["first_evaluated_iteration"] for w in witnesses
                    ),
                }
                for name, witnesses in _witness_groups(group).items()
            },
        }
        for (case, method, budget), group in sorted(groups.items())
    ]


def _observed_median(values):
    observed = [value for value in values if value is not None]
    return float(np.median(observed)) if observed else None


def _witness_groups(rows):
    groups = defaultdict(list)
    for row in rows:
        for witness in row["witnesses"]:
            groups[witness["name"]].append(witness)
    return groups


def markdown_report(summary: list[dict]) -> str:
    lines = [
        "# Search position diagnostics",
        "",
        "Accepted means the annotated tactical expectation was met; it does not prove optimality.",
        "Budgets are independent fresh searches. Timings include tracing overhead.",
        "Iteration medians exclude undiscovered/unsuccessful trials; read them alongside the counts.",
        "",
        "| Case | Method | Iterations | Accepted / scored | Stable from (median) | Witness evaluated / available; first iteration (median) | NN states (median) |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in summary:
        witnesses = (
            "; ".join(
                f"{name}: {w['evaluated']}/{w['available']}; {w['median_first_evaluated_iteration'] if w['median_first_evaluated_iteration'] is not None else '—'}"
                for name, w in row["witnesses"].items()
            )
            or "—"
        )
        lines.append(
            f"| {row['case']} | {row['method']} | {row['iterations']} | "
            f"{row['accepted']}/{row['scored']} | {row['median_stable_accepted_from'] if row['median_stable_accepted_from'] is not None else '—'} | "
            f"{witnesses} | {row['median_evaluator_states']:g} |"
        )
    return "\n".join(lines) + "\n"


def _positive_integers(values, name, minimum):
    if (
        not values
        or len(values) != len(set(values))
        or any(type(v) is not int or v < minimum for v in values)
    ):
        raise ValueError(f"{name} must contain unique integers >= {minimum}")


def run_suite(
    *,
    suite: Path,
    checkpoint: Path,
    plan: Path,
    output: Path,
    allow_dirty: bool = False,
) -> Path:
    """Load weights once; append raw trials and preserve all inputs for comparison."""
    import torch

    from skyjo.experiments.contestants import load_model
    from skyjo.learning.boundary_inference import (
        ScoreBoundaryEvaluator,
        load_boundary_model,
    )
    from skyjo.learning.predictor import LocalPredictor

    positions = load_suite(suite)
    settings = tomllib.loads(plan.read_text())
    if (
        set(settings) - {"version", "iterations", "seeds", "methods"}
        or settings.get("version") != 1
    ):
        raise ValueError("Unsupported diagnostic plan")
    budgets, seeds = settings["iterations"], settings["seeds"]
    _positive_integers(budgets, "iterations", 1)
    _positive_integers(seeds, "seeds", 0)
    methods = []
    for raw in settings["methods"]:
        options = dict(raw)
        name = options.pop("name")
        config = mcts.SearchConfig(**options.pop("search", {}))
        _validate_traced_config(config)
        adapter = options.pop("adapter", "skyjo.search.mcts:run_mcts")
        module, function = adapter.split(":")
        search = getattr(importlib.import_module(module), function)
        boundary_path = options.pop("boundary_checkpoint", None)
        boundary_path = (
            (plan.parent / boundary_path).resolve() if boundary_path else None
        )
        if options:
            raise ValueError(f"Unknown method options: {sorted(options)}")
        methods.append((name, config, search, boundary_path))
    names = [m[0] for m in methods]
    if not names or len(set(names)) != len(names):
        raise ValueError("Methods must have unique names")
    repository = Path(__file__).resolve().parents[3]
    provenance = runs.git_provenance(repository, allow_dirty=allow_dirty)
    model = load_model(checkpoint)
    if any(p.state.players != model.players for p in positions):
        raise ValueError("Position player counts must match checkpoint")
    evaluator = LocalPredictor(model, max_batch_size=512)
    boundaries = {
        path: ScoreBoundaryEvaluator(load_boundary_model(path, model.players))
        for path in {m[3] for m in methods}
        if path is not None
    }
    # Refuse overwrites, including earlier partial runs.
    output.mkdir(parents=True, exist_ok=False)
    frozen = output / "positions"
    frozen.mkdir()
    for i, position in enumerate(positions):
        (frozen / f"{i:03}.json").write_text(
            json.dumps(position.to_dict(), allow_nan=False) + "\n"
        )
    (output / "plan.toml").write_bytes(plan.read_bytes())
    source_paths = {
        *repository.glob("src/skyjo/**/*.py"),
        repository / "run_search_diagnostics.py",
        *(Path(inspect.getfile(m[2])).resolve() for m in methods),
    }
    sources = output / "sources"
    sources.mkdir()
    source_manifest = {}
    for i, path in enumerate(sorted(source_paths)):
        copy = sources / f"{i:03}-{path.name}"
        copy.write_bytes(path.read_bytes())
        source_manifest[str(path)] = {
            "sha256": runs.file_digest(copy),
            "snapshot": str(copy.relative_to(output)),
        }
    manifest = {
        "format": "skyjo.search-diagnostics",
        "version": 1,
        "created_at": runs.utc_now(),
        "git": provenance,
        "status": "running",
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "sha256": runs.file_digest(checkpoint),
        },
        "boundary_checkpoints": {str(p): runs.file_digest(p) for p in boundaries},
        "plan": settings,
        "sources": source_manifest,
        "positions": {p.name: runs.file_digest(p) for p in frozen.glob("*.json")},
        "numpy": np.__version__,
        "torch": torch.__version__,
        "threads": 1,
    }

    def write_manifest():
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n"
        )

    write_manifest()
    threads = torch.get_num_threads()
    rows = []
    try:
        torch.set_num_threads(1)
        baselines = [model_diagnostic(p, evaluator) for p in positions]
        (output / "policy.json").write_text(
            json.dumps(baselines, indent=2, allow_nan=False) + "\n"
        )
        with (output / "trials.jsonl").open("w") as stream:
            for position in positions:
                for budget in budgets:
                    for seed in seeds:
                        for name, config, search, path in methods:
                            result = run_search(
                                position,
                                evaluator,
                                budget,
                                seed,
                                config=config,
                                boundary_evaluator=boundaries.get(path),
                                search=search,
                            )
                            row = {
                                "case": position.id,
                                "method": name,
                                "iterations": budget,
                                "seed": seed,
                                "search": dataclasses.asdict(config),
                                **result,
                            }
                            stream.write(json.dumps(row, allow_nan=False) + "\n")
                            stream.flush()
                            rows.append(row)
        summary = summarize(rows)
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n"
        )
        (output / "report.md").write_text(markdown_report(summary))
        manifest["status"] = "complete"
        manifest["trials"] = len(rows)
    except BaseException as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        torch.set_num_threads(threads)
        write_manifest()
    return output
