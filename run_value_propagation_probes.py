"""Status: experimental
Purpose: Separate neural foresight, continuation policy and search value backup.
Promote when: Value-consistency checks become recurring search validation.
"""

import dataclasses
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import typer

import run_search_target_probes as targets
from skyjo.engine import game
from skyjo.experiments import contestants, runs
from skyjo.experiments import search_diagnostics as diagnostics
from skyjo.experiments.learning_loop_diagnostic import BASE
from skyjo.experiments.search_positions import (
    Expectation,
    Position,
    Witness,
    witness_states,
)
from skyjo.learning.predictor import LocalPredictor
from skyjo.search import mcts
from skyjo.search.evaluator import Prediction


class Preview:
    """Inject only this line's public-state continuation evidence; keep root intact."""

    def __init__(self, evaluator, position, method):
        self.evaluator = evaluator
        self.method = method
        self.prefix, outcomes = witness_states(position, position.witnesses[0])
        self.after = sum(
            weight * prediction.value
            for (_, weight), prediction in zip(
                outcomes,
                evaluator.evaluate([state for state, _ in outcomes]),
                strict=True,
            )
        )
        self.actions = {
            game.hash_skyjo(s): a
            for s, a in zip(
                self.prefix[1:], position.witnesses[0].actions[1:], strict=True
            )
        }
        self.hits = 0

    def evaluate(self, states):
        result = self.evaluator.evaluate(states)
        for i, state in enumerate(states):
            action = self.actions.get(game.hash_skyjo(state))
            if action is None:
                continue
            self.hits += 1
            value, policy = result[i].value, result[i].policy
            if self.method in {"value_preview", "both_preview"}:
                value = self.after.copy()
            if self.method in {"policy_preview", "both_preview"}:
                policy = np.zeros_like(policy)
                policy[action] = 1
            result[i] = Prediction(value, policy)
        return result


def line_nodes(root, actions):
    result = []
    node = root
    for depth, action in enumerate(actions):
        if not isinstance(node, mcts.DecisionStateNode) or not node.is_expanded:
            break
        group = next(g for g in node.action_groups.members if action in g)
        if any(child.visit_count for child in node.children.values()):
            snapshot = diagnostics.root_snapshot(node)
        else:
            snapshot = {
                "action": None,
                "groups": [],
                "value": node.state_value.tolist(),
            }
        result.append(
            {
                "depth": depth,
                "actor": game.get_player(node.state),
                "visits": node.visit_count,
                "model_value": node.model_prediction.value.tolist(),
                "backed_value": node.state_value.tolist(),
                "line_action": action,
                **snapshot,
            }
        )
        if len(group) > 1 and depth < len(actions) - 1:
            raise ValueError("Ambiguous witness prefix symmetry")
        node = node.children[group[0]]
    return result


_evaluator = None


def initialize(checkpoint):
    global _evaluator
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _evaluator = LocalPredictor(contestants.load_model(Path(checkpoint)), 512)


def trial(task):
    position, reference, method, seed, budget = task
    if reference or not position.witnesses:
        config = (
            dataclasses.replace(BASE, valuation="best_response")
            if method == "best_response"
            else BASE
        )
        measured = diagnostics.run_search(
            position, _evaluator, budget, seed, config=config
        )
        return {
            "case": position.id,
            "exact": reference is not None,
            "method": method,
            "seed": seed,
            "snapshots": [
                targets.measure_snapshot(s, position, reference)
                for s in measured["trace"]
                if s["iteration"] in {32, 256}
            ],
            "witnesses": [],
        }
    evaluator = (
        Preview(_evaluator, position, method)
        if method.endswith("preview")
        else _evaluator
    )
    config = (
        dataclasses.replace(BASE, valuation="best_response")
        if method == "best_response"
        else BASE
    )
    records = []
    first = None
    previous_root = None
    previous_snapshot = None

    def observe(root):
        nonlocal first, previous_root, previous_snapshot
        if root is not previous_root:
            records.clear()
            first = None
            previous_snapshot = None
            previous_root = root
        witness = diagnostics.line_snapshot(root, position.witnesses[0].actions)
        discovered = first is None and witness.get("evaluated")
        if discovered:
            first = root.visit_count
        if discovered or root.visit_count in {32, 128, 256, 512, 1024, 2048}:
            snapshot = {
                "iteration": root.visit_count,
                **diagnostics.root_snapshot(root),
            }
            measured = targets.measure_snapshot(
                snapshot, position, None, previous_snapshot
            )
            previous_snapshot = snapshot
            records.append(
                {
                    **measured,
                    "line": line_nodes(root, position.witnesses[0].actions),
                    "witness": witness,
                }
            )

    mcts.run_mcts(
        position.state,
        evaluator,
        budget,
        config=config,
        rng=np.random.default_rng(seed),
        observer=observe,
    )
    return {
        "case": position.id,
        "exact": False,
        "method": method,
        "seed": seed,
        "first_clear_evaluated": first,
        "preview_hits": evaluator.hits if isinstance(evaluator, Preview) else 0,
        "snapshots": records,
    }


def neural_probe(position, evaluator):
    line = position.witnesses[0]
    prefix, outcomes = witness_states(position, line)
    after = sum(
        w * p.value
        for (_, w), p in zip(
            outcomes, evaluator.evaluate([s for s, _ in outcomes]), strict=True
        )
    )
    stages = []
    for state, line_action in zip(prefix, line.actions, strict=True):
        prediction = evaluator.evaluate([state])[0]
        actor = game.get_player(state)
        action_values = []
        for action in game.get_actions(state):
            children = game.chance_outcomes(state, int(action))
            if any(game.get_round_over(s) for s, _ in children):
                raise ValueError("This neural one-ply probe cannot cross round end")
            expected = sum(
                w * p.value
                for (_, w), p in zip(
                    children, evaluator.evaluate([s for s, _ in children]), strict=True
                )
            )
            action_values.append(
                {
                    "action": int(action),
                    "value": expected.tolist(),
                    "prior": float(prediction.policy[action]),
                }
            )
        stages.append(
            {
                "actor": actor,
                "line_action": line_action,
                "raw_value": prediction.value.tolist(),
                "line_prior": float(prediction.policy[line_action]),
                "one_ply_best_action": max(
                    action_values, key=lambda row: row["value"][actor]
                )["action"],
                "actions": action_values,
            }
        )
    return {
        "case": position.id,
        "stages": stages,
        "after_clear_value": after.tolist(),
        "interpretation": "Values in fixed-seat order; one-ply expectations enumerate ordinary chance exactly but use the frozen neural value, not an outcome oracle.",
    }


def turn_completion_probe(state, evaluator):
    """Enumerate one player's draw/take and best placement using neural leaves."""
    actor = game.get_player(state)
    if list(game.get_actions(state)) != [game.MASK_DRAW, game.MASK_TAKE]:
        raise ValueError("Turn-completion probe requires a draw/take decision")
    rows = []
    for action in [game.MASK_DRAW, game.MASK_TAKE]:
        first = game.chance_outcomes(state, action)
        leaves, mapping = [], []
        for child, weight in first:
            assert game.get_player(child) == actor
            placements = []
            for placement in game.get_actions(child):
                outcomes = game.chance_outcomes(child, int(placement))
                if any(game.get_round_over(s) for s, _ in outcomes):
                    raise ValueError("Turn-completion probe cannot cross round end")
                indices = []
                for leaf, probability in outcomes:
                    indices.append((len(leaves), probability))
                    leaves.append(leaf)
                placements.append((int(placement), indices))
            mapping.append((weight, placements))
        predictions = evaluator.evaluate(leaves)
        choices = []
        for weight, placements in mapping:
            values = [
                (a, sum(p * predictions[i].value[actor] for i, p in indices))
                for a, indices in placements
            ]
            best, value = max(values, key=lambda pair: pair[1])
            choices.append(
                {"weight": weight, "best_action": best, "value": float(value)}
            )
        rows.append(
            {
                "action": action,
                "turn_completion_value": float(
                    sum(c["weight"] * c["value"] for c in choices)
                ),
                "enumerated_leaf_states": len(leaves),
                "placement_choices": choices,
            }
        )
    return rows


def probe(checkpoint_plan: Path, output: Path, workers: int = 8, smoke: bool = False):
    root = Path(__file__).resolve().parent
    all_cases = targets.positions(root / "configs/benchmarks/search_positions.json")
    selected = [(p, ref) for p, ref in all_cases if p.witnesses]
    exact = [(p, ref) for p, ref in all_cases if ref]
    plan = json.loads(checkpoint_plan.read_text())
    checkpoints = plan["checkpoints"]
    methods = [
        "base",
        "best_response",
        "value_preview",
        "policy_preview",
        "both_preview",
    ]
    if smoke:
        selected, exact, methods = selected[:1], exact[:1], ["base", "both_preview"]
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running",
        "started_at": runs.utc_now(),
        "git": runs.git_provenance(root, allow_dirty=smoke),
        "script_sha256": runs.file_digest(Path(__file__)),
        "checkpoints": checkpoints,
        "workers": workers,
        "smoke": smoke,
        "interventions": "Preview replaces only non-root witness-prefix values and/or policies. Value comes from exact chance-weighted frozen-network estimates after the known clear, not true outcomes. Policies choose the specified continuation with probability1. No model parameters change.",
    }
    for info in checkpoints.values():
        assert runs.file_digest(Path(info["path"])) == info["sha256"]
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        initialize(checkpoints["candidate12"]["path"])
        probes = []
        completed_turns = []
        for name in (
            ["candidate12"] if smoke else ["candidate12", "before4", "historical87"]
        ):
            evaluator = LocalPredictor(
                contestants.load_model(Path(checkpoints[name]["path"])), 512
            )
            for p, _ in selected:
                probes.append({"model": name, **neural_probe(p, evaluator)})
                if p.id.startswith("giveaway"):
                    state = witness_states(p, p.witnesses[0])[0][1]
                elif p.id == "own_take_clear":
                    state = p.state
                else:
                    continue
                completed_turns.append(
                    {
                        "model": name,
                        "case": p.id,
                        "actions": turn_completion_probe(state, evaluator),
                    }
                )
            if not smoke:
                control = next(
                    p for p, _ in all_cases if p.id == "own_no_clear_control"
                )
                completed_turns.append(
                    {
                        "model": name,
                        "case": control.id,
                        "actions": turn_completion_probe(control.state, evaluator),
                    }
                )
        (output / "neural.json").write_text(json.dumps(probes, indent=2) + "\n")
        (output / "turn-completion.json").write_text(
            json.dumps(completed_turns, indent=2) + "\n"
        )
        tasks = [
            (p, None, method, seed, 32 if smoke else 2048)
            for p, _ in selected
            for method in methods
            for seed in range(1 if smoke else 4)
        ]
        tasks += [
            (p, ref, "best_response", seed, 32 if smoke else 256)
            for p, ref in exact
            for seed in range(1 if smoke else 2)
        ]
        if not smoke:
            for p, _ in selected:
                prefix, _ = witness_states(p, p.witnesses[0])
                for depth, state in enumerate(prefix[1:], 1):
                    action = p.witnesses[0].actions[depth]
                    reroot = Position(
                        f"{p.id}/reroot{depth}",
                        "Fresh search at continuation",
                        (),
                        state,
                        Expectation(
                            "prefer",
                            (action,),
                            "Known clear continuation; not an optimality oracle",
                        ),
                        (Witness("clear", p.witnesses[0].actions[depth:]),),
                    )
                    tasks += [(reroot, None, "base", seed, 256) for seed in range(4)]
        manifest["planned_trials"] = len(tasks)
        with (
            ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=initialize,
                initargs=(checkpoints["candidate12"]["path"],),
            ) as pool,
            (output / "trials.jsonl").open("w") as stream,
        ):
            futures = [pool.submit(trial, task) for task in tasks]
            for i, future in enumerate(as_completed(futures), 1):
                stream.write(json.dumps(future.result(), allow_nan=False) + "\n")
                stream.flush()
                if i % 20 == 0 or i == len(tasks):
                    print(f"Completed {i}/{len(tasks)} value probes", flush=True)
        manifest.update(status="completed", finished_at=runs.utc_now())
    except BaseException as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output


if __name__ == "__main__":
    typer.run(probe)
