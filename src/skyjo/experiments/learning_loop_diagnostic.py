"""Bounded experiments separating tactical search, target quality, and fitting."""

from __future__ import annotations

import dataclasses
import json
import multiprocessing as mp
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from skyjo.engine import game, symmetry
from skyjo.experiments import contestants, runs
from skyjo.experiments import search_diagnostics as diagnostics
from skyjo.experiments.search_positions import Expectation, Position, load_suite
from skyjo.experiments.search_strategy_benchmark import exact_action_values, load_cases
from skyjo.learning import predictor
from skyjo.search import runner
from skyjo.search.configuration import SearchConfig
from skyjo.search.symmetry import ActionGroups
from skyjo.simulation import play
from skyjo.simulation.jobs import GameRandomStreams, derive_game_seed

BASE = SearchConfig(
    c_puct=1.0,
    fpu_mode="parent",
    fpu_reduction=0.0,
    valuation="current_mixture",
    target_strategy="visits",
    merge_symmetric_actions=True,
    dirichlet_epsilon=0.0,
    boundary_samples=1,
)
METHODS = {
    "base": BASE,
    "uniform": dataclasses.replace(BASE, uniform_group_priors=True),
    "mean_return": dataclasses.replace(BASE, valuation="mean_return"),
    "selfplay_noise": dataclasses.replace(BASE, dirichlet_epsilon=0.25),
}
_plan = None
_agents = {}


def initialize(plan):
    global _plan, _agents
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _plan, _agents = plan, {}


def agent(name):
    if name not in _agents:
        _agents[name] = contestants.prepare(
            contestants.ContestantConfig(
                checkpoint=_plan["checkpoints"][name],
                iterations=32,
                search=BASE,
            )
        )
    return _agents[name]


def fresh_positions(index):
    seed = derive_game_seed(_plan["seed"], index, 0x44494147)
    streams = GameRandomStreams.from_seed(seed)
    players = [
        agent("initial").player(np.random.default_rng(int(s)))
        for s in streams.search.integers(0, 2**32, size=2)
    ]
    history = play.play_round(
        players, environment_rng=streams.environment, action_rng=streams.actions
    )
    result = []
    for family in ("draw_take", "placement"):
        eligible = [
            (i, entry.state)
            for i, entry in enumerate(history[:-1])
            if bool(game.actions(entry.state)[game.MASK_DRAW])
            == (family == "draw_take")
            and not game.actions(entry.state)[: game.MASK_DRAW].any()
        ]
        if not eligible:
            continue
        for label, offset in (
            ("early", len(eligible) // 4),
            ("late", 3 * len(eligible) // 4),
        ):
            frame, state = eligible[offset]
            result.append(
                Position(
                    f"fresh-{index:02}-{family}-{label}",
                    "Fresh diagnostic decision",
                    ("fresh", family, label),
                    state,
                    Expectation(
                        "unscored", (), "No oracle; evaluate paired continuations."
                    ),
                    source={
                        "seed": seed,
                        "game_index": index,
                        "frame": frame,
                        "generator": "initial checkpoint, search32, first round",
                    },
                )
            )
    return result


def oracle_metrics(reference, policy, target, choice, q_choice):
    """Exact references use the same full-game win-credit objective as the model."""
    if not reference:
        return {}
    best = max(reference.values())
    expected = lambda p: sum(float(p[a]) * v for a, v in reference.items())
    return {
        "policy_expected_regret": best - expected(policy),
        "target_expected_regret": best - expected(target),
        "choice_regret": best - reference[choice],
        "q_choice_regret": best - reference[q_choice],
    }


def trial(job):
    position, reference, name, method, budget, seed = job
    evaluator = agent(name).evaluator
    config = METHODS[method]
    prediction = evaluator.evaluate([position.state])[0]
    raw_choice = int(
        predictor.PolicyPlayer(evaluator)
        .get_action_probabilities(position.state)
        .argmax()
    )
    if "tactical" in position.tags:
        result = diagnostics.run_search(
            position, evaluator, budget, seed, config=config
        )
        groups = ActionGroups(tuple(tuple(g["actions"]) for g in result["groups"]))
        target = groups.policy_from_visits(
            {g["actions"][0]: g["visits"] for g in result["groups"]}
        )
        if method == "base" and budget == _plan["budgets"][0] and seed == 0:
            result["neural_line"] = diagnostics.model_diagnostic(position, evaluator)
    else:
        searched = runner.search(
            position.state,
            evaluator,
            budget,
            config=config,
            temperature=0.0,
            rng=np.random.default_rng(seed),
        )
        result = diagnostics.root_snapshot(searched.roots[0])
        target = searched.training_policy
    target = symmetry.symmetrize_policy_target(position.state, target)
    q_choice = max(
        (g for g in result["groups"] if g["q"] is not None), key=lambda g: g["q"]
    )["actions"][0]
    return {
        "case": position.id,
        "tags": position.tags,
        "model": name,
        "method": method,
        "budget": budget,
        "seed": seed,
        "raw_choice": raw_choice,
        "raw_policy": prediction.policy.tolist(),
        "target": target.tolist(),
        "q_choice": q_choice,
        "oracle": oracle_metrics(
            reference, prediction.policy, target, result["action"], q_choice
        ),
        **result,
    }


def continue_action(state, action, prepared, seed):
    """Force one legal action, then use the same frozen policy in both seats."""
    streams = GameRandomStreams.from_seed(seed)
    actor = game.get_player(state)
    players = [
        prepared.player(np.random.default_rng(int(s)))
        for s in streams.search.integers(0, 2**32, size=state.players)
    ]
    state = game.apply_action(state, action, rng=streams.environment)
    if not game.get_round_over(state):
        history = play.play_round(
            players,
            start_state=state,
            environment_rng=streams.environment,
            action_rng=streams.actions,
        )
        state = history[-1].state
    round_scores = game.get_fixed_perspective_round_scores(state)
    if game.get_game_over(state):
        scores = game.get_fixed_perspective_game_scores(state)
    else:
        state = game.start_next_round(state, rng=streams.environment)
        scores = np.asarray(
            play.play_game(
                players,
                start_state=state,
                environment_rng=streams.environment,
                action_rng=streams.actions,
            ).final_scores
        )
    winners = scores == min(scores)
    return {
        "win_credit": float(winners[actor] / winners.sum()),
        "round_margin": int(round_scores[1 - actor] - round_scores[actor]),
        "game_margin": int(scores[1 - actor] - scores[actor]),
    }


def continuation(job):
    position, actions, seed = job
    return {
        "case": position.id,
        "seed": seed,
        "actions": {
            str(a): continue_action(position.state, a, agent("historical"), seed)
            for a in actions
        },
    }


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def collect(pool, function, jobs, path):
    rows = []
    futures = [pool.submit(function, job) for job in jobs]
    with path.open("x") as stream:
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            if len(rows) % 20 == 0 or len(rows) == len(jobs):
                print(f"{path.stem}: {len(rows)}/{len(jobs)}", flush=True)
    return rows


def search_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["model"], row["method"], row["budget"]].append(row)
    result = []
    for (model, method, budget), selected in sorted(groups.items()):
        exact = [r for r in selected if r["oracle"]]
        result.append(
            {
                "model": model,
                "method": method,
                "budget": budget,
                "trials": len(selected),
                "exact_trials": len(exact),
                "exact_means": {
                    k: float(np.mean([r["oracle"][k] for r in exact]))
                    for k in exact[0]["oracle"]
                }
                if exact
                else {},
                "tactical": [
                    {
                        "case": case,
                        "accepted": sum(
                            r["accepted"] is True for r in selected if r["case"] == case
                        ),
                        "trials": sum(r["case"] == case for r in selected),
                    }
                    for case in sorted(
                        {r["case"] for r in selected if "tactical" in r["tags"]}
                    )
                ],
            }
        )
    return result


def run(plan_path: Path, output: Path, *, allow_dirty=False):
    from skyjo.experiments.learning_loop_fitting import fit_diagnostics

    plan = json.loads(plan_path.read_text())
    root = Path(__file__).resolve().parents[3]
    provenance = runs.git_provenance(root, allow_dirty=allow_dirty)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", plan)
    manifest = {
        "status": "running",
        "started_at": runs.utc_now(),
        "git": provenance,
        "checkpoints": {
            k: {"path": v, "sha256": runs.file_digest(Path(v))}
            for k, v in plan["checkpoints"].items()
        },
        "methods": {k: dataclasses.asdict(v) for k, v in METHODS.items()},
        "interpretation": "Exact cases are previously used diagnostic controls; only fresh positions are newly held out. Continuations estimate frozen-policy outcomes, not optimal action values. Fixed-target fitting is not a strength experiment.",
    }

    def stage(name):
        manifest["stage"] = name
        write_json(output / "manifest.json", manifest)
        print(f"Stage: {name}", flush=True)

    try:
        stage("prepare")
        tactical = [
            dataclasses.replace(p, tags=(*p.tags, "tactical"))
            for p in load_suite(root / "data/search-positions")
        ]
        _, cases = load_cases(root / "configs/benchmarks/search_positions.json")
        positions = tactical[:]
        references = {}
        for c in cases:
            if c.cohort == "tactical":
                continue
            positions.append(
                Position(
                    c.name,
                    "Frozen diagnostic control",
                    (c.cohort,),
                    c.state,
                    Expectation(
                        "unscored",
                        (),
                        "Scored separately against exact reference where available.",
                    ),
                )
            )
            if c.cohort.startswith("exact"):
                references[c.name] = exact_action_values(c.state)
        with ProcessPoolExecutor(
            max_workers=plan["workers"],
            mp_context=mp.get_context("spawn"),
            initializer=initialize,
            initargs=(plan,),
        ) as pool:
            stage("fresh_positions")
            for batch in pool.map(fresh_positions, range(plan["fresh_games"])):
                positions.extend(batch)
            folder = output / "positions"
            folder.mkdir()
            for p in positions:
                write_json(folder / f"{p.id}.json", p.to_dict())
            write_json(output / "exact_references.json", references)
            stage("search_targets_and_tactical_traces")
            jobs = [
                (p, references.get(p.id, {}), name, method, budget, seed)
                for p in positions
                for name in ("initial", "historical", "trained")
                for method in METHODS
                for budget in plan["budgets"]
                for seed in range(plan["search_seeds"])
                if method == "base"
                or (
                    "tactical" in p.tags
                    and method in ("uniform", "mean_return")
                    and budget == max(plan["budgets"])
                )
                or (method == "selfplay_noise" and name == "trained" and budget == 256)
            ]
            rows = collect(pool, trial, jobs, output / "search.jsonl")
            write_json(output / "search-summary.json", search_summary(rows))
            stage("fixed_target_fitting")
            fit_diagnostics(plan, output, positions, references, rows)
            stage("paired_continuations")
            candidates = []
            for p in positions:
                giveaway = "tactical" in p.tags and p.expectation.kind == "avoid"
                if "fresh" not in p.tags and not giveaway:
                    continue
                selected = [
                    r
                    for r in rows
                    if r["case"] == p.id
                    and r["model"] == "trained"
                    and r["method"] == "base"
                ]
                choices = {"policy": selected[0]["raw_choice"]}
                for budget in plan["budgets"]:
                    counts = Counter(
                        r["action"] for r in selected if r["budget"] == budget
                    )
                    choices[str(budget)] = min(
                        counts, key=lambda action: (-counts[action], action)
                    )
                if giveaway:
                    for label, name, method, field in (
                        ("highest_q", "trained", "base", "q_choice"),
                        ("initial", "initial", "base", "action"),
                        ("uniform", "trained", "uniform", "action"),
                    ):
                        counts = Counter(
                            r[field]
                            for r in rows
                            if r["case"] == p.id
                            and r["model"] == name
                            and r["method"] == method
                            and r["budget"] == max(plan["budgets"])
                        )
                        choices[label] = min(
                            counts, key=lambda action: (-counts[action], action)
                        )
                actions = sorted(set(choices.values()))
                if len(actions) > 1:
                    candidates.append((p, actions, choices))
            candidates = sorted(
                candidates, key=lambda x: ("tactical" not in x[0].tags, x[0].id)
            )[: plan["continuation_cases"]]
            write_json(
                output / "continuation-candidates.json",
                [{"case": p.id, "choices": c, "actions": a} for p, a, c in candidates],
            )
            jobs = [
                (p, actions, derive_game_seed(plan["seed"], i, 0x434F4E54))
                for p, actions, _ in candidates
                for i in range(plan["continuation_seeds"])
            ]
            continued = collect(
                pool, continuation, jobs, output / "continuations.jsonl"
            )
            summaries = []
            for p, actions, choices in candidates:
                subset = sorted(
                    (r for r in continued if r["case"] == p.id), key=lambda r: r["seed"]
                )
                for label, action in choices.items():
                    differences = np.array(
                        [
                            [
                                r["actions"][str(action)][metric]
                                - r["actions"][str(choices["policy"])][metric]
                                for metric in (
                                    "win_credit",
                                    "round_margin",
                                    "game_margin",
                                )
                            ]
                            for r in subset
                        ]
                    )
                    rng = np.random.default_rng(plan["seed"])
                    ci = np.quantile(
                        differences[
                            rng.integers(0, len(subset), (10000, len(subset)))
                        ].mean(axis=1),
                        [0.025, 0.975],
                        axis=0,
                    )
                    summaries.append(
                        {
                            "case": p.id,
                            "choice": label,
                            "action": action,
                            "seeds": len(subset),
                            "metrics": ["win_credit", "round_margin", "game_margin"],
                            "difference_vs_policy": differences.mean(axis=0).tolist(),
                            "paired_95_interval": ci.tolist(),
                        }
                    )
            write_json(output / "continuation-summary.json", summaries)
        manifest.update(status="completed", completed_at=runs.utc_now())
    except BaseException as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        write_json(output / "manifest.json", manifest)
    return output
