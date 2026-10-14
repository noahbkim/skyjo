"""Frozen tactical and exact-endgame comparisons of composable search strategies."""

from __future__ import annotations

import dataclasses
import json
import multiprocessing
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from skyjo.engine import game
from skyjo.experiments.runs import file_digest, git_provenance
from skyjo.experiments.terminal_benchmark import enumerate_one_hidden
from skyjo.search.mcts import SearchConfig


@dataclasses.dataclass(frozen=True)
class Case:
    name: str
    cohort: str
    state: game.Skyjo
    metadata: dict


def load_cases(path: Path) -> tuple[dict, list[Case]]:
    """Load compact public states; every included state is validated by the engine."""
    payload = json.loads(path.read_text())
    if (
        payload["format"] != "skyjo.search-benchmark-fixtures"
        or payload["version"] != 1
    ):
        raise ValueError("Unsupported search benchmark fixture format")
    cases = []
    for raw in payload["cases"]:
        data = dict(raw["state"])
        cards = np.asarray(data.pop("cards"), dtype=int)
        if cards.shape != (data["players"], game.ROW_COUNT, game.COLUMN_COUNT):
            raise ValueError("Fixture card layout differs from player count")
        table = np.zeros(
            (game.PLAYER_COUNT, game.ROW_COUNT, game.COLUMN_COUNT, game.FINGER_SIZE),
            dtype=np.int16,
        )
        table[: data["players"]] = np.eye(game.FINGER_SIZE, dtype=np.int16)[cards]
        state = game.Skyjo(
            table=table,
            **{
                k: np.array(v, dtype=np.int16) if k in {"game", "deck"} else v
                for k, v in data.items()
            },
        )
        game.validate(state)
        cases.append(
            Case(
                raw["name"],
                raw["cohort"],
                state,
                {k: v for k, v in raw.items() if k != "state"},
            )
        )
    return payload["source"], cases


def exact_action_values(state: game.Skyjo) -> dict[int, float]:
    """Integrate terminal cards and, if needed, the same player's final placement."""
    actor = game.get_player(state)
    if state.countdown == 1:
        result = {}
        for action in game.get_actions(state):
            values, terminal, weights = enumerate_one_hidden(state, int(action))
            if not terminal.all():
                raise ValueError("Exact benchmark reference requires terminal outcomes")
            result[int(action)] = float(weights @ values[:, actor])
        return result
    if state.countdown != 2 or list(game.get_actions(state)) != [
        game.MASK_DRAW,
        game.MASK_TAKE,
    ]:
        raise ValueError("Expected a final placement or final draw-or-take decision")
    prepared = game.prepare_draw_pile(state)
    if prepared.deck.sum() <= 1:
        raise ValueError("Exact draw reference needs two unseen cards")
    take = max(exact_action_values(game.apply_action(state, game.MASK_TAKE)).values())
    draw = 0.0
    for card in np.flatnonzero(prepared.deck):
        drawn = game.apply_action(game.preordain(prepared, int(card)), game.MASK_DRAW)
        draw += float(prepared.deck[card] / prepared.deck.sum()) * max(
            exact_action_values(drawn).values()
        )
    return {game.MASK_DRAW: draw, game.MASK_TAKE: take}


def recorded_selection(case: Case) -> dict:
    """Re-evaluate persisted action statistics without rerunning the model."""
    rows = case.metadata["recorded_actions"]
    visit = max(rows, key=lambda row: row["visits"])
    value = max(
        (row for row in rows if row["value"] is not None), key=lambda row: row["value"]
    )
    return {
        "case": case.name,
        "visit_choice": visit["action"],
        "value_choice": value["action"],
        "value_advantage": value["value"] - visit["value"],
    }


@dataclasses.dataclass(frozen=True)
class Variant:
    name: str
    config: SearchConfig
    include_by_default: bool = True


def variants(recorded_settings: dict) -> list[Variant]:
    """The compatibility control, canonical presets, and paired uncertainty sweeps."""
    base = SearchConfig(**recorded_settings)
    result = [Variant("control", base)]
    for valuation in ("current_mixture", "mean_return", "best_response"):
        alpha = dataclasses.replace(
            base, valuation=valuation, fpu_mode="zero", target_strategy="visits"
        )
        result.append(Variant(f"puct-{valuation}", alpha))
        for scale in (0.0, 1.0):
            gumbel = dataclasses.replace(
                alpha,
                root_strategy="gumbel",
                interior_strategy="completed_q",
                action_strategy="gumbel",
                target_strategy="completed_q",
                gumbel_scale=scale,
            )
            result.append(Variant(f"gumbel{int(scale)}-{valuation}", gumbel))
        result.append(
            Variant(
                f"gumbel-puct-{valuation}",
                dataclasses.replace(gumbel, interior_strategy="puct"),
                include_by_default=False,
            )
        )
        for exploration, config in (("puct", alpha), ("gumbel", gumbel)):
            for z in (0.0, 1.0, 2.0):
                repeated = dataclasses.replace(
                    config,
                    replicas=8,
                    action_strategy="lower_confidence",
                    uncertainty_z=z,
                )
                result.append(
                    Variant(f"{exploration}-{valuation}-replicas8-z{int(z)}", repeated)
                )
    return result


class CountingEvaluator:
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.states = self.calls = 0

    def evaluate(self, states):
        self.states += len(states)
        self.calls += bool(states)
        return self.evaluator.evaluate(states)


class CountingBoundary:
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.samples = self.calls = 0

    def evaluate(self, states, rng):
        self.samples += len(states)
        self.calls += bool(states)
        return self.evaluator.evaluate(states, rng)


def _reply_coverage(roots, sequence: list[int]) -> dict:
    """Read retained edges on GIVE → TAKE → CLEAR; never expand a node."""
    visits = np.zeros(len(sequence), dtype=int)
    evaluated = np.zeros(len(sequence), dtype=int)
    for root in roots:
        node = root
        for depth, action in enumerate(sequence):
            groups = getattr(node, "action_groups", None)
            if groups is None:
                break
            representative = next((g[0] for g in groups.members if action in g), None)
            if representative is None or representative not in node.children:
                break
            node = node.children[representative]
            visits[depth] += node.visit_count
            evaluated[depth] += bool(node.has_value_estimate)
    return {"visits": visits.tolist(), "evaluated_replicas": evaluated.tolist()}


def run_trial(
    case: Case, variant: Variant, budget: int, seed: int, evaluator, *, enumerated: bool
) -> dict:
    from skyjo.search.evaluator import NextDealEvaluator
    from skyjo.search.runner import search

    counted = CountingEvaluator(evaluator)
    boundary = CountingBoundary(NextDealEvaluator(counted))
    config = dataclasses.replace(
        variant.config, after_state_evaluate_all_children=enumerated
    )
    started = time.perf_counter()
    result = search(
        case.state,
        counted,
        budget,
        config=config,
        temperature=0.0,
        boundary_evaluator=boundary,
        rng=np.random.default_rng(seed),
    )
    elapsed = time.perf_counter() - started
    summary = result.summary
    actor = game.get_player(case.state)
    choice = int(result.action_probabilities.argmax())
    giveaway = case.metadata.get("giveaway_actions", [])
    reference = case.metadata.get("exact_action_values")
    best = float(max(reference.values())) if reference else None
    actions = np.asarray(summary.actions)
    eligible = np.asarray(summary.available, dtype=bool)
    q_values = np.asarray(summary.values)[:, actor]
    value_choice = int(actions[np.flatnonzero(eligible)[q_values[eligible].argmax()]])
    visits_choice = int(actions[np.asarray(summary.visits).argmax()])
    rows = []
    for index, action in enumerate(actions):
        group = summary.groups.members[index]
        rows.append(
            {
                "actions": list(group),
                "prior": float(summary.priors[index]),
                "visits": int(summary.visits[index]),
                "value": float(q_values[index]) if eligible[index] else None,
                "action_probability": float(
                    result.action_probabilities[list(group)].sum()
                ),
                "training_probability": float(
                    result.training_policy[list(group)].sum()
                ),
                "standard_error": float(summary.standard_error[index, actor])
                if summary.standard_error is not None
                else None,
            }
        )
    return {
        "case": case.name,
        "cohort": case.cohort,
        "variant": variant.name,
        "budget": budget,
        "seed": seed,
        "chance": "enumerated" if enumerated else "sampled",
        "choice": choice,
        "visit_choice": visits_choice,
        "value_choice": value_choice,
        "giveaway": choice in giveaway if giveaway else None,
        "value_choice_giveaway": value_choice in giveaway if giveaway else None,
        "giveaway_probability": float(result.action_probabilities[giveaway].sum())
        if giveaway
        else None,
        "exact_regret": best - float(reference[str(choice)]) if reference else None,
        "value_choice_exact_regret": best - float(reference[str(value_choice)])
        if reference
        else None,
        "actor_value": float(summary.state_value[actor]),
        "seconds": elapsed,
        "evaluator_states": counted.states,
        "evaluator_calls": counted.calls,
        "boundary_samples": boundary.samples,
        "boundary_calls": boundary.calls,
        "reply_coverage": _reply_coverage(result.roots, case.metadata["clear_sequence"])
        if giveaway
        else None,
        "actions": rows,
    }


def summarize(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[
            row["case"], row["cohort"], row["variant"], row["budget"], row["chance"]
        ].append(row)
    result = []
    for (case, cohort, variant, budget, chance), items in sorted(groups.items()):
        entry = {
            "case": case,
            "cohort": cohort,
            "variant": variant,
            "budget": budget,
            "chance": chance,
            "searches": len(items),
            "choices": dict(Counter(row["choice"] for row in items)),
        }
        for field in (
            "giveaway",
            "value_choice_giveaway",
            "giveaway_probability",
            "exact_regret",
            "value_choice_exact_regret",
            "seconds",
            "evaluator_states",
            "boundary_samples",
        ):
            values = [row[field] for row in items if row[field] is not None]
            entry[f"mean_{field}"] = float(np.mean(values)) if values else None
        result.append(entry)
    return result


_worker_evaluator = None


def _initialize_worker(checkpoint: Path):
    import torch

    from skyjo.experiments.contestants import load_model
    from skyjo.learning.predictor import LocalPredictor

    global _worker_evaluator
    torch.set_num_threads(1)
    _worker_evaluator = LocalPredictor(load_model(checkpoint), max_batch_size=512)


def _worker(task):
    case, variant, budget, seed, enumerated = task
    return run_trial(
        case, variant, budget, seed, _worker_evaluator, enumerated=enumerated
    )


def run_benchmark(
    *,
    fixture_path: Path,
    checkpoint: Path,
    output: Path,
    budgets: tuple[int, ...] = (256, 2048, 8192),
    seeds: int = 8,
    base_seed: int = 20261012,
    case_names: tuple[str, ...] = (),
    variant_names: tuple[str, ...] = (),
    chance: str = "both",
    ensemble_budget: int = 2048,
    workers: int = 1,
) -> Path:
    """Run paired seeds, preserving completed rows immediately if later work fails."""
    if not budgets or any(b < 1 for b in budgets) or seeds < 1 or workers < 1:
        raise ValueError("Budgets, seed count, and worker count must be positive")
    if chance not in {"enumerated", "sampled", "both"}:
        raise ValueError("Chance mode must be enumerated, sampled, or both")
    source, cases = load_cases(fixture_path)
    all_variants = variants(source["recorded_player"]["search"])
    for names, available, label in (
        (case_names, {c.name for c in cases}, "cases"),
        (variant_names, {v.name for v in all_variants}, "variants"),
    ):
        if set(names) - available:
            raise ValueError(f"Unknown {label}: {sorted(set(names) - available)}")
    chosen_cases = [c for c in cases if not case_names or c.name in case_names]
    chosen_variants = [
        v
        for v in all_variants
        if (v.name in variant_names if variant_names else v.include_by_default)
    ]
    digest = file_digest(checkpoint)
    if digest != source["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash differs from the frozen fixture checkpoint")
    tasks = []
    for case_index, case in enumerate(cases):
        if case not in chosen_cases:
            continue
        for variant in chosen_variants:
            selected_budgets = (
                (ensemble_budget,) if variant.config.replicas > 1 else budgets
            )
            for budget in selected_budgets:
                modes = (
                    (True, False)
                    if chance == "both" and budget == 2048
                    else (chance != "sampled",)
                )
                for enumerated in modes:
                    for repetition in range(seeds):
                        seed = int(
                            np.random.SeedSequence(
                                [base_seed, case_index, budget, repetition]
                            ).generate_state(1)[0]
                        )
                        tasks.append((case, variant, budget, seed, enumerated))
    repository = Path(__file__).resolve().parents[3]
    manifest = {
        "status": "running",
        "source": source,
        "checkpoint_sha256": digest,
        "fixture_sha256": file_digest(fixture_path),
        "benchmark_sha256": file_digest(Path(__file__)),
        "git": git_provenance(repository, allow_dirty=True),
        "search_sources": {
            p.name: file_digest(p)
            for p in sorted((repository / "src/skyjo/search").glob("*.py"))
        },
        "planned_searches": len(tasks),
        "seeds": seeds,
        "base_seed": base_seed,
        "budgets": budgets,
        "ensemble_budget": ensemble_budget,
        "chance": chance,
        "cases": [c.name for c in chosen_cases],
        "variants": {v.name: dataclasses.asdict(v.config) for v in chosen_variants},
        "workers": workers,
        "recorded_selection": [
            recorded_selection(c) for c in cases if c.cohort == "tactical"
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "benchmark.py").write_text(Path(__file__).read_text())
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    try:
        with (output / "searches.jsonl").open("w") as stream:
            if workers == 1:
                _initialize_worker(checkpoint)
                for task in tasks:
                    row = _worker(task)
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
                    rows.append(row)
            else:
                with ProcessPoolExecutor(
                    max_workers=workers,
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=_initialize_worker,
                    initargs=(checkpoint,),
                ) as pool:
                    for future in as_completed(
                        [pool.submit(_worker, task) for task in tasks]
                    ):
                        row = future.result()
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                        stream.flush()
                        rows.append(row)
        manifest["status"] = "complete"
    except BaseException:
        manifest["status"] = "failed"
        raise
    finally:
        manifest["completed_searches"] = len(rows)
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (output / "summary.json").write_text(
            json.dumps(summarize(rows), indent=2) + "\n"
        )
    return output
