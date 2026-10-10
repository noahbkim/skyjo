"""Paired search timings and separate diagnostics on frozen public states."""

from __future__ import annotations

from skyjo.analytics.reports import symmetry_benchmark_report

import dataclasses
import gc
import json
import shutil
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from skyjo.learning import boundary_inference
from skyjo.experiments.contestants import load_model
from skyjo.engine import game
from skyjo.search import mcts
from skyjo.learning import observations
from skyjo.learning import predictor
from skyjo.experiments import runs
from skyjo.learning import checkpoint as checkpoint_io
from skyjo.experiments.terminal_benchmark import reconstruct
from skyjo.search.symmetry import ActionGroups
from skyjo.search.evaluator import NextDealEvaluator


@dataclasses.dataclass(frozen=True)
class Settings:
    positions: int = 128
    iterations: tuple[int, ...] = (32, 128)
    sweeps: int = 3
    warmup: int = 2
    seed: int = 20261009


@contextmanager
def _benchmark_runtime():
    rng_state = checkpoint_io.capture_rng_state()
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        yield
    finally:
        checkpoint_io.restore_rng_state(rng_state)
        torch.set_num_threads(threads)


@dataclasses.dataclass(frozen=True)
class Case:
    cohort: str
    index: int
    game_id: int | None
    state: game.Skyjo

    def metadata(self) -> dict:
        return {
            "cohort": self.cohort,
            "index": self.index,
            "game_id": self.game_id,
            "phase": ("opening", "draw_or_take", "flip_or_replace", "replace")[
                game.get_action(self.state)
            ],
        }


def select_cases(folder: Path, *, cohort: str, count: int, seed: int) -> list[Case]:
    """Freeze uniformly chosen positions without replacement, retaining replay IDs."""
    spatial = np.load(folder / "spatial_inputs.npy", mmap_mode="r")
    non_spatial = np.load(folder / "non_spatial_inputs.npy", mmap_mode="r")
    masks = np.load(folder / "action_masks.npy", mmap_mode="r")
    offsets = np.load(folder / "game_offsets.npy")
    ids = np.load(folder / "game_indices.npy")
    if count > len(spatial):
        raise ValueError(f"Requested {count} positions from {len(spatial)} in {folder}")
    indices = np.random.default_rng(seed).choice(len(spatial), count, replace=False)
    cases = []
    for index in indices:
        state = reconstruct(spatial[index], non_spatial[index])
        np.testing.assert_array_equal(game.actions(state), masks[index])
        cases.append(
            Case(
                cohort,
                int(index),
                int(ids[np.searchsorted(offsets, index, side="right") - 1]),
                state,
            )
        )
    return cases


def fixture_cases() -> list[Case]:
    """Separate fully asymmetric and deliberately unsafe recycling stress cases."""
    asymmetric = game.new(players=2, top=game.CARD_0)
    asymmetric.game[game.GAME_ACTION : game.GAME_ACTION + game.ACTION_SIZE] = 0
    asymmetric.game[game.GAME_ACTION + game.ACTION_FLIP_OR_REPLACE] = 1
    for slot in range(11):
        row, column = divmod(slot, game.COLUMN_COUNT)
        asymmetric.table[0, row, column] = 0
        asymmetric.table[0, row, column, slot] = 1
        asymmetric.deck[slot] -= 1

    unsafe = game.new(players=2, top=game.CARD_0)
    unsafe.game[game.GAME_ACTION : game.GAME_ACTION + game.ACTION_SIZE] = 0
    unsafe.game[game.GAME_ACTION + game.ACTION_REPLACE] = 1
    unsafe.table[:2] = 0
    unsafe.table[:2, ..., game.FINGER_CLEARED] = 1
    for row, values in enumerate(((None, 1, None), (1, None, 2), (1, 1, 3))):
        for column, value in enumerate(values):
            finger = game.FINGER_HIDDEN if value is None else value + 2
            unsafe.table[0, row, column] = 0
            unsafe.table[0, row, column, finger] = 1
            if value is not None:
                unsafe.deck[finger] -= 1
    remaining = np.zeros(game.CARD_SIZE, dtype=np.int16)
    remaining[game.CARD_P1] = 2
    unsafe.game[game.GAME_DISCARDS : game.GAME_DISCARDS + game.CARD_SIZE] = (
        unsafe.deck - remaining
    )
    unsafe.deck[:] = remaining
    unsafe = dataclasses.replace(unsafe, countdown=1)
    for state in (asymmetric, unsafe):
        game.validate(state)
    return [
        Case("asymmetric_fixture", -1, None, asymmetric),
        Case("unsafe_fixture", -1, None, unsafe),
    ]


def _seed(seed: int, sweep: int, case: int, iterations: int, evaluator: int) -> int:
    value = int(
        np.random.SeedSequence(
            [seed, sweep, case, iterations, evaluator]
        ).generate_state(1)[0]
    )
    return value


class _CountingPredictor:
    def __init__(self, inference):
        self.inference, self.states, self.calls = inference, 0, 0

    def evaluate(self, states):
        self.states += len(states)
        self.calls += bool(states)
        return self.inference.evaluate(states)


class _CountingBoundary:
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.states = self.calls = self.samples = 0

    def evaluate(self, states, rng):
        continuing = sum(not game.get_game_over(state) for state in states)
        self.states += continuing
        self.calls += bool(continuing)
        self.samples += len(states)
        return self.evaluator.evaluate(states, rng)


def diagnose(root: mcts.DecisionStateNode) -> dict:
    """Count retained nodes and collapsed edges after an untimed search."""
    counts = defaultdict(int)
    stack = [root]
    while stack:
        node = stack.pop()
        counts["tree_nodes"] += 1
        counts[type(node).__name__] += 1
        if isinstance(node, mcts.RoundBoundaryNode):
            continue
        stack.extend(node.children.values())
        if isinstance(node, mcts.DecisionStateNode) and node.is_expanded:
            legal = int(game.actions(node.state).sum())
            counts["expanded_decision_nodes"] += 1
            counts["legal_action_edges"] += legal
            counts["retained_action_groups"] += len(node.children)
            counts["grouped_decision_nodes"] += len(node.children) < legal
    return {
        **counts,
        "effective_merge_symmetric_actions": root.effective_merge_symmetric_actions,
        "root_legal_actions": int(game.actions(root.state).sum()),
        "root_groups": len(root.children),
    }


def summarize_timings(rows: list[dict]) -> list[dict]:
    """Compare total paired work, retaining cohort and phase denominators."""
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        # Fixtures remain separate from the representative replay aggregate.
        cohorts = [row["cohort"]]
        if row["cohort"] in {"control", "variant"}:
            cohorts.append("all_replay")
        for cohort in cohorts:
            for phase in ("all", row["phase"]):
                key = cohort, phase, row["evaluator"], row["iterations"]
                groups[key][row["merge_symmetric_actions"]].append(row["seconds"])
    summaries = []
    for (cohort, phase, evaluator, iterations), modes in sorted(groups.items()):
        if set(modes) != {False, True} or len(modes[False]) != len(modes[True]):
            raise ValueError("Timings require equal numbers of paired pooling modes")
        result = dict(
            cohort=cohort, phase=phase, evaluator=evaluator, iterations=iterations
        )
        for merge, label in ((False, "off"), (True, "on")):
            values = np.asarray(modes[merge])
            result[label] = {
                "searches": len(values),
                "total_seconds": float(values.sum()),
                "median_ms": float(np.median(values) * 1000),
                "p95_ms": float(np.quantile(values, 0.95) * 1000),
                "p99_ms": float(np.quantile(values, 0.99) * 1000),
                "searches_per_second": float(len(values) / values.sum()),
            }
        result["throughput_ratio_on_over_off"] = (
            result["off"]["total_seconds"] / result["on"]["total_seconds"]
        )
        summaries.append(result)
    return summaries


def summarize_diagnostics(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        cohorts = [row["cohort"]]
        if row["cohort"] in {"control", "variant"}:
            cohorts.append("all_replay")
        for cohort in cohorts:
            groups[
                cohort,
                row["evaluator"],
                row["iterations"],
                row["merge_symmetric_actions"],
            ].append(row)
    summaries = []
    for (cohort, evaluator, iterations, merge), items in sorted(groups.items()):
        count = len(items)
        total = lambda name: sum(item.get(name, 0) for item in items)
        summaries.append(
            {
                "cohort": cohort,
                "evaluator": evaluator,
                "iterations": iterations,
                "merge_symmetric_actions": merge,
                "searches": count,
                "effective_pooling_fraction": total("effective_merge_symmetric_actions")
                / count,
                "safety_fallback_fraction": (
                    1 - total("effective_merge_symmetric_actions") / count
                )
                if merge
                else 0,
                "root_grouping_fraction": sum(
                    item["root_groups"] < item["root_legal_actions"] for item in items
                )
                / count,
                "grouped_decision_fraction": total("grouped_decision_nodes")
                / total("expanded_decision_nodes"),
                **{
                    f"mean_{name}": total(name) / count
                    for name in (
                        "tree_nodes",
                        "retained_action_groups",
                        "legal_action_edges",
                        "gameplay_evaluated_states",
                        "boundary_evaluated_states",
                    )
                },
            }
        )
    return summaries


def measure_group_construction(cases: list[Case], repeats: int = 100) -> list[dict]:
    """Separate microtiming, including action-mask work, outside all search timings."""
    results = []
    for case in cases:
        for merge in (False, True):
            tick = time.perf_counter()
            for _ in range(repeats):
                ActionGroups.from_state(case.state, merge=merge)
            results.append(
                {
                    **case.metadata(),
                    "merge_symmetric_actions": merge,
                    "repetitions": repeats,
                    "mean_microseconds": (time.perf_counter() - tick) * 1e6 / repeats,
                }
            )
    return results


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def _record_sources(
    repository: Path, recorder: runs.RunRecorder, inputs: list[Path]
) -> list[dict]:
    sources = []
    # Snapshot all package sources: uncommitted boundary and search changes matter.
    for source in [
        repository / "run_mcts_benchmark.py",
        *sorted((repository / "src/skyjo").rglob("*.py")),
    ]:
        destination = recorder.path / "source" / source.relative_to(repository)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        sources.append(
            {
                "path": str(source),
                "sha256": runs.file_digest(destination),
                "snapshot": str(destination.relative_to(recorder.path)),
            }
        )
    for path in inputs:
        sources.append(
            {
                "path": str(path),
                "sha256": runs.file_digest(path),
                "bytes": path.stat().st_size,
            }
        )
    _write_json(recorder.path / "sources.json", sources)
    return sources


def run_benchmark(
    *,
    control_replay: Path,
    variant_replay: Path,
    checkpoint: Path,
    boundary_checkpoint: Path,
    runs_dir: Path,
    settings: Settings,
    allow_dirty: bool = False,
) -> Path:
    if (
        settings.positions < 1
        or settings.sweeps < 1
        or settings.warmup < 0
        or not settings.iterations
        or min(settings.iterations) < 1
    ):
        raise ValueError(
            "Positions, sweeps, and simulation budgets must be positive; warmup nonnegative"
        )
    if len(set(settings.iterations)) != len(settings.iterations):
        raise ValueError("Simulation budgets must be distinct")
    control_replay, variant_replay = control_replay.resolve(), variant_replay.resolve()
    checkpoint, boundary_checkpoint = (
        checkpoint.resolve(),
        boundary_checkpoint.resolve(),
    )
    search_settings = dict(
        c_puct=1.0,
        fpu_reduction=0.0,
        dirichlet_epsilon=0.0,
        after_state_evaluate_all_children=False,
    )
    configuration = {
        "name": "symmetric-mcts-benchmark",
        "seed": settings.seed,
        "settings": dataclasses.asdict(settings),
        "search": search_settings,
        "control_replay": str(control_replay),
        "variant_replay": str(variant_replay),
        "checkpoint": str(checkpoint),
        "boundary_checkpoint": str(boundary_checkpoint),
        "execution": {"device": "cpu", "torch_threads": 1, "max_batch_size": 1},
        "evaluators": {
            "next_deal": {"boundary_samples": 1},
            "frozen": {"boundary_samples": 10},
        },
    }
    invocation = [
        "--control-replay",
        str(control_replay),
        "--variant-replay",
        str(variant_replay),
        "--checkpoint",
        str(checkpoint),
        "--boundary-checkpoint",
        str(boundary_checkpoint),
        "--runs-dir",
        str(runs_dir.resolve()),
        "--positions",
        str(settings.positions),
        "--sweeps",
        str(settings.sweeps),
        "--warmup",
        str(settings.warmup),
        "--seed",
        str(settings.seed),
        *[
            item
            for value in settings.iterations
            for item in ("--iterations", str(value))
        ],
        *(["--allow-dirty"] if allow_dirty else []),
    ]
    repository = Path(__file__).resolve().parents[3]
    recorder = runs.RunRecorder.create(
        root=runs_dir,
        repository=repository,
        input_path=Path("mcts-benchmark.json"),
        input_bytes=json.dumps(configuration, indent=2).encode(),
        configuration=configuration,
        entrypoint="run_mcts_benchmark.py:benchmark",
        invocation=invocation,
        allow_dirty=allow_dirty,
    )
    with _benchmark_runtime(), recorder:
        print(f"MCTS benchmark: {recorder.path}", flush=True)
        started = time.perf_counter()
        inputs = [checkpoint, boundary_checkpoint]
        cases, datasets = [], []
        for number, (cohort, folder) in enumerate(
            (("control", control_replay), ("variant", variant_replay))
        ):
            files = [
                folder / name
                for name in (
                    "manifest.json",
                    "spatial_inputs.npy",
                    "non_spatial_inputs.npy",
                    "action_masks.npy",
                    "game_indices.npy",
                    "game_offsets.npy",
                )
            ]
            inputs.extend(files)
            manifest = json.loads(files[0].read_text())
            selection_seed = int(
                np.random.SeedSequence([settings.seed, number]).generate_state(1)[0]
            )
            cases.extend(
                select_cases(
                    folder, cohort=cohort, count=settings.positions, seed=selection_seed
                )
            )
            datasets.append(
                {
                    "cohort": cohort,
                    "path": str(folder),
                    "dataset_id": manifest["dataset_id"],
                    "selection_seed": selection_seed,
                }
            )
        cases.extend(fixture_cases())
        signatures = [(p.stat().st_size, p.stat().st_mtime_ns) for p in inputs]
        _record_sources(repository, recorder, inputs)
        _write_json(
            recorder.path / "selection.json",
            {"datasets": datasets, "cases": [c.metadata() for c in cases]},
        )
        np.savez_compressed(
            recorder.path / "data/states.npz",
            spatial=np.stack(
                [observations.get_spatial_state_numpy(c.state) for c in cases]
            ),
            non_spatial=np.stack(
                [observations.get_non_spatial_state_numpy(c.state) for c in cases]
            ),
        )
        inference = predictor.LocalPredictor(load_model(checkpoint), max_batch_size=1)
        score_model = boundary_inference.load_boundary_model(
            boundary_checkpoint, players=inference.model.players
        )
        score_evaluator = boundary_inference.ScoreBoundaryEvaluator(score_model)
        evaluators = (("next_deal", None, 1), ("frozen", score_evaluator, 10))
        rows, diagnostics = [], []
        for evaluator_index, (evaluator_name, boundary, samples) in enumerate(
            evaluators
        ):
            configs = {
                merge: mcts.SearchConfig(
                    **search_settings,
                    boundary_samples=samples,
                    merge_symmetric_actions=merge,
                )
                for merge in (False, True)
            }
            for iterations in settings.iterations:
                for warmup in range(settings.warmup):
                    for merge in (False, True):
                        seed = _seed(
                            settings.seed, 0, warmup, iterations, evaluator_index
                        )
                        mcts.run_mcts(
                            cases[warmup % len(cases)].state,
                            inference,
                            iterations,
                            config=configs[merge],
                            boundary_evaluator=boundary,
                            rng=np.random.default_rng(seed),
                        )
                for sweep in range(settings.sweeps):
                    for case_number, case in enumerate(cases):
                        order = (
                            (False, True)
                            if (sweep + case_number) % 2 == 0
                            else (True, False)
                        )
                        for position, merge in enumerate(order):
                            gc.collect()  # Reclaim prior cyclic trees outside the timing.
                            seed = _seed(
                                settings.seed,
                                sweep,
                                case_number,
                                iterations,
                                evaluator_index,
                            )
                            rng = np.random.default_rng(seed)
                            tick = time.perf_counter()
                            root = mcts.run_mcts(
                                case.state,
                                inference,
                                iterations,
                                config=configs[merge],
                                boundary_evaluator=boundary,
                                rng=rng,
                            )
                            elapsed = time.perf_counter() - tick
                            del root
                            row = {
                                **case.metadata(),
                                "case_number": case_number,
                                "evaluator": evaluator_name,
                                "iterations": iterations,
                                "sweep": sweep,
                                "seed": seed,
                                "order": position,
                                "merge_symmetric_actions": merge,
                                "seconds": elapsed,
                            }
                            rows.append(row)
                            with (recorder.path / "timings.jsonl").open("a") as stream:
                                stream.write(json.dumps(row) + "\n")
                        if (case_number + 1) % 32 == 0 or case_number + 1 == len(cases):
                            print(
                                f"Timing {evaluator_name}, {iterations} simulations, sweep {sweep + 1}/{settings.sweeps}: {case_number + 1}/{len(cases)} pairs",
                                flush=True,
                            )
                    recorder.record_event(
                        "timing_sweep",
                        progress={"timed_searches": len(rows)},
                        context={
                            "evaluator": evaluator_name,
                            "iterations": iterations,
                            "sweep": sweep,
                        },
                    )
                # Repeat sweep zero with evaluator wrappers outside timed searches.
                for case_number, case in enumerate(cases):
                    for merge in (False, True):
                        counter = _CountingPredictor(inference)
                        counted_boundary = _CountingBoundary(
                            boundary
                            if boundary is not None
                            else NextDealEvaluator(counter)
                        )
                        seed = _seed(
                            settings.seed, 0, case_number, iterations, evaluator_index
                        )
                        root = mcts.run_mcts(
                            case.state,
                            counter,
                            iterations,
                            config=configs[merge],
                            boundary_evaluator=counted_boundary,
                            rng=np.random.default_rng(seed),
                        )
                        diagnostics.append(
                            {
                                **case.metadata(),
                                "case_number": case_number,
                                "evaluator": evaluator_name,
                                "iterations": iterations,
                                "merge_symmetric_actions": merge,
                                **diagnose(root),
                                "gameplay_evaluated_states": counter.states,
                                "gameplay_prediction_calls": counter.calls,
                                "boundary_evaluated_states": counted_boundary.states,
                                "boundary_prediction_calls": counted_boundary.calls,
                                "boundary_outcome_samples": counted_boundary.samples,
                            }
                        )
                        del root
                    if (case_number + 1) % 32 == 0 or case_number + 1 == len(cases):
                        print(
                            f"Diagnostics {evaluator_name}, {iterations} simulations: {case_number + 1}/{len(cases)} pairs",
                            flush=True,
                        )
                _write_json(recorder.path / "diagnostics.json", diagnostics)
        if signatures != [(p.stat().st_size, p.stat().st_mtime_ns) for p in inputs]:
            raise RuntimeError("Benchmark inputs changed while running")
        report = {
            "settings": configuration,
            "wall_seconds": time.perf_counter() - started,
            "timing_summary": summarize_timings(rows),
            "diagnostic_summary": summarize_diagnostics(diagnostics),
            "limitations": [
                "This measures search speed at fixed simulation counts, not playing strength.",
                "Equal seeds improve reproducibility but pooling changes search trajectories.",
                "Fresh roots, one CPU Torch thread, no root noise; each pair receives its own identically seeded search generator.",
                "Search timing includes ordinary inference and expansion; seed setup, GC before each search, file writes, diagnostics, and warm-up are excluded. GC remains enabled during search.",
                "Diagnostics are a separate rerun of sweep zero. Fixtures are reported separately from replay states.",
                "Gameplay evaluation counts include next-deal bootstrap requests. Boundary counts track continuing rounds supplied to either evaluator, and therefore overlap gameplay counts for next-deal evaluation.",
            ],
        }
        _write_json(
            recorder.path / "group-construction.json", measure_group_construction(cases)
        )
        _write_json(recorder.path / "report.json", report)
        (recorder.path / "report.md").write_text(symmetry_benchmark_report(report))
        for path in sorted(recorder.path.rglob("*")):
            if path.is_file() and path.name not in {
                "run.json",
                "trajectory.jsonl",
                "artifacts.jsonl",
            }:
                recorder.register_artifact(
                    path, kind="mcts_benchmark", progress={"timed_searches": len(rows)}
                )
    return recorder.path
