"""Reproducible, experimental search performance measurements.

No training or production-default changes. Profiling and allocation measurement
are separate from timed trials. Outputs are incremental and never overwritten.
"""

from __future__ import annotations

import cProfile
import dataclasses
import gc
import hashlib
import json
import multiprocessing
import os
import platform
import pstats
import resource
import shutil
import statistics
import subprocess
import time
import tracemalloc
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from skyjo.engine import game
from skyjo.experiments.contestants import ContestantConfig, load_model
from skyjo.experiments.generation import configure_torch_worker
from skyjo.experiments.runs import file_digest
from skyjo.experiments.search_performance_poc import make_predictor, snapshot_variant
from skyjo.experiments.terminal_benchmark import reconstruct
from skyjo.learning import checkpoint as checkpoint_io
from skyjo.search import runner
from skyjo.search.player import SearchPlayer
from skyjo.simulation import play, recording
from skyjo.simulation.jobs import GameRandomStreams, derive_game_seed


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def append(path, value):
    with path.open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")


def host():
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cpu_count": os.cpu_count(),
        "load_average": os.getloadavg(),
        "torch_threads": torch.get_num_threads(),
        "hardware": subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        if platform.system() == "Darwin"
        else platform.processor(),
    }


def prepare(output, checkpoint, config, replay, count, seed):
    """Balance phase/hidden-count/boundary strata from a random candidate pool."""
    if count < 1:
        raise ValueError("positions must be positive")
    manifest_bytes = (replay / "manifest.json").read_bytes()
    spatial = np.load(replay / "spatial_inputs.npy", mmap_mode="r")
    non_spatial = np.load(replay / "non_spatial_inputs.npy", mmap_mode="r")
    masks = np.load(replay / "action_masks.npy", mmap_mode="r")
    offsets = np.load(replay / "game_offsets.npy")
    ids = np.load(replay / "game_indices.npy")
    rng = np.random.default_rng(seed)
    candidates = rng.choice(
        len(spatial), min(len(spatial), max(8192, count)), replace=False
    )
    strata = defaultdict(list)
    for i in candidates:
        phase = int(
            non_spatial[
                i, game.GAME_ACTION : game.GAME_ACTION + game.ACTION_SIZE
            ].argmax()
        )
        hidden = int(spatial[i, ..., game.FINGER_HIDDEN].sum())
        boundary = bool(non_spatial[i, game.GAME_SIZE + game.CARD_SIZE + 1] == 1)
        strata[(phase, min(hidden // 8, 2), boundary)].append(int(i))
    selected = []
    while len(selected) < count:
        added = False
        for key in sorted(strata):
            if strata[key] and len(selected) < count:
                selected.append((strata[key].pop(), key))
                added = True
        if not added:
            raise ValueError("Not enough replay positions")
    cases = []
    for index, stratum in selected:
        state = reconstruct(spatial[index], non_spatial[index])
        np.testing.assert_array_equal(game.actions(state), masks[index])
        cases.append(
            {
                "index": index,
                "game_id": int(ids[np.searchsorted(offsets, index, side="right") - 1]),
                "stratum": stratum,
                "state": recording.state_to_dict(state),
            }
        )
    if (replay / "manifest.json").read_bytes() != manifest_bytes:
        raise ValueError(
            "Replay changed while selecting positions; retry from a stable snapshot"
        )
    shutil.copyfile(checkpoint, output / "checkpoint.pth")
    shutil.copyfile(config, output / "resolved-config.json")
    save(output / "cases.json", cases)
    save(
        output / "inputs.json",
        {
            "checkpoint_source": str(checkpoint),
            "checkpoint_sha256": file_digest(output / "checkpoint.pth"),
            "config_source": str(config),
            "config_sha256": file_digest(output / "resolved-config.json"),
            "replay_source": str(replay),
            "replay_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "replay_manifest": json.loads(manifest_bytes),
            "cases_sha256": file_digest(output / "cases.json"),
            "selection": "Round-robin balanced strata from a seeded random candidate pool; not population weighted",
            "candidate_count": len(candidates),
            "seed": seed,
            "strata": dict(Counter(str(tuple(c["stratum"])) for c in cases)),
        },
    )


def load_inputs(inputs):
    metadata = json.loads((inputs / "inputs.json").read_text())
    for name, key in (
        ("checkpoint.pth", "checkpoint_sha256"),
        ("resolved-config.json", "config_sha256"),
        ("cases.json", "cases_sha256"),
    ):
        if file_digest(inputs / name) != metadata[key]:
            raise ValueError(f"Frozen input changed: {name}")
    config = json.loads((inputs / "resolved-config.json").read_text())
    settings = ContestantConfig.from_search_settings(config["search"])
    cases = json.loads((inputs / "cases.json").read_text())
    states = [
        recording.state_from_dict(case["state"], config["players"]) for case in cases
    ]
    return load_model(inputs / "checkpoint.pth"), settings, states


class Counted:
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.batches = Counter()

    def evaluate(self, states):
        if states:
            self.batches[len(states)] += 1
        return self.evaluator.evaluate(states)


def search_once(state, evaluator, settings, iterations, seed):
    rng = np.random.default_rng(seed)
    result = runner.search(
        state,
        evaluator,
        iterations,
        config=settings.search,
        temperature=settings.temperature,
        rng=rng,
    )
    return result, rng.bit_generator.state


def result_digest(result, rng_state):
    digest = hashlib.sha256(json.dumps(rng_state, sort_keys=True).encode())
    for field in dataclasses.fields(result.summary):
        value = getattr(result.summary, field.name)
        if isinstance(value, np.ndarray):
            digest.update(str((value.shape, value.dtype)).encode())
            digest.update(value.tobytes())
        else:
            digest.update(repr(value).encode())
    digest.update(result.action_probabilities.tobytes())
    digest.update(result.training_policy.tobytes())
    digest.update(
        repr([r.effective_merge_symmetric_actions for r in result.roots]).encode()
    )
    return digest.hexdigest()


def searches(output, inputs, variants, budgets, sweeps, seed, limit):
    model, settings, states = load_inputs(inputs)
    states = states[:limit]
    evaluators = {v: make_predictor(model, v) for v in variants}
    expected = {}
    # Establish exact reference results separately, also warming inference.
    baseline = make_predictor(model, "baseline")
    for budget in budgets:
        for i, state in enumerate(states):
            counter = Counted(baseline)
            result, rng = search_once(
                state, counter, settings, budget, derive_game_seed(seed, i, budget)
            )
            expected[budget, i] = result_digest(result, rng), counter.batches
    for sweep in range(sweeps):
        order = variants if sweep % 2 == 0 else variants[::-1]
        for budget in budgets:
            for variant in order:
                evaluator = evaluators[variant]
                with snapshot_variant(variant):
                    search_once(states[0], evaluator, settings, budget, seed)
                    gc.collect()
                    block_started = time.perf_counter()
                    load_before = os.getloadavg()
                    for i, state in enumerate(states):
                        counter = Counted(evaluator)
                        cpu_started = time.process_time()
                        started = time.perf_counter()
                        result, rng = search_once(
                            state,
                            counter,
                            settings,
                            budget,
                            derive_game_seed(seed, i, budget),
                        )
                        elapsed = time.perf_counter() - started
                        cpu_elapsed = time.process_time() - cpu_started
                        signature = result_digest(result, rng)
                        reference, batch_counts = expected[budget, i]
                        if signature != reference or counter.batches != batch_counts:
                            raise AssertionError(
                                f"Behavior changed: {variant}, budget={budget}, case={i}"
                            )
                        append(
                            output / "searches.jsonl",
                            {
                                "variant": variant,
                                "budget": budget,
                                "sweep": sweep,
                                "case": i,
                                "seconds": elapsed,
                                "cpu_seconds": cpu_elapsed,
                                "result_sha256": signature,
                                "evaluated_states": sum(
                                    n * count for n, count in counter.batches.items()
                                ),
                                "batch_sizes": counter.batches,
                            },
                        )
                    append(
                        output / "blocks.jsonl",
                        {
                            "variant": variant,
                            "budget": budget,
                            "sweep": sweep,
                            "load_before": load_before,
                            "load_after": os.getloadavg(),
                        },
                    )
                print(
                    f"search {variant} budget={budget} sweep={sweep}: {time.perf_counter() - block_started:.2f}s including checks",
                    flush=True,
                )
    summarize_searches(output)


def summarize_searches(output):
    """Summarize completed paired blocks; also usable on an existing run folder."""
    rows = [
        json.loads(line)
        for line in (output / "searches.jsonl").read_text().splitlines()
    ]
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    case_sets = defaultdict(set)
    signatures = {}
    for row in rows:
        identity = row["budget"], row["case"]
        if identity in signatures and signatures[identity] != row["result_sha256"]:
            raise ValueError("Cannot summarize mismatched search results")
        signatures[identity] = row["result_sha256"]
        case_sets[row["budget"], row["variant"], row["sweep"]].add(row["case"])
        for metric in ("seconds", "cpu_seconds"):
            if metric in row:
                groups[row["budget"], row["variant"]][row["sweep"]][metric] += row[
                    metric
                ]
    if not rows or any(
        cases != next(iter(case_sets.values())) for cases in case_sets.values()
    ):
        raise ValueError("Cannot summarize empty or incompletely paired search blocks")
    summaries = []
    for (budget, variant), sweeps in sorted(groups.items()):
        summary = {"budget": budget, "variant": variant, "sweeps": len(sweeps)}
        for metric in next(iter(sweeps.values())):
            totals = [values[metric] for _, values in sorted(sweeps.items())]
            summary[metric] = {"totals": totals, "median": statistics.median(totals)}
            if (budget, "baseline") in groups:
                ratios = [
                    groups[budget, "baseline"][sweep][metric] / values[metric]
                    for sweep, values in sorted(sweeps.items())
                ]
                summary[metric]["paired_speedups"] = ratios
                summary[metric]["median_paired_speedup"] = statistics.median(ratios)
        summaries.append(summary)
    save(
        output / "summary.json",
        {
            "trials": len(rows),
            "exact_parity": True,
            "results": summaries,
            "note": "Ratios above one favor the variant; load and short-sweep variation can dominate small effects.",
        },
    )
    for row in summaries:
        wall = row["seconds"].get("median_paired_speedup", float("nan"))
        cpu = row.get("cpu_seconds", {}).get("median_paired_speedup", float("nan"))
        print(
            f"summary budget={row['budget']} {row['variant']}: wall {wall:.3f}x, CPU {cpu:.3f}x",
            flush=True,
        )
    return summaries


def profiles(output, inputs, variants, budget, seed, limit):
    model, settings, states = load_inputs(inputs)
    states = states[:limit]
    for variant in variants:
        evaluator = Counted(make_predictor(model, variant))

        def work(evaluator=evaluator):
            for i, state in enumerate(states):
                search_once(
                    state,
                    evaluator,
                    settings,
                    budget,
                    derive_game_seed(seed, i, budget),
                )

        with snapshot_variant(variant):
            work()
            profiler = cProfile.Profile()
            profiler.runcall(work)
            profiler.dump_stats(str(output / f"{variant}.prof"))
            stats = pstats.Stats(profiler)
            rows = [
                {
                    "file": f,
                    "line": line,
                    "name": name,
                    "calls": nc,
                    "self_seconds": tt,
                    "cumulative_seconds": ct,
                }
                for (f, line, name), (_, nc, tt, ct, _) in stats.stats.items()
            ]
            save(
                output / f"{variant}-profile.json",
                {
                    "total_seconds": stats.total_tt,
                    "functions": sorted(
                        rows, key=lambda row: row["cumulative_seconds"], reverse=True
                    ),
                },
            )
            gc.collect()
            tracemalloc.start()
            retained = [
                search_once(
                    state,
                    evaluator,
                    settings,
                    budget,
                    derive_game_seed(seed, i, budget),
                )[0]
                for i, state in enumerate(states[:4])
            ]
            current, peak = tracemalloc.get_traced_memory()
            snapshot = tracemalloc.take_snapshot()
            top = [
                {
                    "location": str(stat.traceback),
                    "bytes": stat.size,
                    "allocations": stat.count,
                }
                for stat in snapshot.statistics("lineno")[:25]
            ]
            tracemalloc.stop()
            save(
                output / f"{variant}-memory.json",
                {
                    "retained_searches": len(retained),
                    "traced_current_bytes": current,
                    "traced_peak_bytes": peak,
                    "live_allocations": sum(
                        s.count for s in snapshot.statistics("lineno")
                    ),
                    "top": top,
                    "process_peak_rss_native_units": resource.getrusage(
                        resource.RUSAGE_SELF
                    ).ru_maxrss,
                    "note": "tracemalloc excludes some native Torch allocations; RSS is process lifetime high-water mark, bytes on macOS, KiB on Linux",
                },
            )
            del retained
        print(f"profile {variant} complete", flush=True)


def batch_benchmark(output, inputs, variants, sweeps):
    model, _, states = load_inputs(inputs)
    for variant in variants:
        evaluator = make_predictor(model, variant, max_batch_size=128)
        singles = evaluator.evaluate(states[:1])
        # Compare identical repeated states to isolate arithmetic changes by batch size.
        for size in (1, 4, 16, 64, 128):
            selected = [states[0]] * size
            predictions = evaluator.evaluate(selected)
            error = max(
                float(np.max(np.abs(p.value - singles[0].value))) for p in predictions
            )
            policy_error = max(
                float(np.max(np.abs(p.policy - singles[0].policy))) for p in predictions
            )
            for sweep in range(sweeps):
                start = time.perf_counter()
                for _ in range(max(16, 1024 // size)):
                    evaluator.evaluate(selected)
                elapsed = time.perf_counter() - start
                append(
                    output / "batches.jsonl",
                    {
                        "variant": variant,
                        "batch_size": size,
                        "sweep": sweep,
                        "states": max(16, 1024 // size) * size,
                        "seconds": elapsed,
                        "max_value_error_vs_single": error,
                        "max_policy_error_vs_single": policy_error,
                    },
                )


def game_trial(args):
    inputs, variant, iterations, seed, index = args
    torch.set_num_threads(1)
    inputs = Path(inputs)
    config = json.loads((inputs / "resolved-config.json").read_text())
    settings = ContestantConfig.from_search_settings(config["search"])
    model = load_model(inputs / "checkpoint.pth")
    evaluator = Counted(make_predictor(model, variant))
    streams = GameRandomStreams.from_seed(derive_game_seed(seed, index))
    agent = SearchPlayer(
        evaluator,
        iterations,
        config=settings.search,
        temperature=settings.temperature,
        rng=streams.search,
    )
    cpu_start = time.process_time()
    start = time.perf_counter()
    with snapshot_variant(variant):
        result = play.play_game(
            [agent] * model.players,
            environment_rng=streams.environment,
            action_rng=streams.actions,
        )
    elapsed = time.perf_counter() - start
    cpu_seconds = time.process_time() - cpu_start
    digest = hashlib.sha256()
    decisions = 0
    for round_ in result.rounds:
        for entry in round_.history:
            digest.update(
                json.dumps(
                    recording.state_to_dict(entry.state), sort_keys=True
                ).encode()
            )
            digest.update(str(entry.action).encode())
            for array in (entry.action_probabilities, entry.training_policy):
                if array is not None:
                    digest.update(array.tobytes())
            decisions += entry.action is not None
    for rng in (streams.environment, streams.actions, streams.search):
        digest.update(json.dumps(rng.bit_generator.state, sort_keys=True).encode())
    return {
        "game": index,
        "seconds": elapsed,
        "cpu_seconds": cpu_seconds,
        "decisions": decisions,
        "rounds": len(result.rounds),
        "scores": result.final_scores,
        "trajectory_sha256": digest.hexdigest(),
        "batch_sizes": evaluator.batches,
        "evaluated_states": sum(n * count for n, count in evaluator.batches.items()),
        "peak_rss_native_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def games(output, inputs, variants, budget, workers, count, sweeps, seed):
    load_inputs(inputs)  # Verify frozen input hashes before starting any workers.
    expected = {}
    for sweep in range(sweeps):
        for worker_count in workers:
            for variant in variants if sweep % 2 == 0 else variants[::-1]:
                tasks = [
                    (str(inputs.resolve()), variant, budget, seed, i)
                    for i in range(count)
                ]
                start = time.perf_counter()
                with ProcessPoolExecutor(
                    max_workers=worker_count,
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=configure_torch_worker,
                    initargs=(1,),
                ) as pool:
                    results = list(pool.map(game_trial, tasks))
                seconds = time.perf_counter() - start
                for row in results:
                    if (
                        row["game"] in expected
                        and row["trajectory_sha256"] != expected[row["game"]]
                    ):
                        raise AssertionError(
                            f"Game trajectory changed: {variant}, {row['game']}"
                        )
                    expected[row["game"]] = row["trajectory_sha256"]
                append(
                    output / "games.jsonl",
                    {
                        "variant": variant,
                        "workers": worker_count,
                        "sweep": sweep,
                        "iterations": budget,
                        "wall_seconds_including_startup": seconds,
                        "results": results,
                        "host": host(),
                    },
                )
                print(
                    f"games {variant} workers={worker_count}: {seconds:.2f}s",
                    flush=True,
                )


def run(
    *,
    stage,
    output,
    inputs,
    checkpoint=None,
    config=None,
    replay=None,
    variants=("baseline", "inference", "snapshots", "combined"),
    budgets=(32, 128, 256),
    sweeps=3,
    positions=128,
    seed=20261017,
    workers=(1, 4, 8),
    game_count=8,
):
    if min(sweeps, positions, game_count, *budgets, *workers) < 1:
        raise ValueError("Counts, budgets and workers must be positive")
    if stage == "games" and variants[0] != "baseline":
        raise ValueError("Game comparisons must start with baseline")
    output.mkdir(parents=True, exist_ok=False)
    caller_rng, caller_threads = (
        checkpoint_io.capture_rng_state(),
        torch.get_num_threads(),
    )
    torch.set_num_threads(1)
    manifest = {
        "stage": stage,
        "status": "running",
        "host_start": host(),
        "variants": variants,
        "budgets": budgets,
        "sweeps": sweeps,
        "positions": positions,
        "seed": seed,
        "workers": workers,
        "game_count": game_count,
        "inputs": str(inputs.resolve()),
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
        "sources": {
            str(p): file_digest(p)
            for p in [
                *Path("src/skyjo").rglob("*.py"),
                Path("run_search_performance.py"),
            ]
        },
    }
    save(output / "manifest.json", manifest)
    try:
        if stage == "prepare":
            prepare(output, checkpoint, config, replay, positions, seed)
        elif stage == "searches":
            searches(output, inputs, variants, budgets, sweeps, seed, positions)
        elif stage == "profile":
            profiles(output, inputs, variants, budgets[-1], seed, positions)
        elif stage == "batches":
            batch_benchmark(output, inputs, variants, sweeps)
        elif stage == "games":
            games(
                output, inputs, variants, budgets[-1], workers, game_count, sweeps, seed
            )
        else:
            raise ValueError(f"Unknown stage: {stage}")
        manifest["status"] = "complete"
    except BaseException as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        manifest["host_end"] = host()
        save(output / "manifest.json", manifest)
        checkpoint_io.restore_rng_state(caller_rng)
        torch.set_num_threads(caller_threads)
    return output
