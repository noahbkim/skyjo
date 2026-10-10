"""Independent final-action sampling audit, with explicit continuation bounds."""

from __future__ import annotations

from skyjo.analytics.reports import terminal_sampling_report

import dataclasses
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np

from skyjo.engine import game
from skyjo.learning import observations
from skyjo.experiments import runs


@dataclasses.dataclass(frozen=True)
class Settings:
    exact_cases: int = 32
    sampled_cases: int = 8
    max_sampled_actions: int = 4
    reference_samples: int = 8192
    repetitions: int = 32
    sample_counts: tuple[int, ...] = (1, 10, 100)
    screen_samples: int = 64
    seed: int = 20261006


def reconstruct(spatial: np.ndarray, non_spatial: np.ndarray) -> game.Skyjo:
    """Recover the full public simulation state and check persisted orientation."""
    players = spatial.shape[0]
    table = np.zeros((game.PLAYER_COUNT, 3, 4, game.FINGER_SIZE), dtype=np.int16)
    table[:players] = spatial
    state = game.Skyjo(
        game=non_spatial[: game.GAME_SIZE].astype(np.int16),
        table=table,
        deck=non_spatial[game.GAME_SIZE : game.GAME_SIZE + game.CARD_SIZE].astype(
            np.int16
        ),
        players=players,
        turn=int(non_spatial[65]),
        countdown=None if non_spatial[66] < 0 else int(non_spatial[66]),
    )
    game.validate(state)
    np.testing.assert_array_equal(
        observations.get_non_spatial_state_numpy(state), non_spatial
    )
    np.testing.assert_array_equal(observations.get_spatial_state_numpy(state), spatial)
    return state


def random_stream(seed: int, stage: int, case: int, action: int) -> random.Random:
    """Separate selection, screening, reference, and estimate random streams."""
    value = np.random.SeedSequence([seed, stage, case, action]).generate_state(2)
    return random.Random(int(value[0]) | (int(value[1]) << 32))


def terminal_credit(state: game.Skyjo) -> tuple[np.ndarray, bool]:
    """Return terminal win contribution in fixed-player order, zero if continuing."""
    if not game.get_round_over(state):
        raise ValueError("The action must finish the round")
    if not game.get_game_over(state):
        return np.zeros(state.players), False
    scores = game.get_fixed_perspective_game_scores(state)
    winners = scores == scores.min()
    return winners / winners.sum(), True


def sample_outcomes(state: game.Skyjo, action: int, count: int, rng: random.Random):
    values = np.empty((count, state.players), dtype=np.float64)
    terminal = np.empty(count, dtype=np.bool_)
    for i in range(count):
        values[i], terminal[i] = terminal_credit(
            game.apply_action(state, action, rng=rng)
        )
    return values, terminal


class _OneDraw:
    def __init__(self, quantile: float):
        self.quantile, self.calls = quantile, 0

    def random(self) -> float:
        self.calls += 1
        if self.calls != 1:
            raise ValueError("One-hidden enumeration encountered a second random draw")
        return self.quantile


def enumerate_one_hidden(state: game.Skyjo, action: int):
    """Integrate one remaining hidden card using exact remaining-deck weights."""
    if state.table[..., game.FINGER_HIDDEN].sum() != 1:
        raise ValueError("Exact enumeration requires exactly one hidden card")
    # A visible replacement can change recyclable discards before the first draw.
    # Enumerating an empty starting deck therefore needs action-dependent support.
    if not state.deck.any():
        raise ValueError("Exact enumeration requires a nonempty initial deck")
    deck = state.deck
    weights = deck.astype(float) / deck.sum()
    cards = np.flatnonzero(weights)
    quantiles = weights.cumsum() - weights / 2
    values, terminal = [], []
    for card in cards:
        rng = _OneDraw(float(quantiles[card]))
        credit, ended = terminal_credit(game.apply_action(state, action, rng=rng))
        if rng.calls != 1:
            raise ValueError("One-hidden enumeration expected exactly one random draw")
        values.append(credit)
        terminal.append(ended)
    return np.array(values), np.array(terminal), weights[cards]


def guaranteed_terminal(state: game.Skyjo) -> bool:
    """Prove termination from an unaffected, fully visible opponent's total.

    The final actor can only alter their own board. The other player's minimum
    possible charge is min(raw, 2*raw), regardless of the round penalty.
    """
    prior = game.get_game_scores(state)
    return any(
        game.get_facedown_count(state, p) == 0
        and prior[p] + min(game.get_score(state, p), 2 * game.get_score(state, p))
        >= 100
        for p in range(1, state.players)
    )


def summarize_outcomes(values, terminal, *, weights=None, alpha: float = 0.05):
    """Bound eventual wins without assigning a learned value to continuing games."""
    exact = weights is not None
    if weights is None:
        weights = np.full(len(terminal), 1 / len(terminal))
    lower = np.average(values, axis=0, weights=weights)
    terminal_probability = float(np.average(terminal, weights=weights))
    upper = np.minimum(1, lower + 1 - terminal_probability)
    # Union bound for both endpoints for every player and terminal probability.
    epsilon = (
        0.0
        if exact
        else math.sqrt(
            math.log(2 * (2 * values.shape[1] + 1) / alpha) / (2 * len(values))
        )
    )
    return {
        "exact": exact,
        "outcomes": len(values),
        "terminal_probability": terminal_probability,
        "terminal_probability_confidence": [
            max(0, terminal_probability - epsilon),
            min(1, terminal_probability + epsilon),
        ],
        "terminal_win_contribution": lower.tolist(),
        "eventual_win_bounds": np.stack((lower, upper), axis=1).tolist(),
        "eventual_win_confidence_bounds": np.stack(
            (np.maximum(0, lower - epsilon), np.minimum(1, upper + epsilon)), axis=1
        ).tolist(),
        "hoeffding_radius": epsilon,
    }


def _select_cases(folder: Path, settings: Settings):
    spatial = np.load(folder / "spatial_inputs.npy", mmap_mode="r")
    non_spatial = np.load(folder / "non_spatial_inputs.npy", mmap_mode="r")
    masks = np.load(folder / "action_masks.npy", mmap_mode="r")
    indices = np.flatnonzero((non_spatial[:, 66] == 1) & (masks[:, 16:].sum(1) > 0))
    hidden = spatial[indices, ..., game.FINGER_HIDDEN].sum((1, 2, 3))
    selection = np.random.default_rng(np.random.SeedSequence([settings.seed, 0]))
    selected = []
    counts = {
        "final_action_candidates": len(indices),
        "one_hidden_candidates": int((hidden == 1).sum()),
        "multi_hidden_candidates": int((hidden > 1).sum()),
        "exact_screened": 0,
        "exact_skipped_empty_deck": 0,
        "multi_screened": 0,
    }
    for index in selection.permutation(indices[hidden == 1]):
        if sum(c["cohort"] == "exact" for c in selected) >= settings.exact_cases:
            break
        counts["exact_screened"] += 1
        state = reconstruct(spatial[index], non_spatial[index])
        if not state.deck.any():
            counts["exact_skipped_empty_deck"] += 1
            continue
        actions = list(map(int, game.get_actions(state)))
        distributions = [enumerate_one_hidden(state, action) for action in actions]
        if all(terminal.all() for _, terminal, _ in distributions):
            selected.append(
                {
                    "index": int(index),
                    "state": state,
                    "actions": actions,
                    "cohort": "exact",
                    "proof": "all legal actions and positive-probability hidden-card outcomes enumerated",
                }
            )
    # Prefer half with a deterministic proof; retain separately screened near-terminal cases.
    proof_quota = (settings.sampled_cases + 1) // 2
    near_quota = settings.sampled_cases - proof_quota
    quotas = {True: proof_quota, False: near_quota}
    for index in selection.permutation(indices[hidden > 1]):
        if not any(quotas.values()):
            break
        if counts["multi_screened"] >= 512:
            break
        state = reconstruct(spatial[index], non_spatial[index])
        proof = guaranteed_terminal(state)
        if quotas[proof] == 0:
            continue
        counts["multi_screened"] += 1
        legal = np.array(list(game.get_actions(state)), dtype=int)
        actions = sorted(
            map(
                int,
                selection.choice(
                    legal,
                    size=min(settings.max_sampled_actions, len(legal)),
                    replace=False,
                ),
            )
        )
        if not proof:
            terminal = [
                sample_outcomes(
                    state,
                    action,
                    settings.screen_samples,
                    random_stream(settings.seed, 1, int(index), action),
                )[1]
                for action in actions
            ]
            if not all(t.all() for t in terminal):
                continue
        selected.append(
            {
                "index": int(index),
                "state": state,
                "actions": actions,
                "cohort": "sampled",
                "proof": "unaffected visible opponent total" if proof else None,
            }
        )
        quotas[proof] -= 1
    counts["sampled_unfilled_quotas"] = {
        "proven_terminal": quotas[True],
        "screened_near_terminal": quotas[False],
    }
    offsets = np.load(folder / "game_offsets.npy")
    game_ids = np.load(folder / "game_indices.npy")
    for case in selected:
        case["game_id"] = int(
            game_ids[np.searchsorted(offsets, case["index"], side="right") - 1]
        )
    return selected, counts


def _evaluate_case(case: dict, settings: Settings, total_actions: int):
    state, actions, index = case["state"], case["actions"], case["index"]
    actor = game.get_player(state)
    summaries, estimates, confidence = [], [], []
    reference_means = []
    reference_samples = {}
    started = time.perf_counter()
    for action in actions:
        if case["cohort"] == "exact":
            values, terminal, weights = enumerate_one_hidden(state, action)
        else:
            values, terminal = sample_outcomes(
                state,
                action,
                settings.reference_samples,
                random_stream(settings.seed, 2, index, action),
            )
            weights = None
        reference_samples[f"{index}_{action}_values"] = values
        reference_samples[f"{index}_{action}_terminal"] = terminal
        if weights is not None:
            reference_samples[f"{index}_{action}_weights"] = weights
        summary = summarize_outcomes(
            values, terminal, weights=weights, alpha=0.05 / total_actions
        )
        summaries.append({"action": action, **summary})
        reference_means.append(summary["terminal_win_contribution"][actor])
        confidence.append(summary["eventual_win_confidence_bounds"][actor])
        rng = random_stream(settings.seed, 3, index, action)
        draws = settings.repetitions * max(settings.sample_counts)
        if weights is None:
            sampled, _ = sample_outcomes(state, action, draws, rng)
        else:
            # Categorical resampling is exactly the engine's enumerated distribution.
            sampled = values[rng.choices(range(len(values)), weights=weights, k=draws)]
        sampled = sampled[:, actor].reshape(settings.repetitions, -1)
        estimates.append(
            np.array([sampled[:, :k].mean(1) for k in settings.sample_counts])
        )
    estimates = np.stack(estimates, axis=-1)
    reference = np.array(reference_means)
    bounds = np.array(confidence)
    all_terminal = case["cohort"] == "exact" or case["proof"] is not None
    metrics = {}
    for row, k in enumerate(settings.sample_counts):
        chosen = estimates[row].argmax(axis=1)
        regret = reference.max() - reference[chosen]
        low = np.maximum(0, bounds[:, 0].max() - bounds[chosen, 1])
        high = np.maximum(0, bounds[:, 1].max() - bounds[chosen, 0])
        metrics[str(k)] = {
            "terminal_contribution_mse": float(
                np.mean((estimates[row] - reference) ** 2)
            ),
            "mean_reference_action_regret": float(regret.mean())
            if all_terminal
            else None,
            "eventual_action_regret_confidence_bounds": [
                float(low.mean()),
                float(high.mean()),
            ],
            "suboptimal_fraction": float((regret > 1e-12).mean())
            if case["cohort"] == "exact"
            else None,
            "chosen_actions": [actions[a] for a in chosen],
        }
    record = {k: v for k, v in case.items() if k != "state"}
    record.update(
        {
            "actor_fixed_player": actor,
            "hidden_cards": int(state.table[..., game.FINGER_HIDDEN].sum()),
            "previous_scores_current_order": game.get_game_scores(state).tolist(),
            "reference": summaries,
            "sampling": metrics,
            "reference_action_value_spread": float(np.ptp(reference)),
            "has_stochastic_terminal_value": bool(
                np.any((reference > 1e-12) & (reference < 1 - 1e-12))
            ),
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    reference_samples[f"{index}_estimates"] = estimates
    return record, reference_samples


def run_benchmark(
    source_run: Path, runs_dir: Path, settings: Settings, *, allow_dirty=False
) -> Path:
    if (
        min(settings.exact_cases, settings.sampled_cases) < 0
        or settings.exact_cases + settings.sampled_cases == 0
    ):
        raise ValueError("Request at least one nonnegative cohort size")
    if (
        min(
            settings.reference_samples,
            settings.repetitions,
            settings.screen_samples,
            settings.max_sampled_actions,
            *settings.sample_counts,
        )
        <= 0
    ):
        raise ValueError("Sampling budgets must be positive")
    source_run = source_run.resolve()
    folder = source_run / "data/replay"
    configuration = {
        "name": "terminal-boundary-benchmark",
        "source_run": str(source_run),
        "settings": dataclasses.asdict(settings),
        "seed": settings.seed,
        "execution": {"device": "cpu"},
    }
    repository = Path(__file__).resolve().parents[3]
    recorder = runs.RunRecorder.create(
        root=runs_dir,
        repository=repository,
        input_path=Path("terminal-benchmark.json"),
        input_bytes=json.dumps(configuration, indent=2).encode(),
        configuration=configuration,
        entrypoint="run_terminal_benchmark.py:benchmark",
        invocation=[
            "--source-run",
            str(source_run),
            "--runs-dir",
            str(runs_dir.resolve()),
            "--exact-cases",
            str(settings.exact_cases),
            "--sampled-cases",
            str(settings.sampled_cases),
            "--reference-samples",
            str(settings.reference_samples),
            "--repetitions",
            str(settings.repetitions),
            "--seed",
            str(settings.seed),
            *(["--allow-dirty"] if allow_dirty else []),
        ],
        allow_dirty=allow_dirty,
    )
    with recorder:
        print(f"Terminal benchmark: {recorder.path}", flush=True)
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
        signatures = [(p.stat().st_size, p.stat().st_mtime_ns) for p in files]
        sources = [
            {"path": str(p), "sha256": runs.file_digest(p), "bytes": p.stat().st_size}
            for p in files
        ]
        cases, selection = _select_cases(folder, settings)
        if not cases:
            raise ValueError("No eligible terminal cases found")
        np.savez_compressed(
            recorder.path / "data/states.npz",
            indices=[c["index"] for c in cases],
            spatial=np.stack(
                [observations.get_spatial_state_numpy(c["state"]) for c in cases]
            ),
            non_spatial=np.stack(
                [observations.get_non_spatial_state_numpy(c["state"]) for c in cases]
            ),
        )
        report = {
            "settings": dataclasses.asdict(settings),
            "source_run": str(source_run),
            "selection": selection,
            "cases": [],
            "limitations": [
                "Targeted final-action stress cases, not a representative strength test.",
                "Exact cases include all legal actions. Sampled cases compare only the disclosed candidate subset.",
                "Continuing outcomes contribute zero to the terminal lower bound and their entire probability mass to the upper bound; no conditioning or learned bootstrap.",
                "95% Hoeffding confidence bounds are simultaneous over every sampled action, player endpoint, and terminal probability in this benchmark.",
                "Sampling estimates use independent streams from screening and references; K values share prefixes within a repetition.",
                "Exact-cohort action regret is exact. Proven-terminal sampled-cohort regret uses a finite reference and is accompanied by uncertainty bounds.",
            ],
        }
        saved = {}
        total_actions = sum(len(c["actions"]) for c in cases)
        for number, case in enumerate(cases, 1):
            record, arrays = _evaluate_case(case, settings, total_actions)
            report["cases"].append(record)
            saved.update(arrays)
            print(
                f"Completed {number}/{len(cases)}: {case['cohort']} index {case['index']}, {record['elapsed_seconds']:.1f}s",
                flush=True,
            )
        report["summary"] = {}
        for cohort in ("exact", "sampled"):
            subset = [c for c in report["cases"] if c["cohort"] == cohort]
            if not subset:
                continue
            report["summary"][cohort] = {
                "cases": len(subset),
                "actions": sum(len(c["actions"]) for c in subset),
                "stochastic_value_cases": sum(
                    c["has_stochastic_terminal_value"] for c in subset
                ),
                "action_sensitive_cases": sum(
                    c["reference_action_value_spread"] > 1e-12 for c in subset
                ),
                "minimum_reference_terminal_probability": min(
                    a["terminal_probability"] for c in subset for a in c["reference"]
                ),
                "sampling": {},
            }
            for k in settings.sample_counts:
                items = [c["sampling"][str(k)] for c in subset]
                eligible_regrets = [
                    m["mean_reference_action_regret"]
                    for m in items
                    if m["mean_reference_action_regret"] is not None
                ]
                report["summary"][cohort]["sampling"][str(k)] = {
                    "mean_case_terminal_contribution_mse": float(
                        np.mean([m["terminal_contribution_mse"] for m in items])
                    ),
                    "mean_reference_action_regret": float(np.mean(eligible_regrets))
                    if eligible_regrets
                    else None,
                    "regret_cases": len(eligible_regrets),
                }
        if signatures != [(p.stat().st_size, p.stat().st_mtime_ns) for p in files]:
            raise RuntimeError("Source replay changed during the benchmark")
        (recorder.path / "sources.json").write_text(
            json.dumps(sources, indent=2) + "\n"
        )
        (recorder.path / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        np.savez_compressed(recorder.path / "data/outcomes.npz", **saved)
        (recorder.path / "report.md").write_text(terminal_sampling_report(report))
        for relative in (
            "run_terminal_benchmark.py",
            "src/skyjo/analytics/reports.py",
            "src/skyjo/experiments/terminal_benchmark.py",
            "src/skyjo/engine/game.py",
            "src/skyjo/learning/observations.py",
        ):
            destination = recorder.path / "source" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(repository / relative, destination)
        for path in sorted(recorder.path.rglob("*")):
            if path.is_file() and path.name not in {
                "run.json",
                "trajectory.jsonl",
                "artifacts.jsonl",
            }:
                recorder.register_artifact(path, kind="terminal_benchmark", progress={})
        print(json.dumps(report["summary"], indent=2), flush=True)
    return recorder.path
