"""Concise Markdown views of structured experiment results.

Renderers only consume results. Recipes own execution and artifact recording;
complete metrics and provenance remain in the linked JSON and data files.
"""


def boundary_value_report(report: dict) -> str:
    lines = [
        "# Round-boundary value experiment",
        "",
        "Logistic and two-layer MLP models predict eventual win credit from "
        "cumulative scores / 100 in next-starter order. Games stay within one "
        "split; validation MSE selects checkpoints. Results average seed metrics.",
        "",
        "| Split | Games with boundaries | Boundaries |",
        "| --- | ---: | ---: |",
    ]
    for split, counts in report["splits"].items():
        lines.append(f"| {split} | {counts['games']} | {counts['boundaries']} |")
    lines += [
        "",
        "| Model | Test value MSE | Brier | Cross entropy | ECE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for kind, metrics in report["test_averages"].items():
        lines.append(
            f"| {kind} | {metrics['value_mse']:.6f} | {metrics['brier_score']:.6f} "
            f"| {metrics['cross_entropy']:.6f} | {metrics['ece']:.6f} |"
        )
    comparison = report["mlp_minus_logistic_test_mse"]
    low, high = comparison["ci95"]
    conclusion = (
        "No predictive advantage is established."
        if low <= 0 <= high
        else "The interval favors the MLP."
        if high < 0
        else "The interval favors logistic regression."
    )
    lines += [
        "",
        f"MLP minus logistic MSE: **{comparison['mean']:+.6f}**, paired "
        f"game-bootstrap 95% interval **[{low:+.6f}, {high:+.6f}]**. {conclusion}",
        "",
        "MSE averages over players; Brier sums over players. Calibration alone "
        "does not measure usefulness. The interval conditions on fitted models "
        "and this split of the recorded policy mixture; it does not establish "
        "future-policy generalization or playing strength.",
        "",
        "Artifacts: [full results](results.json), [curves](curves.jsonl), "
        "[test predictions](predictions.npz), [data](data/boundaries.npz), "
        "[split](split.json), [sources](sources.json), and `source/`.",
        "",
    ]
    return "\n".join(lines)


def boundary_methods_report(report: dict) -> str:
    dataset = report["dataset"]
    lines = [
        "# Held-out round-boundary valuation",
        "",
        f"{dataset['boundaries']} nonterminal boundaries from "
        f"{dataset['games_with_boundaries']} games generated after the gameplay "
        "checkpoint's training and absent from the score-model dataset. The "
        "gameplay checkpoint generated these games. Next-deal budgets share "
        "sample prefixes; errors average replicate/seed metrics, not ensembles.",
        "",
        "| Method | Value MSE | Brier | ECE | ms / boundary |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, method in report["methods"].items():
        metrics = method["mean_metrics"]
        lines.append(
            f"| {name} | {metrics['value_mse']:.6f} | {metrics['brier_score']:.6f} "
            f"| {metrics['ece']:.6f} | {method['milliseconds_per_boundary']:.3f} |"
        )
    lines += [
        "",
        "| MSE difference (negative favors first) | Estimate | Game-bootstrap 95% interval |",
        "| --- | ---: | --- |",
    ]
    for name, comparison in report["paired_test_mse"].items():
        low, high = comparison["ci95"]
        lines.append(
            f"| {name} | {comparison['mean']:+.6f} | [{low:+.6f}, {high:+.6f}] |"
        )
    lines += [
        "",
        "Scores are held fixed: this does not resample final actions/reveals. "
        "Observed winners are noisy labels. Intervals condition on fitted models "
        "and deals; model families have different training histories. Lower error "
        "does not establish better training or play. Cost includes dealing and "
        "batched CPU inference, excludes loading, and is not search-node latency.",
        "",
        "Artifacts: [full results](results.json), [sources](sources.json), "
        "`data/`, `checkpoints/`, and `source/` retain predictions, labels, "
        "replicate calibration, inputs and code/model snapshots.",
        "",
    ]
    return "\n".join(lines)


def terminal_sampling_report(report: dict) -> str:
    lines = [
        "# Final-action boundary sampling",
        "",
        "Sampled action estimates are compared with exact or independent "
        "simulation references. This measures estimator accuracy, not full-game play.",
        "",
        "| Cohort | Cases | Actions | K | Mean case MSE | Mean action regret | Regret cases |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cohort, summary in report["summary"].items():
        for samples, row in summary["sampling"].items():
            regret = row["mean_reference_action_regret"]
            regret_text = "undefined" if regret is None else f"{regret:.7f}"
            lines.append(
                f"| {cohort} | {summary['cases']} | {summary['actions']} | {samples} "
                f"| {row['mean_case_terminal_contribution_mse']:.7f} "
                f"| {regret_text} | {row['regret_cases']} |"
            )
    lines += [
        "",
        *[f"- {limit}" for limit in report["limitations"]],
        "",
        "Artifacts: [full report](report.json), [outcomes](data/outcomes.npz), "
        "[sources](sources.json), and `source/`. These retain selection "
        "counts, per-action probabilities/bounds, states, references and estimates.",
        "",
    ]
    return "\n".join(lines)


def symmetry_benchmark_report(report: dict) -> str:
    lines = [
        "# Symmetric-action MCTS speed benchmark",
        "",
        "Throughput ratios above one favor grouping; fixed simulation counts "
        "measure speed rather than playing strength. Cohorts remain separate.",
        "",
        "| Cohort | Evaluator | Visits | Off total s | On total s | Throughput on/off | Off p95 ms | On p95 ms |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["timing_summary"]:
        if row["phase"] != "all":
            continue
        off, on = row["off"], row["on"]
        lines.append(
            f"| {row['cohort']} | {row['evaluator']} | {row['iterations']} "
            f"| {off['total_seconds']:.3f} | {on['total_seconds']:.3f} "
            f"| {row['throughput_ratio_on_over_off']:.3f} "
            f"| {off['p95_ms']:.3f} | {on['p95_ms']:.3f} |"
        )
    lines += [
        "",
        *[f"- {limit}" for limit in report["limitations"]],
        "",
        "Artifacts: [full report](report.json), [sources](sources.json), "
        "and registered raw timings/diagnostics retain phase summaries, "
        "node and inference counts, selected positions and input hashes.",
        "",
    ]
    return "\n".join(lines)
