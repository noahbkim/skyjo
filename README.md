# Skyjo

A NumPy game engine, Monte Carlo tree search, and neural self-play experiments
for full-game Skyjo. Models learn game-win probabilities and search policies,
with optional round-score objectives.

## Quick start

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```sh
uv sync --group dev
uv run pytest
uv run python distributed_main.py --config configs/smoke.toml --allow-dirty
```

The CPU smoke run generates two games, trains, and saves checkpoint/replay
artifacts under `.runs/`. It checks the pipeline, not playing strength. For
research runs, commit first and omit `--allow-dirty`. Use each entry point's
`--help` and the [example configurations](configs/) for options.

- [Architecture](docs/architecture.md): boundaries and the reasoning behind them.
- [Experiments](docs/experiments.md): comparison methodology and preserving evidence.
- [Findings](docs/experiments.md#findings): current research conclusions.

The separate `skyjo2` prototype is outside the maintained architecture.
