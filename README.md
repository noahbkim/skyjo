# Skyjo

A NumPy game engine, Monte Carlo tree search, and neural self-play experiments
for full-game Skyjo. Models learn game-win probabilities and search policies,
with optional round-score objectives.

## Quick start

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```sh
uv sync --group dev
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run python distributed_main.py --config configs/smoke.toml --allow-dirty
```

The CPU smoke run generates two games, trains, and saves checkpoint/replay
artifacts under `.runs/`. It checks the pipeline, not playing strength. For
research runs, commit first and omit `--allow-dirty`. Use each entry point's
`--help` and the [example configurations](configs/) for options.

- [Architecture](docs/architecture.md): boundaries and the reasoning behind them.
- [Search strategies](docs/search-strategies.md): composing exploration, valuation, actions, and learning targets.
- [Search benchmarks](docs/search-benchmarks.md): frozen tactical positions and exact-endgame comparisons.
- [Experiments](docs/experiments.md): comparison methodology and preserving evidence.
- [Findings](docs/experiments.md#findings): current research conclusions.
- [Search diagnostics](docs/search-diagnostics.md): reusable positions, tactical discovery,
  and paired search comparisons.

## Play, watch, and replay

```sh
uv run python -m skyjo.ui
uv run python -m skyjo.ui --preset configs/interactive.toml
uv run python -m skyjo.ui --recording .runs/interactive/game-example.json
```

The terminal app supports 2–8 human, random, policy-only, or search-model seats.
Choose checkpoint paths and settings in setup, then **Play** with human seats or
**Watch** an automated matchup. Models must support the table's player count.
Inference runs on CPU. Save/load TOML presets to repeat matchups.

Use Left/Right for previous/next decision, Space for play/pause, Home/End for
first/last position, `[`/`]` for round navigation, and `l` to return to the live
position. Watching starts paused. For automated decisions, Next first computes
and displays a proposal; Next again commits the move. Automatic playback follows
these same steps at the selected interval. Browsing backward pauses play and
does not undo moves. Escape returns to setup; `q` quits.

Human actions are clickable. Use `d`/`t` to draw/take, `f`/`r` to select
flip/replace, then `1`–`9`, `a`–`c` for the twelve slots in row order. Initial
second-card choices are Below/Right buttons. Tab and Enter also operate controls.
Boards stay in fixed seat order; `?` means hidden and `·` means cleared. Narrow
terminals stack the scrollable boards and diagnostics vertically.

Games autosave after each transition under `.runs/interactive/`; **Save As**
exports another copy. Both finished and unfinished recordings can be replayed
without models. Recordings preserve player-supplied diagnostics, selected moves,
and board snapshots. The existing training replay datasets are a separate format;
historical converters and experiment recording hooks are deferred.

See [interactive format and extension contracts](docs/interactive.md).

The separate `skyjo2` prototype is outside the maintained architecture.
