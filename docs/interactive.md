# Interactive games and recordings

`skyjo.ui` is a Textual frontend over `simulation.session.GameSession`. The
session owns environment and action-selection RNGs and advances one engine
decision at a time. Each configured model seat has its own search RNG. Existing
batch simulation APIs and training replay datasets retain their contracts.

## Player diagnostics

Probability-only players continue to implement `get_action_probabilities(state)`.
For diagnostics, implement the optional `DiagnosticPlayer` protocol:

```python
from skyjo.simulation.decisions import DiagnosticSection, PlayerDecision


def get_decision(self, state, *, diagnostics=True):
    probabilities, evaluation = self.evaluate_once(state)
    sections = (
        (
            DiagnosticSection(
                "Evaluation",
                text="Values describe the state before the selected action.",
                metrics={"Confidence": evaluation.confidence},
                columns=("Candidate", "Score"),
                rows=tuple(evaluation.candidates),
            ),
        )
        if diagnostics
        else ()
    )
    return PlayerDecision(probabilities, sections)
```

Probabilities must be finite, nonnegative, normalized, and supported on legal
actions. Diagnostic values are JSON scalars (strings, numbers, booleans, null),
table rows match their columns, and numbers must be finite. Players supply data,
not Textual widgets. The app renders text literally. Diagnostics must come from
the actual decision evaluation, without additional searches or RNG consumption.
Committed decisions are copied so subsequent player calls cannot alter history.
Sections may include `row_actions`, a tuple aligned with table rows: legal action
indices identify individual action rows, while `None` identifies group summaries.
Indices must be integers in the engine action range. Omit this metadata or leave
it empty for ordinary tables. The renderer marks the exact committed action;
pending proposals have no selected action until the session samples a move.

New search decisions use one action table with group totals followed by indented
member actions; singleton groups occupy one row. Model prior is the original
model policy (summed on group rows); search prior is the effective group prior,
including configured root noise. Visits and visit share (`visits / total visits`)
describe search usage before temperature. Selection probability is the actual
distribution from the configured action strategy, summed on group rows. The
default applies temperature to visits; other strategies select by value or a
Gumbel winner. Training probability shows the separate learning target. Search
value is the acting seat's estimated win probability; unevaluated groups display `—`. Group-only
statistics are blank on member rows because search does not evaluate them
independently. Root win estimates compare initial model and updated search values
in fixed seat order. Iterations, temperature, and whether symmetry merging was
actually enabled appear once above the action table. Even at large search budgets,
symmetry remains enabled unless an explored path encounters deck-recycling risk;
that single-tree search restarts without grouping. Replica searches retain the
conservative whole-budget safety check so every replica uses the same partition.
When replicas are enabled, a standard-error column measures search variability,
not model bias. See [search strategies](search-strategies.md).

Policy players show model prior alongside their greedy selection distribution.
Probability-only and human decisions use a simple action/probability table.
Numeric diagnostics display nine significant digits, with horizontal scrolling
for wide tables rather than truncating numbers. Player-supplied diagnostics take
precedence over the generic action-selection panel. Older recordings retain their
saved diagnostic tables; their sections are not reconstructed or migrated.

At every terminal width, boards sit above full-width diagnostics in a two-column
grid. The board pane fits its content up to half the available content height,
then scrolls to reach additional seats and human controls. Diagnostics take the
remaining height.

The highlighted card distinguishes the available discard, a drawn card, and a
card taken from the discard. The draw-pile count excludes hidden board cards from
the engine's unseen pool; both the unseen total and hidden count are also shown.
If that pool has fewer cards than the hidden boards, the draw-pile display bottoms
out at zero; gameplay continues to follow the engine's existing recycling rules.

## Recording version 2

The public API is `simulation.recording.save(recording, path)` / `load(path)`,
with `to_dict` / `from_dict` for converters. The format has no Torch or Textual
dependency. JSON writes replace files atomically and reject nonfinite values.

Top-level fields:

| Field | Meaning |
| --- | --- |
| `format`, `version` | `skyjo.game-recording`, `2` |
| `created_at` | UTC ISO timestamp |
| `players` | Fixed seat order; name, kind, settings, and checkpoint paths/digests when applicable |
| `seed` | Session seed |
| `complete` | Whether the final frame ends the full game |
| `frames` | Chronological full-state snapshots |

Each frame contains `round_number` (one-based), `event` (`decision`, `round_end`,
or `game_end`), and the complete engine `state`: `game`, padded `table`, `deck`,
`players`, `turn`, `pending_card`, and `countdown`. Arrays are nested integer
lists. State remains in the engine's current-player orientation; rendering
converts to fixed seats. Card indices retain the engine's encoding, not printed
card values. Hidden cards are unknown until sampled by the engine.

A committed decision frame contains its actual `action` (engine mask index),
`probabilities` (28 entries), optional `training_policy` (28 entries), and optional
`diagnostics` sections. The action is sampled from `probabilities`; a separate
training policy supplies the learning target. When absent, learning uses the
action distribution. Both distributions must be normalized over legal actions,
but the chosen action may have zero probability in the training target. These
describe the frame's pre-action state. The next frame stores the observed resulting state.
An unfinished recording ends with an uncommitted decision snapshot, whose action
and probabilities are null. Diagnostics default to an empty list.
Optional diagnostic `row_actions` metadata is supported in both recording versions;
recordings without it continue to load and render their original sections.

Round/game-end frames have null actions and probabilities, explicit fixed-seat
`round_scores` and `cumulative_scores`, and `winners` (zero-based seat indices)
only at game end. A nonfinal round-end frame is followed by the next round's
initial decision snapshot. Initial dealing is already reflected in that snapshot.

Loading validates version, state dimensions/invariants, action legality,
probabilities, diagnostics, round ordering, and outcome consistency. Playback
reads snapshots; it does not verify stochastic transitions by resimulation.
Version-1 recordings remain readable, with no separate training policy. Unknown
versions fail explicitly. Model files are not required for playback.
Recordings are for inspection, not exact RNG continuation or game branching.

## Presets and execution

Version-1 matchup TOML contains `seed` and 2–8 `[[seats]]` tables. Each seat uses
`name`, `kind` (`human`, `random`, `policy`, `search`), optional `checkpoint`,
`iterations`, `temperature`, `boundary_checkpoint`, and `[seats.search]` with the
existing `SearchConfig` fields. Paths loaded from presets resolve relative to the
preset; saved presets use absolute paths. Model configuration is validated through
the existing contestant factory, including incompatible policy/search options.

`uniform_group_priors = true` replaces the policy prior with equal probability
for every effective search group at every decision node. Model value estimates
remain in use. Diagnostics retain the original **Model prior** and show the
effective **Search prior**, with **Prior: Uniform groups** identifying the mode.
Use `dirichlet_epsilon = 0` for strictly uniform search priors; configured root
noise is otherwise mixed in as usual. If symmetry merging is disabled or falls
back for recycling safety, each legal action forms its own singleton group.

Checkpoint loading and player evaluation run in Textual thread workers. Each
session allows one outstanding proposal. Pausing buffers a completed proposal;
it does not apply a move. Leaving a table invalidates callbacks from that session.
The in-flight computation may finish in the background. UI errors pause automatic
progression and expose a retry or setup path. Saving uses the committed history;
pending proposals are not part of the recording.

Historical imports, experiment hooks, alternative-model analysis, and resuming
recordings are intentionally left for later work against this format.
