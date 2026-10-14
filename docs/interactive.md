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

The model-policy panel differs from the recorded action-selection distribution:
a policy player chooses greedily, while search uses its configured action strategy.
The default applies temperature to visit counts; other strategies choose by values
or a Gumbel winner. Acting and learning distributions have separate panels.
The MCTS panel leads with raw counts and visit fractions (`visits / total visits`).
Search rows describe symmetry groups, including grouped priors, visits, values,
and optional replica standard errors. Unevaluated action values are absent.
Win estimates use fixed seat order. See [search strategies](search-strategies.md)
for composing these choices.
Numeric diagnostics display nine significant digits, with horizontal scrolling
for wide tables rather than truncating numbers. Player-supplied diagnostics take
precedence over the generic action-selection panel.

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
and policies are null. Diagnostics default to an empty list.

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

Checkpoint loading and player evaluation run in Textual thread workers. Each
session allows one outstanding proposal. Pausing buffers a completed proposal;
it does not apply a move. Leaving a table invalidates callbacks from that session.
The in-flight computation may finish in the background. UI errors pause automatic
progression and expose a retry or setup path. Saving uses the committed history;
pending proposals are not part of the recording.

Historical imports, experiment hooks, alternative-model analysis, and resuming
recordings are intentionally left for later work against this format.
