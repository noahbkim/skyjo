# Skyjo

AI model, training, and gameplay for Skyjo

## Usage

Install the project in editable mode while developing:

```sh
uv sync
```

Then import the core game API directly from `skyjo`, or import supporting
modules from the package:

```python
import skyjo as sj
from skyjo import play, skynet

state = sj.new(players=2)
model_cls = skynet.EquivariantSkyNet
```

## Full games and individual rounds

Play a full game with the existing player implementations:

```python
from skyjo import play, player

result = play.play_game([player.RandomPlayer(), player.RandomPlayer()])
print(result.final_scores)  # Cumulative scores in fixed player order
print(result.winners)       # All winning player indices, including ties

for round_result in result.rounds:
    print(round_result.round_scores, round_result.cumulative_scores)
    print(round_result.ending_player)
    history = round_result.history  # Decisions followed by the final snapshot
```

A game ends after a fully scored round brings any cumulative score to **100
or more**. The lowest cumulative total wins, with shared winners for ties.
Player zero starts the first round; the player who ends a round starts the
next one. Existing round penalties still apply.

`play.play_round(players)` plays just one round. `play.play(players)` remains
a compatibility alias, and existing training, search, and faceoff routines
continue to operate on individual rounds. `RoundHistory` names that history
format explicitly; the legacy `GameHistory` name remains available. Each
`RoundResult.history` can be passed to `play.game_history_to_game_data()` for
existing round statistics and targets. AI objectives and model formats have
not changed.

For direct state control, `sj.get_round_over(state)` identifies a completed
round, while **`sj.get_game_over(state)` now checks the full-game threshold**.
`sj.apply_action()` stops at the completed round; call
`sj.start_next_round(state)` explicitly to reset and deal another round. It
raises `ValueError` if the round is still active or the game has finished.
Completed rounds have no legal actions and remain available for inspection.

`sj.get_game_scores(state)` returns cumulative scores in current-player
order; `sj.get_fixed_perspective_game_scores(state)` uses fixed player order.
During a round they include only previous rounds; at completion they also
include the current round's penalized points. Reading them never mutates the
state. The stored `GAME_SCORES` slots always exclude the current round.

`sj.get_round_about_to_end(state)` indicates that the next action ends the
round. The old `sj.get_game_about_to_end()` name remains a compatibility alias
for that round condition. Existing winner helpers still report round winners;
use `GameResult.winners` for full-game winners.
