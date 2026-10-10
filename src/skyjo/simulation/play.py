"""Play games with supplied players and retain observed histories."""

from __future__ import annotations

import dataclasses
import logging
import typing

import numpy as np

from skyjo.engine import game as sj
from skyjo.engine import values

from . import player

logger = logging.getLogger(__name__)

# MARK: Types

type FloatArray = np.ndarray[tuple[int, ...], np.float32]
type ActionProbabilities = FloatArray


class RoundHistoryEntry(typing.NamedTuple):
    """One observed decision point from self-play.

    Terminal entries use None for action and action_probabilities.
    """

    state: sj.Skyjo
    action: sj.SkyjoAction | None
    action_probabilities: ActionProbabilities | None


type RoundHistory = list[RoundHistoryEntry]


@dataclasses.dataclass(frozen=True, slots=True)
class RoundResult:
    """A completed round, with scores and ending player in fixed player order."""

    history: RoundHistory
    round_scores: tuple[int, ...]
    cumulative_scores: tuple[int, ...]
    ending_player: int


@dataclasses.dataclass(frozen=True, slots=True)
class GameResult:
    """A full game's ordered rounds and outcome in fixed player order."""

    rounds: tuple[RoundResult, ...]

    @property
    def final_scores(self) -> tuple[int, ...]:
        return self.rounds[-1].cumulative_scores

    @property
    def winners(self) -> tuple[int, ...]:
        best_score = min(self.final_scores)
        return tuple(
            i for i, score in enumerate(self.final_scores) if score == best_score
        )


def print_round_history(
    game_history: RoundHistory,
):
    for game_state, action, action_probabilities in game_history:
        print(sj.visualize_state(game_state))
        if action is not None:
            print(f"ACTION: {sj.get_action_name(action)}")
        print(f"ACTION PROBABILITIES: {action_probabilities}")


# MARK: Selfplay


def play_round(
    players: list[player.Player],
    debug: bool = False,
    start_state: sj.Skyjo | None = None,
    *,
    environment_rng: np.random.Generator | None = None,
    action_rng: np.random.Generator | None = None,
) -> RoundHistory:
    """Play one round, retaining each decision and its final snapshot."""
    environment_rng = (
        environment_rng if environment_rng is not None else np.random.default_rng()
    )
    action_rng = action_rng if action_rng is not None else np.random.default_rng()
    if start_state is None:
        game_state = sj.new(players=len(players), rng=environment_rng)
        game_state = sj.start_round(game_state, rng=environment_rng)
    else:
        game_state = start_state
    game_history = []

    if debug:
        logger.info(f"{sj.visualize_state(game_state)}")
    while not sj.get_round_over(game_state):
        action_probabilities = players[
            sj.get_player(game_state)
        ].get_action_probabilities(game_state)
        action = action_rng.choice(sj.MASK_SIZE, p=action_probabilities)
        assert sj.actions(game_state)[action]
        game_history.append(RoundHistoryEntry(game_state, action, action_probabilities))
        game_state = sj.apply_action(game_state, action, rng=environment_rng)
        if debug:
            print(sj.get_action_name(action))
            logger.info(f"ACTION PROBABILITIES\n{action_probabilities}")
            logger.info(f"ACTION: {sj.get_action_name(action)}")
            logger.info(f"{sj.visualize_state(game_state)}")

    game_history.append(RoundHistoryEntry(game_state, None, None))
    if debug:
        outcome = values.skyjo_to_state_value(game_state)
        fixed_perspective_score = sj.get_fixed_perspective_round_scores(game_state)
        logger.info("ROUND OVER")
        logger.info(f"OUTCOME: {outcome}")
        logger.info(f"SCORES: {fixed_perspective_score}")
        logger.info(f"TOTAL TURNS: {sj.get_turn(game_state)}")
    return game_history


def play_game(
    players: list[player.Player],
    debug: bool = False,
    start_state: sj.Skyjo | None = None,
    *,
    environment_rng: np.random.Generator | None = None,
    action_rng: np.random.Generator | None = None,
) -> GameResult:
    """Play to 100 or more from a fresh game or an optional active round."""
    if not 2 <= len(players) <= sj.PLAYER_COUNT:
        raise ValueError(f"Expected between 2 and {sj.PLAYER_COUNT} players")

    environment_rng = (
        environment_rng if environment_rng is not None else np.random.default_rng()
    )
    action_rng = action_rng if action_rng is not None else np.random.default_rng()
    state = (
        sj.start_round(
            sj.new(players=len(players), rng=environment_rng), rng=environment_rng
        )
        if start_state is None
        else start_state
    )
    if sj.get_player_count(state) != len(players) or sj.get_round_over(state):
        raise ValueError("Expected an active round matching the supplied players")
    rounds = []
    while True:
        history = play_round(
            players,
            debug=debug,
            start_state=state,
            environment_rng=environment_rng,
            action_rng=action_rng,
        )
        state = history[-1].state
        rounds.append(
            RoundResult(
                history=history,
                round_scores=tuple(
                    map(int, sj.get_fixed_perspective_round_scores(state))
                ),
                cumulative_scores=tuple(
                    map(int, sj.get_fixed_perspective_game_scores(state))
                ),
                ending_player=sj.get_player(state),
            )
        )
        if sj.get_game_over(state):
            result = GameResult(rounds=tuple(rounds))
            if debug:
                logger.info("GAME OVER")
                logger.info(f"SCORES: {result.final_scores}")
                logger.info(f"WINNERS: {result.winners}")
            return result
        state = sj.start_next_round(state, rng=environment_rng)
