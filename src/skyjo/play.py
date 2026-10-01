"""Module for playing Skyjo games and generating training data."""

from __future__ import annotations

import dataclasses
import logging
import typing

import numpy as np

from . import game as sj
from . import player, skynet

# MARK: Types

FloatArray: typing.TypeAlias = np.ndarray[tuple[int, ...], np.float32]
ActionProbabilities: typing.TypeAlias = FloatArray


class RoundHistoryEntry(typing.NamedTuple):
    """One observed decision point from self-play.

    Terminal entries use None for action and action_probabilities.
    """

    state: sj.Skyjo
    action: sj.SkyjoAction | None
    action_probabilities: ActionProbabilities | None


RoundHistory: typing.TypeAlias = list[RoundHistoryEntry]


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


class GameDataPoint(typing.NamedTuple):
    """One generated training row.

    Targets are intentionally opaque to gameplay generation. Training utilities
    decide how to interpret them for a specific model/loss setup.
    """

    state: sj.Skyjo
    action: sj.SkyjoAction | None
    targets: typing.Any


GameData: typing.TypeAlias = list[GameDataPoint]


@dataclasses.dataclass(slots=True)
class GameStats:
    game_length: int
    outcome_state_value: skynet.StateValue
    scores_state_value: skynet.StateValue
    action_counts: np.ndarray[tuple[int], np.float32]
    action_possibility_counts: np.ndarray[tuple[int], np.float32]
    clear_count: int
    flip_count: int
    flip_possibility_count: int
    replace_face_up_count: int
    replace_face_down_count: int
    replace_possibility_count: int

    def to_record_dict(self) -> dict[str, typing.Any]:
        record_dict = {
            "game_length": self.game_length,
            "clear_count": self.clear_count,
        }
        for i, outcome in enumerate(self.outcome_state_value):
            record_dict[f"outcome_{i}"] = outcome
        for i, score in enumerate(self.scores_state_value):
            record_dict[f"score_{i}"] = score

        flip_rate = self.flip_count / max(self.flip_possibility_count, 1)
        replace_face_up_rate = self.replace_face_up_count / max(
            self.replace_possibility_count, 1
        )
        replace_face_down_rate = self.replace_face_down_count / max(
            self.replace_possibility_count, 1
        )
        record_dict["flip_rate"] = flip_rate
        record_dict["replace_face_up_rate"] = replace_face_up_rate
        record_dict["replace_face_down_rate"] = replace_face_down_rate

        action_rates = self.action_counts / np.maximum(
            self.action_possibility_counts, 1
        )
        for i in range(sj.MASK_SIZE):
            record_dict[f"{sj.get_action_name(i)}"] = action_rates[i]
        return record_dict


def _round_history_to_game_data(
    game_history: RoundHistory,
    outcome_state_value: np.ndarray,
    fixed_perspective_score: np.ndarray,
) -> tuple[GameData, GameStats]:
    training_data = []
    action_counts = np.zeros(sj.MASK_SIZE, dtype=np.float32)
    action_possibility_counts = np.zeros(sj.MASK_SIZE, dtype=np.float32)
    flip_count, flip_possibility_count = 0, 0
    replace_face_up_count, replace_face_down_count, replace_possibility_count = 0, 0, 0
    for game_state, action, mcts_probs in game_history[:-1]:
        action_mask = sj.actions(game_state).astype(np.float32)
        assert action is not None, "expected non-terminal action"
        assert mcts_probs is not None, "expected non-terminal action probabilities"
        player = sj.get_player(game_state)
        targets = {
            "value": np.roll(outcome_state_value, -player),
            "policy": skynet.symmetrize_policy_target(
                game_state,
                mcts_probs,
            ),
        }
        training_data.append(
            GameDataPoint(
                game_state,  # game
                action,  # realized action
                targets,
            )
        )
        action_counts[action] += 1
        action_possibility_counts += action_mask
        if action < sj.MASK_FLIP:
            continue

        # Always possible to replace a face up card or face down card
        replace_possibility_count += 1
        if np.any(action_mask[sj.MASK_FLIP : sj.MASK_FLIP + sj.FINGER_COUNT]):
            flip_possibility_count += 1

        if sj.MASK_FLIP <= action < sj.MASK_REPLACE:
            flip_count += 1
        else:
            row, col = divmod(action - sj.MASK_REPLACE, sj.COLUMN_COUNT)
            if sj.get_finger(game_state, row, col, 0) == sj.FINGER_HIDDEN:
                replace_face_down_count += 1
            else:
                replace_face_up_count += 1

    cleared_cards = (
        sj.get_table(game_history[-1].state)[:, :, :, sj.FINGER_CLEARED].sum()
    )
    assert cleared_cards % 3 == 0, (
        f"Cleared cards is not divisible by 3: {cleared_cards}"
    )
    clear_count = cleared_cards // 3
    game_stats = GameStats(
        game_length=len(game_history) - 1,
        outcome_state_value=outcome_state_value,
        scores_state_value=fixed_perspective_score,
        action_counts=action_counts,
        action_possibility_counts=action_possibility_counts,
        clear_count=clear_count,
        flip_count=flip_count,
        flip_possibility_count=flip_possibility_count,
        replace_face_up_count=replace_face_up_count,
        replace_face_down_count=replace_face_down_count,
        replace_possibility_count=replace_possibility_count,
    )
    return training_data, game_stats


def game_result_to_game_data(result: GameResult) -> tuple[GameData, GameStats]:
    """Label all decisions with the observed game result, without outcome resampling."""
    if not result.rounds:
        raise ValueError("Expected a completed full game")
    final_state = result.rounds[-1].history[-1].state
    outcome = skynet.skyjo_to_game_state_value(final_state)
    scores = sj.get_fixed_perspective_game_scores(final_state)
    data = []
    stats = None
    for round_result in result.rounds:
        history = round_result.history
        if len(history) < 2 or not sj.get_round_over(history[-1].state):
            raise ValueError("Expected a completed round with decisions")
        rows, round_stats = _round_history_to_game_data(history, outcome, scores)
        data.extend(rows)
        if stats is None:
            stats = round_stats
        else:
            for field in dataclasses.fields(GameStats):
                if field.name not in ("outcome_state_value", "scores_state_value"):
                    setattr(
                        stats, field.name,
                        getattr(stats, field.name) + getattr(round_stats, field.name),
                    )
    assert stats is not None
    return data, stats


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
    players: list[player.AbstractPlayer],
    debug: bool = False,
    start_state: sj.Skyjo | None = None,
) -> RoundHistory:
    """Play one round, retaining each decision and its final snapshot."""
    if start_state is None:
        game_state = sj.new(players=len(players))
        game_state = sj.start_round(game_state)
    else:
        game_state = start_state
    game_history = []

    if debug:
        logging.info(f"{sj.visualize_state(game_state)}")
    while not sj.get_round_over(game_state):
        action_probabilities = players[
            sj.get_player(game_state)
        ].get_action_probabilities(game_state)
        action = np.random.choice(sj.MASK_SIZE, p=action_probabilities)
        assert sj.actions(game_state)[action]
        game_history.append(RoundHistoryEntry(game_state, action, action_probabilities))
        game_state = sj.apply_action(game_state, action)
        if debug:
            print(sj.get_action_name(action))
            logging.info(f"ACTION PROBABILITIES\n{action_probabilities}")
            logging.info(f"ACTION: {sj.get_action_name(action)}")
            logging.info(f"{sj.visualize_state(game_state)}")

    game_history.append(RoundHistoryEntry(game_state, None, None))
    if debug:
        outcome = skynet.skyjo_to_state_value(game_state)
        fixed_perspective_score = sj.get_fixed_perspective_round_scores(game_state)
        logging.info("ROUND OVER")
        logging.info(f"OUTCOME: {outcome}")
        logging.info(f"SCORES: {fixed_perspective_score}")
        logging.info(f"TOTAL TURNS: {sj.get_turn(game_state)}")
    return game_history


def play_game(
    players: list[player.AbstractPlayer],
    debug: bool = False,
    start_state: sj.Skyjo | None = None,
) -> GameResult:
    """Play to 100 or more from a fresh game or an optional active round."""
    if not 2 <= len(players) <= sj.PLAYER_COUNT:
        raise ValueError(f"Expected between 2 and {sj.PLAYER_COUNT} players")

    state = (
        sj.start_round(sj.new(players=len(players)))
        if start_state is None else start_state
    )
    if sj.get_player_count(state) != len(players) or sj.get_round_over(state):
        raise ValueError("Expected an active round matching the supplied players")
    rounds = []
    while True:
        history = play_round(players, debug=debug, start_state=state)
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
                logging.info("GAME OVER")
                logging.info(f"SCORES: {result.final_scores}")
                logging.info(f"WINNERS: {result.winners}")
            return result
        state = sj.start_next_round(state)
