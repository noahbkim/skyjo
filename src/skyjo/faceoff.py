import dataclasses
import logging
import random
import typing

import numpy as np
import torch

from . import config, player, predictor
from . import game as sj
from . import skynet


@dataclasses.dataclass(slots=True)
class MCTSPromotionConfig(config.Config):
    model_player_config: player.ModelPlayerConfig
    paired_rounds: int = 100
    seed: int = 0


@dataclasses.dataclass(frozen=True, slots=True)
class MCTSFaceoffResult:
    candidate_wins: int = 0
    champion_wins: int = 0
    candidate_score_total: float = 0.0
    champion_score_total: float = 0.0
    candidate_clears: float = 0.0
    champion_clears: float = 0.0

    @property
    def games(self) -> int:
        return self.candidate_wins + self.champion_wins

    @property
    def candidate_mean_score(self) -> float:
        return self.candidate_score_total / self.games

    @property
    def champion_mean_score(self) -> float:
        return self.champion_score_total / self.games

    @property
    def candidate_mean_clears(self) -> float:
        return self.candidate_clears / self.games

    @property
    def champion_mean_clears(self) -> float:
        return self.champion_clears / self.games

    def __add__(self, other: typing.Self) -> typing.Self:
        return type(self)(
            candidate_wins=self.candidate_wins + other.candidate_wins,
            champion_wins=self.champion_wins + other.champion_wins,
            candidate_score_total=(
                self.candidate_score_total + other.candidate_score_total
            ),
            champion_score_total=(
                self.champion_score_total + other.champion_score_total
            ),
            candidate_clears=self.candidate_clears + other.candidate_clears,
            champion_clears=self.champion_clears + other.champion_clears,
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, tuple) and len(other) == 2:
            return (self.candidate_wins, self.champion_wins) == other
        if not isinstance(other, MCTSFaceoffResult):
            return NotImplemented
        return all(
            getattr(self, field.name) == getattr(other, field.name)
            for field in dataclasses.fields(self)
        )


def single_game_faceoff(
    players: list[player.AbstractPlayer],
    start_state: sj.Skyjo | None = None,
    debug: bool = False,
    return_cleared_columns: bool = False,
):
    if start_state is None:
        start_state = sj.new(players=len(players))
        start_state = sj.start_round(start_state)
    game_state = start_state

    if debug:
        print(sj.visualize_state(game_state))

    while not sj.get_round_over(game_state):
        player = players[sj.get_player(game_state)]
        action = player.get_action(game_state)
        assert sj.actions(game_state)[action]
        game_state = sj.apply_action(game_state, action)
        assert sj.validate(game_state)

        if debug:
            logging.info(f"action: {sj.get_action_name(action)}")
            logging.info(sj.visualize_state(game_state))

    if debug:
        logging.info(f"Winner: {sj.get_fixed_perspective_winner(game_state)}")
        logging.info(
            f"Round scores: {sj.get_fixed_perspective_round_scores(game_state)}"
        )
    result = (
        skynet.skyjo_to_state_value(game_state),
        sj.get_fixed_perspective_round_scores(game_state),
    )
    if return_cleared_columns:
        return (
            *result,
            sj.get_fixed_perspective_cleared_columns(game_state).sum(axis=1),
        )
    return result


def model_policy_single_game_faceoff(
    model1: skynet.SkyNet,
    model2: skynet.SkyNet,
    temperature: float = 0.1,
    start_state: sj.Skyjo | None = None,
):
    model1_player = player.PureModelPolicyPlayer(
        model1,
        temperature,
    )
    model2_player = player.PureModelPolicyPlayer(
        model2,
        temperature,
    )
    return single_game_faceoff([model1_player, model2_player], start_state)


def model_value_single_game_faceoff(
    model1: skynet.SkyNet,
    model2: skynet.SkyNet,
    terminal_state_rollouts: int = 10,
    start_state: sj.Skyjo | None = None,
):
    model1_player = player.PureModelValuePlayer(
        model1, terminal_state_rollouts=terminal_state_rollouts
    )
    model2_player = player.PureModelValuePlayer(
        model2, terminal_state_rollouts=terminal_state_rollouts
    )
    return single_game_faceoff([model1_player, model2_player], start_state)


def model_policy_faceoff(
    model1: skynet.SkyNet,
    model2: skynet.SkyNet,
    rounds: int = 1,
    temperature: float = 0.0,
    start_state_generator: typing.Callable[[], sj.Skyjo] | None = None,
):
    model1_wins, model2_wins = 0, 0
    model1_point_differential, model2_point_differential = 0, 0
    for _ in range(rounds):
        start_state = (
            start_state_generator() if start_state_generator is not None else None
        )
        outcome, round_scores = model_policy_single_game_faceoff(
            model1,
            model2,
            temperature,
            start_state,
        )
        if skynet.state_value_for_player(outcome, 0) == 1:
            model1_wins += 1
            assert round_scores[0] <= round_scores[1]
            model2_point_differential += int(round_scores[1] - round_scores[0])
        else:
            model2_wins += 1
            assert round_scores[1] <= round_scores[0]
            model1_point_differential += int(round_scores[0] - round_scores[1])
        outcome2, round_scores2 = model_policy_single_game_faceoff(
            model2,
            model1,
            temperature,
            start_state,
        )
        if skynet.state_value_for_player(outcome2, 0) == 1:
            model2_wins += 1
            assert round_scores2[0] <= round_scores2[1]
            model1_point_differential += int(round_scores2[1] - round_scores2[0])
        else:
            model1_wins += 1
            assert round_scores2[1] <= round_scores2[0]
            model2_point_differential += int(round_scores2[0] - round_scores2[1])
    logging.info(f"Model 1 wins: {model1_wins}, Model 2 wins: {model2_wins}")
    logging.info(
        f"Model 1 avg point differential: {model1_point_differential / (2 * rounds)} Model 2 avg point differential: {model2_point_differential / (2 * rounds)}"
    )
    return model1_wins, model2_wins


def model_value_faceoff(
    model1: skynet.SkyNet,
    model2: skynet.SkyNet,
    rounds: int = 1,
    terminal_state_rollouts: int = 10,
    start_state_generator: typing.Callable[[], sj.Skyjo] | None = None,
):
    model1_wins, model2_wins = 0, 0
    model1_point_differential, model2_point_differential = 0, 0
    for _ in range(rounds):
        start_state = (
            start_state_generator() if start_state_generator is not None else None
        )
        outcome, round_scores = model_value_single_game_faceoff(
            model1,
            model2,
            terminal_state_rollouts,
            start_state,
        )
        if skynet.state_value_for_player(outcome, 0) == 1:
            model1_wins += 1
            assert round_scores[0] <= round_scores[1]
            model2_point_differential += int(round_scores[1] - round_scores[0])
        else:
            model2_wins += 1
            assert round_scores[1] <= round_scores[0]
            model1_point_differential += int(round_scores[0] - round_scores[1])
        outcome2, round_scores2 = model_value_single_game_faceoff(
            model2,
            model1,
            terminal_state_rollouts,
            start_state,
        )
        if skynet.state_value_for_player(outcome2, 0) == 1:
            model2_wins += 1
            assert round_scores2[0] <= round_scores2[1]
            model1_point_differential += int(round_scores2[1] - round_scores2[0])
        else:
            model1_wins += 1
            assert round_scores2[1] <= round_scores2[0]
            model2_point_differential += int(round_scores2[0] - round_scores2[1])
    logging.info(f"Model 1 wins: {model1_wins}, Model 2 wins: {model2_wins}")
    logging.info(
        f"Model 1 avg point differential: {model1_point_differential / (2 * rounds)} Model 2 avg point differential: {model2_point_differential / (2 * rounds)}"
    )
    return model1_wins, model2_wins


def model_mcts_faceoff(
    candidate: skynet.SkyNet,
    champion: skynet.SkyNet,
    model_player_config: player.ModelPlayerConfig,
    paired_rounds: int = 100,
    seed: int = 0,
    start_state_generator: typing.Callable[[], sj.Skyjo] | None = None,
    game_completed_callback: typing.Callable[[], None] | None = None,
    champion_model_player_config: player.ModelPlayerConfig | None = None,
) -> tuple[int, int]:
    """Evaluate deployed MCTS agents with common seeds and swapped seats."""
    result = model_mcts_faceoff_detailed(
        candidate,
        champion,
        model_player_config=model_player_config,
        champion_model_player_config=champion_model_player_config,
        paired_rounds=paired_rounds,
        seed=seed,
        start_state_generator=start_state_generator,
        game_completed_callback=game_completed_callback,
    )
    return result.candidate_wins, result.champion_wins


def model_mcts_faceoff_detailed(
    candidate: skynet.SkyNet,
    champion: skynet.SkyNet,
    model_player_config: player.ModelPlayerConfig,
    paired_rounds: int = 100,
    seed: int = 0,
    start_state_generator: typing.Callable[[], sj.Skyjo] | None = None,
    game_completed_callback: typing.Callable[[], None] | None = None,
    champion_model_player_config: player.ModelPlayerConfig | None = None,
) -> MCTSFaceoffResult:
    """Return win, score, and clear metrics with common seeds and swapped seats."""
    if paired_rounds < 1:
        raise ValueError("paired_rounds must be at least one")
    candidate_evaluation_config = dataclasses.replace(
        model_player_config, action_softmax_temperature=0.0
    )
    champion_evaluation_config = dataclasses.replace(
        champion_model_player_config or model_player_config,
        action_softmax_temperature=0.0,
    )
    candidate_player = player.ModelPlayer(
        predictor.LocalPredictorClient(candidate, max_batch_size=512),
        **candidate_evaluation_config.kwargs(),
    )
    champion_player = player.ModelPlayer(
        predictor.LocalPredictorClient(champion, max_batch_size=512),
        **champion_evaluation_config.kwargs(),
    )
    candidate_wins = champion_wins = 0
    candidate_score_total = champion_score_total = 0.0
    candidate_clears = champion_clears = 0.0
    for pair_index in range(paired_rounds):
        pair_seed = seed + pair_index
        for candidate_seat in (0, 1):
            random.seed(pair_seed)
            np.random.seed(pair_seed)
            torch.manual_seed(pair_seed)
            start_state = (
                start_state_generator() if start_state_generator is not None else None
            )
            players = (
                [candidate_player, champion_player]
                if candidate_seat == 0
                else [champion_player, candidate_player]
            )
            try:
                outcome, round_scores, cleared_columns = single_game_faceoff(
                    players,
                    start_state=start_state,
                    return_cleared_columns=True,
                )
            except TypeError as error:
                if "return_cleared_columns" not in str(error):
                    raise
                outcome, round_scores = single_game_faceoff(
                    players,
                    start_state=start_state,
                )
                cleared_columns = np.zeros(len(players), dtype=np.float32)
            winner = int(np.argmax(outcome))
            if winner == candidate_seat:
                candidate_wins += 1
            else:
                champion_wins += 1
            champion_seat = 1 - candidate_seat
            candidate_score_total += float(round_scores[candidate_seat])
            champion_score_total += float(round_scores[champion_seat])
            candidate_clears += float(cleared_columns[candidate_seat])
            champion_clears += float(cleared_columns[champion_seat])
            if game_completed_callback is not None:
                game_completed_callback()
    return MCTSFaceoffResult(
        candidate_wins=candidate_wins,
        champion_wins=champion_wins,
        candidate_score_total=candidate_score_total,
        champion_score_total=champion_score_total,
        candidate_clears=candidate_clears,
        champion_clears=champion_clears,
    )


def passes_mcts_promotion(
    candidate: skynet.SkyNet,
    champion: skynet.SkyNet,
    promotion_config: MCTSPromotionConfig,
    start_state_generator: typing.Callable[[], sj.Skyjo] | None = None,
) -> bool:
    candidate_wins, champion_wins = model_mcts_faceoff(
        candidate,
        champion,
        promotion_config.model_player_config,
        paired_rounds=promotion_config.paired_rounds,
        seed=promotion_config.seed,
        start_state_generator=start_state_generator,
    )
    return candidate_wins > champion_wins
