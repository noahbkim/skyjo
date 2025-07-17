import logging
import typing

import player
import skyjo as sj
import skynet


def single_game_faceoff(
    players: list[player.AbstractPlayer],
    start_state: sj.Skyjo | None = None,
    debug: bool = False,
):
    if start_state is None:
        start_state = sj.new(players=len(players))
        start_state = sj.start_round(start_state)
    game_state = start_state

    if debug:
        print(sj.visualize_state(game_state))

    while not sj.get_game_over(game_state):
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
    return skynet.skyjo_to_state_value(
        game_state
    ), sj.get_fixed_perspective_round_scores(game_state)


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
