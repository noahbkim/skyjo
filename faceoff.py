import logging

import player
import skyjo as sj
import skynet


def single_game_faceoff(
    players: list[player.AbstractPlayer],
    debug: bool = False,
):
    game_state = sj.new(players=len(players))
    game_state = sj.start_round(game_state)

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


def model_single_game_faceoff(
    model1: skynet.SkyNet,
    model2: skynet.SkyNet,
    mcts_iterations: int = 10,
    temperature: float = 0.5,
    afterstate_initial_realizations: int = 10,
):
    model1_player = player.ModelPlayer(
        model1, temperature, mcts_iterations, afterstate_initial_realizations
    )
    model2_player = player.ModelPlayer(
        model2, temperature, mcts_iterations, afterstate_initial_realizations
    )
    return single_game_faceoff([model1_player, model2_player])


def model_faceoff(
    model1: skynet.SkyNet,
    model2: skynet.SkyNet,
    rounds: int = 1,
    mcts_iterations: int = 10,
    temperature: float = 0.0,
    afterstate_initial_realizations: int = 10,
):
    model1_wins, model2_wins = 0, 0
    model1_point_differential, model2_point_differential = 0, 0
    for _ in range(rounds):
        outcome, round_scores = model_single_game_faceoff(
            model1,
            model2,
            mcts_iterations,
            temperature,
            afterstate_initial_realizations,
        )
        if skynet.state_value_for_player(outcome, 0) == 1:
            model1_wins += 1
            assert round_scores[0] <= round_scores[1]
            model2_point_differential += round_scores[1] - round_scores[0]
        else:
            model2_wins += 1
            assert round_scores[1] <= round_scores[0]
            model1_point_differential += round_scores[0] - round_scores[1]
        outcome2, round_scores2 = model_single_game_faceoff(
            model2,
            model1,
            mcts_iterations,
            temperature,
            afterstate_initial_realizations,
        )
        if skynet.state_value_for_player(outcome2, 0) == 1:
            model2_wins += 1
            assert round_scores2[0] <= round_scores2[1]
            model1_point_differential += round_scores2[1] - round_scores2[0]
        else:
            model1_wins += 1
            assert round_scores2[1] <= round_scores2[0]
            model2_point_differential += round_scores2[0] - round_scores2[1]
    logging.info(
        f"Model 1 avg point differential: {model1_point_differential / (2 * rounds)} Model 2 avg point differential: {model2_point_differential / (2 * rounds)}"
    )
    return model1_wins, model2_wins
