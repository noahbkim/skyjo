import dataclasses
import itertools
import logging
import multiprocessing as mp
import pickle as pkl
import random
import typing

import numpy as np
import torch
import torch.nn as nn

import explain
import mcts_new as mcts
import player
import skyjo as sj
import skynet


@dataclasses.dataclass(slots=True, frozen=True)
class DataBatch:
    spatial_inputs_tensor: torch.Tensor
    non_spatial_inputs_tensor: torch.Tensor
    policy_targets_tensor: torch.Tensor
    value_targets_tensor: torch.Tensor
    points_targets_tensor: torch.Tensor

    @classmethod
    def from_raw(
        cls, raw_data: list[tuple], device: torch.device = torch.device("cpu")
    ) -> typing.Self:
        non_spatial_inputs = torch.tensor(
            np.array([data[0] for data in raw_data]), dtype=torch.float32, device=device
        )
        spatial_inputs = torch.tensor(
            np.array([data[1] for data in raw_data]),
            dtype=torch.float32,
            device=device,
        )
        policy_targets = torch.tensor(
            np.array([data[2] for data in raw_data]),
            dtype=torch.float32,
            device=device,
        )
        value_targets = torch.tensor(
            np.array([data[3] for data in raw_data]),
            dtype=torch.float32,
            device=device,
        )
        points_targets = torch.tensor(
            np.array([data[4] for data in raw_data]),
            dtype=torch.float32,
            device=device,
        )
        return cls(
            spatial_inputs_tensor=spatial_inputs,
            non_spatial_inputs_tensor=non_spatial_inputs,
            policy_targets_tensor=policy_targets,
            value_targets_tensor=value_targets,
            points_targets_tensor=points_targets,
        )


def get_skyjo_symmetries(
    skyjo: sj.Skyjo, policy_probs: np.ndarray[tuple[int], np.uint8]
) -> list[tuple[sj.Skyjo, np.ndarray[tuple[int], np.uint8]]]:
    """Return a list of symmetrically equivalent `Skyjo` states and corresponding policy."""
    symmetries = []
    for col_order in itertools.permutations(range(sj.COLUMN_COUNT)):
        new_table = skyjo[1].copy()
        new_table[0] = new_table[0][:, col_order, :]
        new_policy_probs = policy_probs.copy()
        flip_policy = new_policy_probs[sj.MASK_FLIP : sj.MASK_FLIP + sj.FINGER_COUNT]
        replace_policy = new_policy_probs[
            sj.MASK_REPLACE : sj.MASK_REPLACE + sj.FINGER_COUNT
        ]
        new_policy_probs[sj.MASK_FLIP : sj.MASK_FLIP + sj.FINGER_COUNT] = (
            flip_policy.reshape(sj.ROW_COUNT, sj.COLUMN_COUNT)[:, col_order].reshape(-1)
        )
        new_policy_probs[sj.MASK_REPLACE : sj.MASK_REPLACE + sj.FINGER_COUNT] = (
            replace_policy.reshape(sj.ROW_COUNT, sj.COLUMN_COUNT)[:, col_order].reshape(
                -1
            )
        )
        symmetries.append(
            (
                (
                    skyjo[0],
                    new_table,
                    skyjo[2],
                    skyjo[3],
                    skyjo[4],
                    skyjo[5],
                    skyjo[6],
                    skyjo[7],
                ),
                new_policy_probs,
            )
        )
    return symmetries


def selfplay_game(
    model: skynet.SkyNet,
    players: int = 2,
    mcts_iterations: int = 10,
    mcts_temperature: float = 0.0,
    afterstate_initial_realizations: int = 10,
    debug=False,
) -> list[tuple]:
    model.eval()
    with torch.no_grad():
        game_state = sj.new(players=players)
        game_state = sj.start_round(game_state)
        countdown = game_state[6]
        game_data = []
        while countdown != 0:
            node = mcts.run_mcts(
                game_state,
                model,
                mcts_iterations,
                afterstate_initial_realizations=afterstate_initial_realizations,
            )

            sample_temperature = mcts_temperature
            mcts_probs = node.sample_child_visit_probabilities(
                temperature=sample_temperature
            )
            choice = np.random.choice(sj.MASK_SIZE, p=mcts_probs)
            assert sj.actions(game_state)[choice]
            game_data.append((game_state, mcts_probs))
            symmetries = get_skyjo_symmetries(game_state, mcts_probs)
            for symmetry in symmetries:
                game_data.append((symmetry[0], symmetry[1]))
            if debug:
                logging.info(f"curr facedown: {sj.get_facedown_count(game_state, 0)}")
                logging.info(f"other facedown: {sj.get_facedown_count(game_state, 1)}")
                logging.info(f"curr score: {sj.get_score(game_state, 0)}")
                logging.info(f"other score: {sj.get_score(game_state, 1)}")
                logging.info(
                    f"MCTS ACTION PROBS: {
                        [
                            f'{sj.get_action_name(action)}: {prob}'
                            for action, prob in enumerate(
                                node.sample_child_visit_probabilities(temperature=1.0)
                            )
                            if prob > 0
                        ]
                    }"
                )
                logging.info(
                    f"CHOICE (temp = {sample_temperature}): {sj.get_action_name(choice)}"
                )
            game_state = sj.apply_action(game_state, choice)
            countdown = game_state[6]
        outcome_state_value = skynet.skyjo_to_state_value(game_state)
        fixed_perspective_score = sj.get_fixed_perspective_round_scores(game_state)
        if debug:
            logging.info(f"Outcome: {outcome_state_value}")
            logging.info(f"Scores: {fixed_perspective_score}")
            logging.info(f"total turns: {sj.get_turn(game_state)}")
        return [
            (
                data[0][0],  # game
                data[0][1][:players],  # table
                data[1],  # policy target
                np.roll(outcome_state_value, sj.get_player(data[0])),  # outcome target
                np.roll(
                    fixed_perspective_score,
                    sj.get_player(data[0]),
                ),  # points target
            )
            for data in game_data
        ]


def multiprocessed_selfplay(
    episodes: int,
    processes: int,
    model: skynet.SkyNet,
    players: int,
    mcts_temperature: float,
    mcts_iterations: int,
    afterstate_initial_realizations: int,
):
    with mp.Pool(processes=processes) as pool:
        res_list = pool.starmap(
            selfplay_game,
            [
                (
                    model,
                    players,
                    mcts_iterations,
                    mcts_temperature,
                    afterstate_initial_realizations,
                )
                for _ in range(episodes)
            ],
        )
    return list(itertools.chain.from_iterable(res_list))


def train(
    model: skynet.SkyNet,
    training_epochs: int,
    training_data: list[tuple],
    batch_size=128,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    policy_losses = []
    value_losses = []
    points_losses = []
    total_losses = []
    model.train()

    for _ in range(training_epochs):
        # Create training data batches
        random.shuffle(training_data)
        batches = [
            DataBatch.from_raw(
                training_data[i * batch_size : (i + 1) * batch_size],
                device=model.device,
            )
            for i in range(len(training_data) // batch_size)
        ]
        for _, batch in enumerate(batches):
            (
                torch_predicted_value,
                torch_predicted_points,
                torch_predicted_policy,
            ) = model(batch.spatial_inputs_tensor, batch.non_spatial_inputs_tensor)

            policy_loss = skynet.compute_policy_loss(
                torch_predicted_policy, batch.policy_targets_tensor
            )
            value_loss = skynet.compute_value_loss(
                torch_predicted_value, batch.value_targets_tensor
            )
            value_loss_scale = 5
            points_loss = nn.L1Loss()(
                torch_predicted_points,
                batch.points_targets_tensor,
            )
            points_loss_scale = 1 / 1000
            total_loss = (
                value_loss_scale * value_loss
                + points_loss_scale * points_loss
                + policy_loss
            )

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss_scale * value_loss.item())
            points_losses.append(points_loss_scale * points_loss.item())
            total_losses.append(total_loss.item())
            # compute gradient and do SGD step
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        logging.info(
            f"value loss: {sum(value_losses) / len(value_losses)} "
            f"points loss: {sum(points_losses) / len(points_losses)} "
            f"policy loss: {sum(policy_losses) / len(policy_losses)} "
            f"total loss: {sum(total_losses) / len(total_losses)} "
        )


def single_game_faceoff(
    players: list[player.AbstractPlayer],
):
    game_state = sj.new(players=len(players))
    game_state = sj.start_round(game_state)
    while not sj.get_game_over(game_state):
        # print(sj.visualize_state(game_state))
        player = players[sj.get_player(game_state)]
        action = player.get_action(game_state)
        # print(sj.get_action_name(action))
        assert sj.actions(game_state)[action]
        game_state = sj.apply_action(game_state, action)
        assert sj.validate(game_state)
    # print(sj.visualize_state(game_state))
    return skynet.skyjo_to_state_value(
        game_state
    ), sj.get_fixed_perspective_round_scores(game_state)


def gen_single_game_data(players: list[player.AbstractPlayer]):
    states_data = []
    game_state = sj.new(players=len(players))
    game_state = sj.start_round(game_state)
    while not sj.get_game_over(game_state):
        player = players[sj.get_player(game_state)]
        action = player.get_action(game_state)
        greedy_ev_policy = np.zeros(sj.MASK_SIZE)
        greedy_ev_policy[action] = 1
        states_data.append((game_state, greedy_ev_policy))
        game_state = sj.apply_action(game_state, action)
    outcome_state_value = skynet.skyjo_to_state_value(game_state)
    fixed_perspective_score = sj.get_fixed_perspective_round_scores(game_state)
    final_data = [
        (
            data[0][0],  # game
            data[0][1][: len(players)],  # table
            data[1],  # policy target
            np.roll(outcome_state_value, sj.get_player(data[0])),  # outcome target
            np.roll(
                fixed_perspective_score,
                sj.get_player(data[0]),
            ),  # points target
        )
        for data in states_data
    ]
    for data in states_data:
        print(sj.visualize_state(data[0]))
        print(np.roll(outcome_state_value, sj.get_player(data[0])))
        print(np.roll(fixed_perspective_score, sj.get_player(data[0])))
        print(sj.get_action_name(np.argwhere(data[1] == 1).item()))
    print(sj.visualize_state(game_state))

    return final_data


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


def multiprocessed_model_faceoff(
    processes: int,
    model1: skynet.SkyNet,
    model2: skynet.SkyNet,
    rounds: int = 1,
    mcts_iterations: int = 10,
    temperature: float = 0.0,
    afterstate_initial_realizations: int = 10,
):
    with mp.Pool(processes=processes) as pool:
        res_list = pool.starmap(
            model_faceoff,
            [
                (
                    model1,
                    model2,
                    1,
                    mcts_iterations,
                    temperature,
                    afterstate_initial_realizations,
                )
                for _ in range(rounds)
            ],
        )
    return sum([res[0] for res in res_list]), sum([res[1] for res in res_list])


def multiprocessed_learn(
    learn_iterations: int = 100,
    players: int = 2,
    model_device: torch.device = torch.device("cpu"),
    model_kwargs: dict[str, typing.Any] = {},
    selfplay_episodes: int = 256,
    selfplay_kwargs: dict[str, typing.Any] = {},
    training_epochs: int = 10,
    mcts_iterations: int = 500,
    mcts_temperature: float = 0.0,
    afterstate_initial_realizations: int = 50,
    validation_func: typing.Callable[[skynet.SkyNet], None] = None,
    evaluation_faceoff_rounds: int = 50,
    processes: int = 2,
    max_training_data_size: int = 1e7,
) -> None:
    """Trains a skyjo model using monte carlo tree search, self-play, and model face offs.

        Trains a skyjo model using an approach adapted from AlphaZero/MuZero.
    First training data is generated through self-play then, the model is trained
    using stochastic gradient descent. After each training epoch, the model is
    evaluated against the previous best model through model face offs and the better model is kept.

    Args:
        players: number of players in the game
        model_device: torch device to run the model on
        model_kwargs: kwargs for the model (see skynet.SkyNet1D)
        learn_iterations: number of times to train the model
        selfplay_episodes: number of self-play episodes to run
        selfplay_kwargs: kwargs for the selfplay game
        training_epochs: number of training epochs to run
        mcts_iterations: number of MCTS iterations to run
        afterstate_initial_realizations: number of afterstate initial realizations to run
        validation_func: optional function to validate the model after each training epoch
        evaluation_faceoff_rounds: number of face-offs to run to evaluate the model
        processes: number of processes to run the multiprocessed selfplay and model faceoff
    """

    def temp_func(learn_iter: int) -> float:
        if learn_iter < 50:
            return 1.0
        elif learn_iter < 100:
            return 0.5
        elif learn_iter < 150:
            return 0.25
        elif learn_iter < 200:
            return 0.1
        else:
            return mcts_temperature

    # initialize model
    initial_model = skynet.SkyNet1D(
        spatial_input_shape=(players, sj.ROW_COUNT, sj.COLUMN_COUNT, sj.FINGER_SIZE),
        non_spatial_input_shape=(sj.GAME_SIZE,),
        value_output_shape=(players,),
        policy_output_shape=(sj.MASK_SIZE,),
        **model_kwargs,
    )
    initial_model.set_device(model_device)

    models_dir = pathlib.Path("models")
    models_dir.mkdir(exist_ok=True, parents=True)
    previous_best_model_path = initial_model.save(models_dir)
    model = initial_model
    training_data = []
    # Compute validation statistics if provided
    if validation_func is not None:
        logging.info("Getting model validation stats")
        validation_func(model)
    for learn_iter in range(learn_iterations):
        logging.info("Generating training data")

        new_training_data = multiprocessed_selfplay(
            players=players,
            episodes=selfplay_episodes,
            processes=processes,
            model=model,
            mcts_iterations=mcts_iterations,
            afterstate_initial_realizations=afterstate_initial_realizations,
            mcts_temperature=temp_func(learn_iter),
            **selfplay_kwargs,
        )
        if len(training_data) > max_training_data_size:
            training_data = training_data[len(new_training_data) :]
        training_data.extend(new_training_data)

        logging.info(f"{len(training_data)} training data points")
        logging.info(
            f"value avg: {np.mean([data[3] for data in training_data], axis=0)}"
        )

        logging.info("Training model")
        train(model, training_epochs, training_data)
        trained_model_path = model.save(models_dir)

        # load old model and compare performance
        previous_best_model = skynet.SkyNet1D(
            model.spatial_input_shape,
            model.non_spatial_input_shape,
            policy_output_shape=model.policy_output_shape,
            value_output_shape=model.value_output_shape,
            dropout_rate=model.dropout_rate,
        )
        logging.info(f"Loading previous model from {previous_best_model_path}")
        previous_best_model.load_state_dict(
            torch.load(previous_best_model_path, weights_only=True)
        )
        previous_best_model.set_device(model_device)

        # Compute validation statistics if provided
        if validation_func is not None:
            logging.info("Getting model validation stats")
            validation_func(model)

        logging.info("Comparing performance")
        old_model_wins, new_model_wins = multiprocessed_model_faceoff(
            processes,
            previous_best_model,
            model,
            evaluation_faceoff_rounds,
            mcts_iterations,
            temperature=0.0,
            afterstate_initial_realizations=afterstate_initial_realizations,
        )
        logging.info(
            f"Previous best model wins: {old_model_wins} New model wins: {new_model_wins}"
        )
        if new_model_wins / (old_model_wins + new_model_wins) > 0.51:
            logging.info(f"New model is better: {trained_model_path}")
            previous_best_model_path = trained_model_path
        else:
            logging.info(f"Previous best model is better: {previous_best_model_path}")
            model = previous_best_model
    logging.info(f"Saving final model to {models_dir}")
    model.save(models_dir)


if __name__ == "__main__":
    import pathlib

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename="logs/train.log",
        filemode="a",
    )
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)
    # set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # hyperparameters
    learn_iterations = 1000
    selfplay_episodes = 256
    evaluation_faceoff_rounds = 32
    training_epochs = 1
    mcts_iterations = 400
    afterstate_initial_realizations = 100
    processes = 8
    with open("./data/validation/greedy_ev_validation_games_data.pkl", "rb") as f:
        validation_games_data = pkl.load(f)
    multiprocessed_learn(
        players=2,
        learn_iterations=learn_iterations,
        selfplay_episodes=selfplay_episodes,
        evaluation_faceoff_rounds=evaluation_faceoff_rounds,
        training_epochs=training_epochs,
        mcts_iterations=mcts_iterations,
        afterstate_initial_realizations=afterstate_initial_realizations,
        processes=processes,
        validation_func=lambda model: explain.validate_model(
            model, validation_games_data
        ),
    )

    # model = skynet.SkyNet1D(
    #     spatial_input_shape=(2, sj.ROW_COUNT, sj.COLUMN_COUNT, sj.FINGER_SIZE),
    #     non_spatial_input_shape=(sj.GAME_SIZE,),
    #     value_output_shape=(2,),
    #     policy_output_shape=(sj.MASK_SIZE,),
    # )
    # training_data = selfplay_game(
    #     model,
    #     2,
    #     mcts_iterations=200,
    #     temperature=0.0,
    #     afterstate_initial_realizations=50,
    # )
    # DataBatch.from_raw(training_data, device=device)
    # DataBatch.from_raw(training_data, device=device)
