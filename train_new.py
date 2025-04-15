import dataclasses
import logging
import pathlib
import random
import typing

import numpy as np
import torch

import mcts_new as mcts
import skyjo as sj
import skynet


@dataclasses.dataclass(slots=True, frozen=True)
class DataBatch:
    spatial_inputs_tensor: torch.Tensor
    non_spatial_inputs_tensor: torch.Tensor
    policy_targets_tensor: torch.Tensor
    value_targets_tensor: torch.Tensor
    point_differential_targets_tensor: torch.Tensor

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
        point_differential_targets = torch.tensor(
            np.array([data[4] for data in raw_data]),
            dtype=torch.float32,
            device=device,
        )
        return cls(
            spatial_inputs_tensor=spatial_inputs,
            non_spatial_inputs_tensor=non_spatial_inputs,
            policy_targets_tensor=policy_targets,
            value_targets_tensor=value_targets,
            point_differential_targets_tensor=point_differential_targets,
        )


def selfplay_game(
    model: skynet.SkyNet,
    num_players: int,
    mcts_iterations: int = 10,
    temperature: float = 0.0,
    num_afterstate_outcomes: int = 10,
) -> list[tuple]:
    model.eval()
    with torch.no_grad():
        game_state = sj.new(players=num_players)
        game_state = sj.start_round(game_state)
        countdown = game_state[6]
        game_data = []
        while countdown != 0:
            node = mcts.run_mcts(
                game_state,
                model,
                mcts_iterations,
                num_afterstate_outcomes=num_afterstate_outcomes,
            )
            if len(game_data) < 15:
                temperature = 1.0
            mcts_probs = node.sample_child_visit_probabilities(temperature=temperature)
            choice = np.random.choice(sj.MASK_SIZE, p=mcts_probs)
            assert sj.actions(game_state)[choice]
            game_data.append((game_state, mcts_probs))
            game_state = sj.apply_action(game_state, choice)
            countdown = game_state[6]
        outcome_state_value = skynet.skyjo_to_state_value(game_state)
        fixed_perspective_score = sj.get_fixed_perspective_round_scores(game_state)
        return [
            (
                data[0][0],  # game
                data[0][1][:num_players],  # table
                data[1],  # policy target
                np.roll(outcome_state_value, -sj.get_player(data[0])),  # outcome target
                np.roll(
                    fixed_perspective_score - fixed_perspective_score.min(),
                    sj.get_player(data[0]),
                ),  # point differential target
            )
            for data in game_data
        ]


def train(model: skynet.SkyNet, num_epochs: int, batches: list[list[tuple]]):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    policy_losses = []
    value_losses = []
    point_differential_losses = []
    for _ in range(num_epochs):
        model.train()

        for i, batch in enumerate(batches):
            (
                torch_predicted_value,
                torch_predicted_point_differential,
                torch_predicted_policy,
            ) = model(batch.spatial_inputs_tensor, batch.non_spatial_inputs_tensor)

            policy_loss = skynet.compute_policy_loss(
                torch_predicted_policy, batch.policy_targets_tensor
            )
            value_loss = skynet.compute_value_loss(
                torch_predicted_value, batch.value_targets_tensor
            )
            point_differential_loss = skynet.compute_point_differential_loss(
                torch_predicted_point_differential,
                batch.point_differential_targets_tensor,
            )
            total_loss = 10 * value_loss + point_differential_loss + policy_loss

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            point_differential_losses.append(point_differential_loss.item())

            # compute gradient and do SGD step
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if i % 100 == 0:
                logging.info(
                    f"value loss: {value_loss.item() * 10} "
                    f"point differential loss: {point_differential_loss.item()} "
                    f"policy loss: {policy_loss.item()} "
                    f"total loss: {total_loss.item()} "
                )


def model_single_game_faceoff(
    model1: skynet.SkyNet,
    model2: skynet.SkyNet,
    mcts_iterations: int = 10,
    temperature: float = 0.5,
):
    model1.eval()
    model2.eval()
    with torch.no_grad():
        game_state = sj.new(players=2)
        game_state = sj.start_round(game_state)
        assert sj.validate(game_state)
        players = [model1, model2]
        random.shuffle(players)
        countdown = game_state[6]
        while countdown != 0:
            node = mcts.run_mcts(
                game_state, players[sj.get_player(game_state)], mcts_iterations
            )
            choice = np.random.choice(
                sj.MASK_SIZE, p=node.sample_child_visit_probabilities(temperature)
            )

            assert sj.actions(game_state)[choice]
            game_state = sj.apply_action(game_state, choice)
            assert sj.validate(game_state)
            countdown = game_state[6]
    return skynet.skyjo_to_state_value(
        game_state
    ), sj.get_fixed_perspective_round_scores(game_state)


def model_faceoff(
    model1: skynet.SkyNet,
    model2: skynet.SkyNet,
    num_rounds: int = 1,
    mcts_iterations: int = 10,
    temperature: float = 0.0,
):
    model1_wins, model2_wins = 0, 0
    model1_point_differential, model2_point_differential = 0, 0
    for _ in range(num_rounds):
        outcome, round_scores = model_single_game_faceoff(
            model1, model2, mcts_iterations, temperature
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
            model2, model1, mcts_iterations, temperature
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
        f"Model 1 avg point differential: {model1_point_differential / (2 * num_rounds)} Model 2 avg point differential: {model2_point_differential / (2 * num_rounds)}"
    )
    return model1_wins, model2_wins


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename="logs/train.log",
        filemode="a",
    )
    np.random.seed(0)
    torch.manual_seed(0)
    # set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # hyperparameters
    train_steps = 100
    num_episodes = 250
    num_faceoff_rounds = 25
    num_epochs = 10
    mcts_iterations = 50
    # initialize model
    initial_model = skynet.SkyNet1D(
        spatial_input_shape=(2, sj.ROW_COUNT, sj.COLUMN_COUNT, sj.FINGER_SIZE),
        non_spatial_input_shape=(sj.GAME_SIZE,),
        value_output_shape=(2,),
        policy_output_shape=(sj.MASK_SIZE,),
        dropout_rate=0.3,
    )
    initial_model.set_device(device)

    models_dir = pathlib.Path("models")
    old_saved_model_path = initial_model.save(models_dir)
    model = initial_model
    for train_step in range(train_steps):
        # save initial model for comparison
        models_dir.mkdir(exist_ok=True, parents=True)

        logging.info("Generating training data")
        training_data = []
        for i in range(num_episodes):
            training_data.extend(
                selfplay_game(
                    model,
                    2,
                    mcts_iterations=mcts_iterations,
                    temperature=0.5,
                )
            )

        # Create training data batches
        random.shuffle(training_data)
        batches = [
            DataBatch.from_raw(
                [
                    training_data[data_idx]
                    for data_idx in np.random.randint(len(training_data), size=64)
                ],
                device=device,
            )
            for _ in range(len(training_data) // 64)
        ]
        logging.info(f"{len(training_data)} training data points")
        # train model
        logging.info("Training model")
        train(model, num_epochs, batches)
        trained_model_path = model.save(models_dir)

        # load old model and compare performance
        old_model = skynet.SkyNet1D(
            model.spatial_input_shape,
            model.non_spatial_input_shape,
            policy_output_shape=model.policy_output_shape,
            value_output_shape=model.value_output_shape,
            dropout_rate=0.3,
        )
        old_model.load_state_dict(torch.load(old_saved_model_path, weights_only=True))
        old_model.set_device(device)

        # compare performance
        logging.info("Comparing performance")
        old_model_wins, new_model_wins = model_faceoff(
            old_model, model, num_faceoff_rounds, mcts_iterations, temperature=0.5
        )
        logging.info(
            f"Old model wins: {old_model_wins} New model wins: {new_model_wins}"
        )
        if new_model_wins / (old_model_wins + new_model_wins) > 0.51:
            logging.info("New model is better")
            old_saved_model_path = trained_model_path
        else:
            logging.info("Old model is better")
            model = old_model

    model.save(models_dir)
