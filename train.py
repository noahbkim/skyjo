import logging
import pathlib
import random
import typing

import numpy as np
import torch
import torch.multiprocessing as mp

import buffer
import model_factory
import predictor
import selfplay
import skynet


def constant_basic_learning_rate(train_iter: int) -> float:
    return 1e-5


def constant_basic_selfplay_params(learn_iter: int) -> dict[str, typing.Any]:
    return {
        "players": 2,
        "mcts_iterations": 400,
        "mcts_temperature": 0.5,
        "afterstate_initial_realizations": 100,
        "virtual_loss": 0.5,
        "max_parallel_evaluations": 100,
    }


def train(
    model: skynet.SkyNet,
    batch: skynet.TrainingBatch,
    loss_function: typing.Callable[
        [
            skynet.SkyNetOutput,
            np.ndarray[tuple[int, int], np.float32],
            np.ndarray[tuple[int, int], np.float32],
            np.ndarray[tuple[int, int], np.float32],
        ],
        torch.Tensor,
    ],
    learning_rate: float = 1e-4,
) -> None:
    """Performs a single training step on the model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    (
        spatial_inputs,
        non_spatial_inputs,
        policy_targets,
        outcome_targets,
        points_targets,
    ) = batch
    spatial_inputs = torch.tensor(
        spatial_inputs,
        dtype=torch.float32,
        device=model.device,
    )
    non_spatial_inputs = torch.tensor(
        non_spatial_inputs,
        dtype=torch.float32,
        device=model.device,
    )
    model_output = model(spatial_inputs, non_spatial_inputs)

    loss = loss_function(model_output, outcome_targets, points_targets, policy_targets)
    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def multiprocessed_learn(
    factory: model_factory.SkyNetModelFactory,
    training_steps: int = 100,
    training_data_buffer_max_games: int = 10000,
    predictor_batch_size: int = 2048,
    selfplay_processes: int = 2,
    training_batch_size: int = 128,
    validation_function: typing.Callable[[skynet.SkyNet], None] = None,
    validation_step_interval: int = 10,
    update_best_model_step_interval: int = 2000,
    loss_function: typing.Callable[
        [skynet.SkyNetPrediction, skynet.TrainingBatch], torch.Tensor
    ] = skynet.base_total_loss,
    device: torch.device = torch.device("cpu"),
    debug: bool = False,
):
    try:
        predictor_model_update_queue = mp.Queue()
        predictor_input_queues = {
            i: predictor.PredictorInputQueue(
                queue_id=i,
                max_batch_size=512,
            )
            for i in range(selfplay_processes)
        }
        predictor_output_queues = {
            i: predictor.PredictorOutputQueue(
                queue_id=i,
                max_batch_size=512,
            )
            for i in range(selfplay_processes)
        }
        predictor_process = predictor.PredictorProcess(
            factory,
            predictor_model_update_queue,
            predictor_input_queues,
            predictor_output_queues,
            predictor_batch_size,
            device=device,
            debug=debug,
        )
        predictor_process.start()
        predictor_clients = {
            i: predictor.MultiProcessPredictorClient(
                predictor_input_queues[i], predictor_output_queues[i]
            )
            for i in range(selfplay_processes)
        }
        selfplay_data_queue = mp.Queue()
        logging.info(f"Starting {selfplay_processes} selfplay processes")
        selfplay_actors = [
            selfplay.MultiProcessedSelfplayGenerator(
                predictor_clients[i],
                selfplay_data_queue,
                id=i,
                debug=debug,
                **constant_basic_selfplay_params(0),
            )
            for i in range(selfplay_processes)
        ]
        for actor in selfplay_actors:
            actor.start()

        training_data_buffer = buffer.ReplayBuffer(
            max_games=training_data_buffer_max_games
        )
        logging.info(f"Training for {training_steps} steps")
        model = factory.get_latest_model()
        model.set_device(device)
        for train_iter in range(training_steps):
            # Add training data from the queue into the buffer
            while (
                not selfplay_data_queue.empty() or training_data_buffer.games_count < 2
            ):
                game_data = selfplay_data_queue.get()
                training_data_buffer.add_game_data_with_symmetry(game_data)
            if train_iter == 0:
                logging.info("Enough selfplay data collected, starting training")
            batch = training_data_buffer.sample_batch(batch_size=training_batch_size)

            train(
                model,
                batch,
                loss_function=loss_function,
                learning_rate=constant_basic_learning_rate(train_iter),
            )

            if (
                train_iter % validation_step_interval == 0
                and validation_function is not None
            ):
                validation_function(model)

            if train_iter > 0 and train_iter % update_best_model_step_interval == 0:
                logging.info(
                    f"Training data buffer games: {training_data_buffer.games_count}, total buffer size: {len(training_data_buffer)}"
                )
                saved_path = factory.save_model(model)
                logging.info(f"Saved model to {saved_path}")
                predictor_model_update_queue.put(f"new_model {saved_path}")
    finally:
        predictor_process.terminate()
        for actor in selfplay_actors:
            actor.terminate()
        for actor in selfplay_actors:
            actor.join()
        predictor_process.join()


if __name__ == "__main__":
    import datetime
    import pathlib
    import pickle as pkl

    import explain

    debug = False
    log_dir = pathlib.Path(
        f"logs/multiprocessed_train/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}/"
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=log_dir / "main.log",
        filemode="w",
    )
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)
    players = 2
    selfplay_processes = 6
    device = torch.device("mps")
    models_dir = (
        pathlib.Path("./models")
        / "distributed"
        / f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    models_dir.mkdir(parents=True, exist_ok=True)
    factory = model_factory.SkyNetModelFactory(
        players=players,
        dropout_rate=0.5,
        device=device,
        models_dir=models_dir,
        model_callable=skynet.SkyNet2D,
    )
    with open("./data/validation/greedy_ev_validation_games_data.pkl", "rb") as f:
        validation_games_data = pkl.load(f)
    multiprocessed_learn(
        factory,
        selfplay_processes=selfplay_processes,
        training_steps=int(1e6),
        device=device,
        debug=debug,
        training_batch_size=1024,
        predictor_batch_size=1028,
        validation_function=lambda model: explain.validate_model(
            model, validation_games_data
        ),
        validation_step_interval=10000,
        update_best_model_step_interval=10000,
    )
    # batch_size = 32
    # model_runner_model_update_queue = mp.Queue()
    # model_runner_input_queues = {i: mp.Queue() for i in range(selfplay_processes)}
    # model_runner_output_queues = {i: mp.Queue() for i in range(selfplay_processes)}
    # predictor_process = predictor.PredictorProcess(
    #     model_factory,
    #     model_runner_model_update_queue,
    #     model_runner_input_queues,
    #     model_runner_output_queues,
    #     batch_size,
    #     device=device,
    #     max_wait_seconds=0.2,
    #     debug=True,
    # )
    # try:
    #     predictor_process.start()
    #     predictor = predictor.MultiProcessPredictorClient(
    #         model_runner_input_queues[0], model_runner_output_queues[0]
    #     )
    #     game_state = sj.new(players=players)
    #     game_state = sj.start_round(game_state)
    #     # game_state = explain.create_almost_surely_winning_position()
    #     logging.debug(sj.visualize_state(game_state))
    #     start_time = time.time()
    #     while not sj.get_game_over(game_state):
    #         root_node = parallel_mcts.run_mcts(
    #             game_state,
    #             predictor,
    #             1600,
    #             afterstate_initial_realizations=100,
    #             max_parallel_evaluations=32,
    #         )
    #         logging.debug(f"Visit count: {root_node.visit_count}")
    #         logging.debug(f"State value: {root_node.state_value}")
    #         logging.debug(
    #             f"MCTS Visit Probabilities: {root_node.sample_child_visit_probabilities(temperature=1.0)}"
    #         )
    #         logging.debug(
    #             f"MCTS Visit Probabilities (temp=0.5): {root_node.sample_child_visit_probabilities(temperature=0.5)}"
    #         )
    #         mcts_probs = root_node.sample_child_visit_probabilities(temperature=1.0)
    #         action = np.random.choice(sj.MASK_SIZE, p=mcts_probs)
    #         assert sj.actions(game_state)[action]
    #         game_state = sj.apply_action(game_state, action)
    #         logging.debug(f"ACTION: {sj.get_action_name(action)}")
    #         logging.debug(sj.visualize_state(game_state))
    #     winner = sj.get_fixed_perspective_winner(game_state)
    #     logging.debug(f"Winner: {winner}")
    #     logging.debug(f"Scores: {sj.get_fixed_perspective_round_scores(game_state)}")
    #     logging.debug(f"Time: {time.time() - start_time}")
    # finally:
    #     logging.info("Terminating predictor")
    #     predictor_process.terminate()
    #     predictor_process.join()
