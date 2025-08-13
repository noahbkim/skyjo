import logging
import pathlib
import random
import typing

import numpy as np
import torch

import buffer
import explain
import faceoff
import factory
import mcts
import parallel_mcts
import player
import predictor
import skyjo as sj
import skynet
import train
import train_utils


def create_obvious_clear_or_almost_clear_position() -> sj.Skyjo:
    if np.random.random() < 0.5:
        return explain.create_obvious_clear_position()
    else:
        return explain.create_almost_clear_position()


def create_random_clear_or_almost_clear_position() -> sj.Skyjo:
    r = np.random.random()
    if r < 0.1:
        return explain.create_random_clear_starting_position()
    elif r < 0.2:
        return explain.create_random_almost_clear_position()
    else:
        return None


def create_random_potential_clear_position() -> sj.Skyjo:
    return explain.create_potential_clear_equal_position(
        np.random.randint(0, sj.CARD_SIZE)
    )


def model_faceoff_threshold(
    model: skynet.SkyNet,
    previous_model: skynet.SkyNet,
    policy_rounds: int,
    value_rounds: int,
    temperature: float,
    terminal_state_rollouts: int,
    win_percentage_threshold: float,
    start_state_generator: typing.Callable[[], sj.Skyjo] | None = None,
):
    policy_faceoff_result = faceoff.model_policy_faceoff(
        model,
        previous_model,
        policy_rounds,
        temperature,
        start_state_generator,
    )
    value_faceoff_result = faceoff.model_value_faceoff(
        model,
        previous_model,
        value_rounds,
        terminal_state_rollouts,
        start_state_generator,
    )
    return (
        policy_faceoff_result[0] / (policy_faceoff_result[0] + policy_faceoff_result[1])
        > win_percentage_threshold
    ) or (
        value_faceoff_result[0] / (value_faceoff_result[0] + value_faceoff_result[1])
        > win_percentage_threshold
    )


if __name__ == "__main__":
    import datetime
    import pickle as pkl

    import skyjo as sj

    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)
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

    with open("./data/validation/greedy_ev_validation_batch.pkl", "rb") as f:
        validation_batch = pkl.load(f)
    models_dir = (
        pathlib.Path("./models")
        / "distributed"
        / f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    device = torch.device("cpu")
    players = 2
    model = skynet.EquivariantSkyNet(
        spatial_input_shape=(players, sj.ROW_COUNT, sj.COLUMN_COUNT, sj.FINGER_SIZE),
        non_spatial_input_shape=(sj.GAME_SIZE,),
        value_output_shape=(players,),
        policy_output_shape=(sj.MASK_SIZE,),
        device=device,
        # card_embedding_dimensions=8,
        # column_embedding_dimensions=16,
        # board_embedding_dimensions=32,
        # non_spatial_embedding_dimensions=16,
        embedding_dimensions=32,
        global_state_embedding_dimensions=64,
        num_heads=1,
    )
    model.load_state_dict(
        torch.load(
            # "./models/distributed/20250724_102027/model_20250730_055847.pth",
            # "./models/distributed/20250730_075618/model_20250810_041449.pth",
            "./models/special/20250813/model_20250813_185028.pth",
            weights_only=True,
        )
    )
    #     torch.load(
    #         "./models/distributed/20250608_212522/model_20250609_023411.pth",
    #         weights_only=True,
    #     )
    # )
    # model = skynet.SimpleSkyNet(
    #     [256, 256, 256],
    #     spatial_input_shape=(2, sj.ROW_COUNT, sj.COLUMN_COUNT, sj.FINGER_SIZE),
    #     non_spatial_input_shape=(sj.GAME_SIZE,),
    #     value_output_shape=(2,),
    #     policy_output_shape=(sj.MASK_SIZE,),
    # )

    # main_train_on_greedy_ev_player_games(model, models_dir, validation_games_data)
    # main_overfit_small_training_sample(model, models_dir, small_fixed_training_sample)

    model_factory = factory.SkyNetModelFactory(
        model_callable=skynet.EquivariantSkyNet,
        players=players,
        model_kwargs={
            # "card_embedding_dimensions": 8,
            # "column_embedding_dimensions": 16,
            # "board_embedding_dimensions": 32,
            # "non_spatial_embedding_dimensions": 16,
            "embedding_dimensions": 32,
            "num_heads": 2,
            "global_state_embedding_dimensions": 64,
        },
        device=device,
        models_dir=models_dir,
        initial_model=model,
    )
    training_config = train.TrainConfig(
        epochs=2,
        batch_size=256,
        learn_rate=1e-3,
        loss_function=lambda model_outputs, targets: train_utils.base_loss(
            model_outputs, targets, value_scale=5.0
        ),
    )
    learn_config = train.LearnConfig(
        torch_device=device,
        learn_steps=1000,
        games_generated_per_iteration=2500,
        loss_stats_function=train_utils.loss_details_summary,
        validation_interval=1,
        validation_function=lambda model: explain.validate_model(
            model, validation_batch
        ),
        update_model_interval=1,
        model_faceoff_function=lambda model, previous_model: model_faceoff_threshold(
            model,
            previous_model,
            500,
            25,
            1.0,
            10,
            0.50,
            # create_random_potential_clear_position,
        ),
        # model_faceoff_function=lambda model, previous_model: True,
        **training_config.kwargs("training"),
    )

    predictor_config = predictor.PredictorProcessConfig(
        min_batch_size=1,
        max_batch_size=512,
        max_wait_seconds=0.1,
        torch_device=device,
    )
    # mcts_config = parallel_mcts.Config(
    #     iterations=100,
    #     after_state_evaluate_all_children=False,
    #     virtual_loss=0.5,
    #     batched_leaf_count=4,
    #     terminal_state_rollouts=25,
    #     dirichlet_epsilon=0.25,
    # )
    mcts_config = mcts.MCTSConfig(
        iterations=400,
        after_state_evaluate_all_children=False,
        terminal_state_initial_rollouts=10,
        dirichlet_epsilon=0.25,
        forced_playout_k=None,
    )
    model_player_config = player.ModelPlayerConfig(
        action_softmax_temperature=1.0,
        **mcts_config.kwargs("mcts"),
    )
    batched_mcts_config = parallel_mcts.BatchedMCTSConfig(
        iterations=400,
        after_state_evaluate_all_children=False,
        terminal_state_initial_rollouts=10,
        dirichlet_epsilon=0.25,
        batched_leaf_count=2,
        virtual_loss=0.5,
        forced_playout_k=None,
    )
    batched_model_player_config = player.BatchedModelPlayerConfig(
        action_softmax_temperature=1.0,
        **batched_mcts_config.kwargs("mcts"),
    )
    training_data_buffer_config = buffer.Config(
        max_size=5_000_000,
        spatial_input_shape=(
            players,
            sj.ROW_COUNT,
            sj.COLUMN_COUNT,
            sj.FINGER_SIZE,
        ),
        non_spatial_input_shape=(sj.GAME_SIZE,),
        action_mask_shape=(sj.MASK_SIZE,),
        policy_target_shape=(sj.MASK_SIZE,),
        outcome_target_shape=(players,),
        points_target_shape=(players,),
        cleared_columns_target_shape=(players * sj.COLUMN_COUNT,),
        path=pathlib.Path(
            f"./data/training_data/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}/buffer.pkl"
        ),
    )
    # train.run_multiprocessed_selfplay_with_local_predictor_learning(
    #     process_count=8,
    #     players=players,
    #     model_factory=model_factory,
    #     learn_config=learn_config,
    #     training_config=training_config,
    #     # predictor_config=predictor_config,
    #     training_data_buffer_config=training_data_buffer_config,
    #     model_player_config=model_player_config,
    #     # start_state_generator=create_random_potential_clear_position,
    #     outcome_rollouts=100,
    #     debug=debug,
    #     log_level=logging.DEBUG if debug else logging.INFO,
    #     log_dir=log_dir,
    # )
    train.run_multiprocessed_batched_mcts_selfplay_with_local_predictor_learning(
        process_count=9,
        players=players,
        model_factory=model_factory,
        learn_config=learn_config,
        training_config=training_config,
        training_data_buffer_config=training_data_buffer_config,
        batched_model_player_config=batched_model_player_config,
        outcome_rollouts=100,
        debug=debug,
        log_level=logging.DEBUG if debug else logging.INFO,
        log_dir=log_dir,
        # load_training_data_buffer_path=pathlib.Path(
        #     "./data/training_data/20250730_075618/buffer.pkl"
        # ),
    )
    # train.run_multiprocessed_batched_mcts_selfplay_with_dedicated_predictor_learning(
    #     process_count=9,
    #     players=players,
    #     model_factory=model_factory,
    #     learn_config=learn_config,
    #     training_config=training_config,
    #     predictor_config=predictor_config,
    #     training_data_buffer_config=training_data_buffer_config,
    #     batched_model_player_config=batched_model_player_config,
    #     start_state_generator=create_random_potential_clear_position,
    #     outcome_rollouts=100,
    #     debug=debug,
    #     log_level=logging.DEBUG if debug else logging.INFO,
    #     log_dir=log_dir,
    # )
    # train.learn(
    #     model,
    #     models_dir,
    #     learn_config,
    #     training_config,
    #     selfplay_config,
    #     training_data_buffer_config,
    #     None,
    #     debug,
    # )
    # )
