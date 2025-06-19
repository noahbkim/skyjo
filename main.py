import logging
import pathlib
import random

import numpy as np
import torch

import buffer
import explain
import model_factory
import parallel_mcts
import play
import predictor
import skynet
import train
import train_utils

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
    model = skynet.EquivariantSkyNet(
        spatial_input_shape=(2, sj.ROW_COUNT, sj.COLUMN_COUNT, sj.FINGER_SIZE),
        non_spatial_input_shape=(sj.GAME_SIZE,),
        value_output_shape=(2,),
        policy_output_shape=(sj.MASK_SIZE,),
        device=device,
        # card_embedding_dimensions=8,
        # column_embedding_dimensions=16,
        # board_embedding_dimensions=32,
        # non_spatial_embedding_dimensions=16,
        embedding_dimensions=16,
        global_state_embedding_dimensions=32,
        num_heads=2,
    )
    # model.load_state_dict(
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

    factory = model_factory.SkyNetModelFactory(
        model_callable=skynet.EquivariantSkyNet,
        players=2,
        model_kwargs={
            # "card_embedding_dimensions": 8,
            # "column_embedding_dimensions": 16,
            # "board_embedding_dimensions": 32,
            # "non_spatial_embedding_dimensions": 16,
            "embedding_dimensions": 16,
            "num_heads": 2,
            "global_state_embedding_dimensions": 32,
        },
        device=device,
        models_dir=models_dir,
        initial_model=model,
    )
    training_config = train.TrainingEpochConfig(
        training_batch_size=256,
        learning_rate=1e-3,
        loss_function=train_utils.base_policy_value_loss,
    )
    learn_config = train.MultiProcessedLearnConfig(
        iterations=1000,
        training_epochs=2,
        training_epoch_config=training_config,
        validation_function=lambda model: explain.validate_model(
            model, validation_batch
        ),
        validation_interval=1,
        update_model_interval=1,
        selfplay_processes=9,
        minimum_games_per_iteration=100,
        torch_device=device,
    )

    predictor_config = predictor.Config(
        max_batch_size=512,
        min_batch_size=4,
        torch_device=device,
        max_wait_seconds=0.1,
    )
    mcts_config = parallel_mcts.Config(
        iterations=100,
        after_state_evaluate_all_children=False,
        virtual_loss=0.5,
        batched_leaf_count=4,
        terminal_state_rollouts=100,
        dirichlet_epsilon=0.25,
    )
    selfplay_config = play.Config(
        players=2,
        action_softmax_temperature=0.5,
        outcome_rollouts=100,
        mcts_config=mcts_config,
        start_position=None,
    )
    training_data_buffer_config = buffer.Config(
        max_size=100_000,
        spatial_input_shape=(
            selfplay_config.players,
            sj.ROW_COUNT,
            sj.COLUMN_COUNT,
            sj.FINGER_SIZE,
        ),
        non_spatial_input_shape=(sj.GAME_SIZE,),
        action_mask_shape=(sj.MASK_SIZE,),
        policy_target_shape=(sj.MASK_SIZE,),
        outcome_target_shape=(selfplay_config.players,),
        points_target_shape=(selfplay_config.players,),
    )
    train.multiprocessed_learn(
        factory,
        learn_config,
        training_config,
        predictor_config,
        training_data_buffer_config,
        selfplay_config,
        debug,
    )
