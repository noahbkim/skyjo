"""Small matched offline ablations; self-play histories are shared across arms."""

from __future__ import annotations

import dataclasses
import logging
import random
import time

import numpy as np
import torch

from . import buffer, game as sj, play, player, predictor, skynet, targets, train, train_utils


PRESETS = {
    "baseline": {},
    "raw_score": {"round_raw_score": 0.1},
    "doubling": {"round_doubled": 0.1},
    "both": {"round_raw_score": 0.1, "round_doubled": 0.1},
}


@dataclasses.dataclass(frozen=True)
class AblationConfig:
    seeds: tuple[int, ...] = (0, 1)
    games: int = 8
    training_steps: int = 32
    batch_size: int = 32
    evaluation_pairs: int = 4
    terminal_rollouts: int = 32
    mcts_iterations: int = 16


def make_model(seed, auxiliary_objectives):
    torch.manual_seed(seed)
    return skynet.EquivariantSkyNet(
        spatial_input_shape=(2, sj.ROW_COUNT, sj.COLUMN_COUNT, sj.FINGER_SIZE),
        non_spatial_input_shape=(sj.GAME_SIZE,), value_output_shape=(2,),
        policy_output_shape=(sj.MASK_SIZE,), device=torch.device("cpu"),
        embedding_dimensions=8, global_state_embedding_dimensions=16, num_heads=2,
        auxiliary_objectives=auxiliary_objectives,
    )


def evaluate_policy(model, pairs, seed):
    """Score policy-only play against GreedyExpectedValuePlayer, in both seats."""
    margins, wins = [], []
    for pair in range(pairs):
        for seat in range(2):
            random.seed(seed + pair)
            np.random.seed(seed + pair)
            players = [player.GreedyExpectedValuePlayer(), player.GreedyExpectedValuePlayer()]
            players[seat] = player.PureModelPolicyPlayer(model, temperature=0)
            history = play.play(players)
            scores = sj.get_fixed_perspective_round_scores(history[-1].state)
            margin = int(scores[1 - seat]) - int(scores[seat])
            margins.append(margin)
            wins.append(float(margin > 0) + 0.5 * float(margin == 0))
    return {"win_fraction_ties_half": float(np.mean(wins)),
            "mean_opponent_minus_model_score": float(np.mean(margins)),
            "evaluation_games": len(wins)}


def run_ablation(config: AblationConfig) -> dict:
    if not config.seeds or min(config.games, config.training_steps, config.batch_size,
                               config.evaluation_pairs, config.terminal_rollouts,
                               config.mcts_iterations) < 1:
        raise ValueError("Ablation seeds and all budgets must be nonempty/positive")
    results = []
    for seed in config.seeds:
        random.seed(seed)
        np.random.seed(seed)
        initial = make_model(seed, {})
        initial.eval()
        client = predictor.LocalPredictorClient(initial, max_batch_size=512)
        search_player = player.ModelPlayer(
            client, action_softmax_temperature=1.0,
            mcts_iterations=config.mcts_iterations, mcts_dirichlet_epsilon=0.25,
            mcts_after_state_evaluate_all_children=False,
            mcts_terminal_state_initial_rollouts=1, mcts_forced_playout_k=None,
        )
        histories = [play.model_player_selfplay([search_player, search_player])
                     for _ in range(config.games)]
        initial_metrics = evaluate_policy(initial, config.evaluation_pairs, seed + 10000)
        for name, weights in PRESETS.items():
            logging.info("Ablation seed=%s arm=%s", seed, name)
            model = make_model(seed, weights)
            target_rng = random.Random(seed)
            start = time.perf_counter()
            rows = []
            for history in histories:
                data, _ = targets.build_targets(
                    history, model.objectives,
                    terminal_rollouts=config.terminal_rollouts, rng=target_rng,
                )
                rows.extend(data)
            target_seconds = time.perf_counter() - start
            replay = buffer.ReplayBuffer.from_config(buffer.for_objectives(buffer.Config(
                max_size=len(rows), spatial_input_shape=model.spatial_input_shape,
                non_spatial_input_shape=model.non_spatial_input_shape,
                action_mask_shape=model.policy_output_shape,
            ), model.objectives))
            replay.add_game_data(rows)
            optimizer = train.make_optimizer(model, 1e-3)
            # Reset sampling for exactly matching replay row indices in every arm.
            np.random.seed(seed)
            start = time.perf_counter()
            losses = []
            for _ in range(config.training_steps):
                _, details = train.train_step(
                    model, replay.sample_batch(min(config.batch_size, len(replay))),
                    train_utils.base_loss, optimizer,
                )
                losses.append(details)
            training_seconds = time.perf_counter() - start
            result = {
                "seed": seed, "arm": name, "auxiliary_objectives": weights,
                "training_rows": len(rows), "target_seconds": target_seconds,
                "training_seconds": training_seconds,
                "initial_policy": initial_metrics,
                "trained_policy": evaluate_policy(model, config.evaluation_pairs, seed + 10000),
                "mean_training_losses": {
                    key: float(np.mean([loss[key] for loss in losses])) for key in losses[0]
                },
            }
            results.append(result)
    return {
        "config": dataclasses.asdict(config),
        "method": "Offline comparison on shared initial-model MCTS histories; matched core initialization, terminal samples and minibatch indices. Policy-only evaluation against greedy EV in both seats. Small runs are smoke tests, not strength evidence.",
        "results": results,
    }
