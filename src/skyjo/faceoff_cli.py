"""
Status: manual
Purpose: Run a fair, paired-seat faceoff between two Skyjo model checkpoints.
Promote when: Checkpoint faceoffs become part of a scheduled evaluation workflow.
"""

from __future__ import annotations

import concurrent.futures
import multiprocessing
import pathlib
import typing

import torch
import typer
from tqdm.auto import tqdm

from . import checkpoint, faceoff, player, skynet
from . import game as sj

DEFAULT_GAMES = 20
DEFAULT_MCTS_ITERATIONS = 100
DEFAULT_TERMINAL_STATE_ROLLOUTS = 10
DEFAULT_EMBEDDING_DIMENSIONS = 32
DEFAULT_GLOBAL_STATE_EMBEDDING_DIMENSIONS = 64
DEFAULT_NUM_HEADS = 2

_WORKER_CANDIDATE: skynet.SkyNet | None = None
_WORKER_CHAMPION: skynet.SkyNet | None = None
_WORKER_PLAYER_CONFIG: player.ModelPlayerConfig | None = None
_WORKER_CHAMPION_PLAYER_CONFIG: player.ModelPlayerConfig | None = None


def _checkpoint_configuration(path: pathlib.Path) -> dict[str, typing.Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        return {}
    configuration = payload.get("configuration")
    return configuration if isinstance(configuration, dict) else {}


def _checkpoint_model_configuration(path: pathlib.Path) -> dict[str, typing.Any]:
    configuration = _checkpoint_configuration(path)
    model_configuration = configuration.get("model")
    return model_configuration if isinstance(model_configuration, dict) else {}


def _validate_checkpoint_architecture(
    path: pathlib.Path,
    model_configuration: dict[str, typing.Any],
) -> None:
    architecture_name = model_configuration.get("name")
    if architecture_name is None:
        return
    if architecture_name not in {
        skynet.EQUIVARIANT_ARCHITECTURE_NAME,
        skynet.EQUIVARIANT_AUX_ARCHITECTURE_NAME,
    }:
        raise ValueError(
            f"Checkpoint {path} uses unsupported model architecture "
            f"{architecture_name!r}; expected "
            "a supported equivariant architecture. "
            "Legacy EquivariantSkyNet checkpoints cannot be loaded."
        )


def _resolve_model_parameter(
    *,
    name: str,
    override: int | None,
    candidate_configuration: dict[str, typing.Any],
    champion_configuration: dict[str, typing.Any],
    default: int,
) -> int:
    if override is not None:
        return override
    candidate_value = candidate_configuration.get(name)
    champion_value = champion_configuration.get(name)
    if (
        candidate_value is not None
        and champion_value is not None
        and candidate_value != champion_value
    ):
        raise ValueError(
            f"Checkpoint model settings disagree for {name}: "
            f"{candidate_value!r} != {champion_value!r}. "
            f"Pass --{name.replace('_', '-')} explicitly to choose a value."
        )
    value = candidate_value if candidate_value is not None else champion_value
    return default if value is None else int(value)


def _build_model(
    *,
    checkpoint_path: pathlib.Path,
    device: torch.device,
    embedding_dimensions: int,
    global_state_embedding_dimensions: int,
    num_heads: int,
    architecture_name: str | None = None,
) -> skynet.EquivariantSkyNet:
    players = 2
    model_class = (
        skynet.EquivariantSkyNetWithAuxiliaryHeads
        if architecture_name == skynet.EQUIVARIANT_AUX_ARCHITECTURE_NAME
        else (
            skynet.EquivariantSkyNetWithRoundScoreAux
            if architecture_name == skynet.EQUIVARIANT_SCORE_AUX_ARCHITECTURE_NAME
            else skynet.EquivariantSkyNet
        )
    )
    model = model_class(
        spatial_input_shape=(
            players,
            sj.ROW_COUNT,
            sj.COLUMN_COUNT,
            sj.FINGER_SIZE,
        ),
        non_spatial_input_shape=skynet.get_non_spatial_input_shape(players),
        value_output_shape=(players,),
        policy_output_shape=(sj.MASK_SIZE,),
        device=device,
        embedding_dimensions=embedding_dimensions,
        global_state_embedding_dimensions=global_state_embedding_dimensions,
        num_heads=num_heads,
    )
    checkpoint.load_checkpoint(
        checkpoint_path,
        model=model,
        restore_rng=False,
        map_location=device,
    )
    model.to(device)
    model.eval()
    return model


def _model_player_config(
    *,
    mcts_iterations: int,
    terminal_state_rollouts: int,
    score_utility_weight: float = 0.0,
) -> player.ModelPlayerConfig:
    return player.ModelPlayerConfig(
        action_softmax_temperature=0.0,
        mcts_iterations=mcts_iterations,
        mcts_dirichlet_epsilon=0.0,
        mcts_after_state_evaluate_all_children=False,
        mcts_terminal_state_initial_rollouts=terminal_state_rollouts,
        mcts_score_utility_weight=score_utility_weight,
    )


def _checkpoint_score_utility_weight(path: pathlib.Path) -> float:
    configuration = _checkpoint_configuration(path)
    player_configuration = configuration.get("player", {})
    if not isinstance(player_configuration, dict):
        player_configuration = {}
    if "mcts_score_utility_weight" in player_configuration:
        return float(player_configuration["mcts_score_utility_weight"])
    search_configuration = configuration.get("search", {})
    if not isinstance(search_configuration, dict):
        return 0.0
    return float(search_configuration.get("score_utility_weight", 0.0))


def _initialize_faceoff_worker(
    candidate_checkpoint: pathlib.Path,
    champion_checkpoint: pathlib.Path,
    device_name: str,
    model_parameters: dict[str, int],
    model_player_config: player.ModelPlayerConfig,
    candidate_architecture: str | None,
    champion_architecture: str | None,
    champion_model_player_config: player.ModelPlayerConfig,
) -> None:
    global _WORKER_CANDIDATE, _WORKER_CHAMPION, _WORKER_PLAYER_CONFIG
    global _WORKER_CHAMPION_PLAYER_CONFIG
    torch.set_num_threads(1)
    device = torch.device(device_name)
    _WORKER_CANDIDATE = _build_model(
        checkpoint_path=candidate_checkpoint,
        device=device,
        architecture_name=candidate_architecture,
        **model_parameters,
    )
    _WORKER_CHAMPION = _build_model(
        checkpoint_path=champion_checkpoint,
        device=device,
        architecture_name=champion_architecture,
        **model_parameters,
    )
    _WORKER_PLAYER_CONFIG = model_player_config
    _WORKER_CHAMPION_PLAYER_CONFIG = champion_model_player_config


def _run_worker_pair(pair_seed: int) -> faceoff.MCTSFaceoffResult:
    if (
        _WORKER_CANDIDATE is None
        or _WORKER_CHAMPION is None
        or _WORKER_PLAYER_CONFIG is None
        or _WORKER_CHAMPION_PLAYER_CONFIG is None
    ):
        raise RuntimeError("faceoff worker was not initialized")
    return faceoff.model_mcts_faceoff_detailed(
        _WORKER_CANDIDATE,
        _WORKER_CHAMPION,
        model_player_config=_WORKER_PLAYER_CONFIG,
        paired_rounds=1,
        seed=pair_seed,
        champion_model_player_config=_WORKER_CHAMPION_PLAYER_CONFIG,
    )


def _run_parallel_faceoff(
    *,
    candidate_checkpoint: pathlib.Path,
    champion_checkpoint: pathlib.Path,
    paired_rounds: int,
    seed: int,
    workers: int,
    device_name: str,
    model_parameters: dict[str, int],
    model_player_config: player.ModelPlayerConfig,
    champion_model_player_config: player.ModelPlayerConfig | None = None,
    candidate_architecture: str | None = None,
    champion_architecture: str | None = None,
    game_completed_callback: typing.Callable[[], None] | None,
) -> faceoff.MCTSFaceoffResult:
    result = faceoff.MCTSFaceoffResult()
    worker_count = min(workers, paired_rounds)
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=context,
        initializer=_initialize_faceoff_worker,
        initargs=(
            candidate_checkpoint,
            champion_checkpoint,
            device_name,
            model_parameters,
            model_player_config,
            candidate_architecture,
            champion_architecture,
            champion_model_player_config or model_player_config,
        ),
    ) as executor:
        futures = [
            executor.submit(_run_worker_pair, seed + pair_index)
            for pair_index in range(paired_rounds)
        ]
        for completed in concurrent.futures.as_completed(futures):
            completed_result = completed.result()
            if isinstance(completed_result, tuple):
                completed_result = faceoff.MCTSFaceoffResult(
                    candidate_wins=completed_result[0],
                    champion_wins=completed_result[1],
                )
            result = result + completed_result
            if game_completed_callback is not None:
                game_completed_callback()
                game_completed_callback()
    return result


def run_faceoff_detailed(
    *,
    candidate_checkpoint: pathlib.Path,
    champion_checkpoint: pathlib.Path,
    games: int,
    seed: int,
    device_name: str,
    mcts_iterations: int,
    terminal_state_rollouts: int,
    embedding_dimensions: int | None,
    global_state_embedding_dimensions: int | None,
    num_heads: int | None,
    game_completed_callback: typing.Callable[[], None] | None = None,
    workers: int = 1,
) -> faceoff.MCTSFaceoffResult:
    """Load two checkpoints and return paired win, score, and clear metrics."""
    if games < 2 or games % 2:
        raise ValueError("games must be a positive even number (one game per seat)")
    if mcts_iterations < 1:
        raise ValueError("mcts_iterations must be at least one")
    if terminal_state_rollouts < 1:
        raise ValueError("terminal_state_rollouts must be at least one")
    if workers < 1:
        raise ValueError("workers must be at least one")
    device = torch.device(device_name)
    if workers > 1 and device.type != "cpu":
        raise ValueError("multiprocessing is supported only with --device cpu")

    candidate_configuration = _checkpoint_model_configuration(candidate_checkpoint)
    champion_configuration = _checkpoint_model_configuration(champion_checkpoint)
    _validate_checkpoint_architecture(candidate_checkpoint, candidate_configuration)
    _validate_checkpoint_architecture(champion_checkpoint, champion_configuration)
    model_parameters = {
        "embedding_dimensions": _resolve_model_parameter(
            name="embedding_dimensions",
            override=embedding_dimensions,
            candidate_configuration=candidate_configuration,
            champion_configuration=champion_configuration,
            default=DEFAULT_EMBEDDING_DIMENSIONS,
        ),
        "global_state_embedding_dimensions": _resolve_model_parameter(
            name="global_state_embedding_dimensions",
            override=global_state_embedding_dimensions,
            candidate_configuration=candidate_configuration,
            champion_configuration=champion_configuration,
            default=DEFAULT_GLOBAL_STATE_EMBEDDING_DIMENSIONS,
        ),
        "num_heads": _resolve_model_parameter(
            name="num_heads",
            override=num_heads,
            candidate_configuration=candidate_configuration,
            champion_configuration=champion_configuration,
            default=DEFAULT_NUM_HEADS,
        ),
    }
    candidate_architecture = candidate_configuration.get("name")
    champion_architecture = champion_configuration.get("name")
    model_player_config = _model_player_config(
        mcts_iterations=mcts_iterations,
        terminal_state_rollouts=terminal_state_rollouts,
        score_utility_weight=_checkpoint_score_utility_weight(candidate_checkpoint),
    )
    champion_model_player_config = _model_player_config(
        mcts_iterations=mcts_iterations,
        terminal_state_rollouts=terminal_state_rollouts,
        score_utility_weight=_checkpoint_score_utility_weight(champion_checkpoint),
    )
    if workers > 1:
        return _run_parallel_faceoff(
            candidate_checkpoint=candidate_checkpoint,
            champion_checkpoint=champion_checkpoint,
            paired_rounds=games // 2,
            seed=seed,
            workers=workers,
            device_name=device_name,
            model_parameters=model_parameters,
            model_player_config=model_player_config,
            champion_model_player_config=champion_model_player_config,
            candidate_architecture=candidate_architecture,
            champion_architecture=champion_architecture,
            game_completed_callback=game_completed_callback,
        )

    candidate = _build_model(
        checkpoint_path=candidate_checkpoint,
        device=device,
        architecture_name=candidate_architecture,
        **model_parameters,
    )
    champion = _build_model(
        checkpoint_path=champion_checkpoint,
        device=device,
        architecture_name=champion_architecture,
        **model_parameters,
    )
    return faceoff.model_mcts_faceoff_detailed(
        candidate,
        champion,
        model_player_config=model_player_config,
        paired_rounds=games // 2,
        seed=seed,
        game_completed_callback=game_completed_callback,
        champion_model_player_config=champion_model_player_config,
    )


def run_faceoff(
    *, detailed: bool = False, **kwargs: typing.Any
) -> tuple[int, int] | faceoff.MCTSFaceoffResult:
    """Backward-compatible win-count interface."""
    result = run_faceoff_detailed(**kwargs)
    if detailed:
        return result
    return result.candidate_wins, result.champion_wins


def faceoff_checkpoints(
    candidate_checkpoint: typing.Annotated[
        pathlib.Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Checkpoint for model A (the candidate).",
        ),
    ],
    champion_checkpoint: typing.Annotated[
        pathlib.Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Checkpoint for model B (the champion).",
        ),
    ],
    games: typing.Annotated[
        int,
        typer.Option(help="Total games; must be even so both models play each seat."),
    ] = DEFAULT_GAMES,
    seed: typing.Annotated[
        int,
        typer.Option(help="Seed for the first paired round."),
    ] = 0,
    device_name: typing.Annotated[
        str,
        typer.Option("--device", help="Torch device, such as cpu, cuda, or mps."),
    ] = "cpu",
    mcts_iterations: typing.Annotated[
        int,
        typer.Option(help="MCTS iterations per move."),
    ] = DEFAULT_MCTS_ITERATIONS,
    terminal_state_rollouts: typing.Annotated[
        int,
        typer.Option(help="Initial terminal-state rollouts in MCTS."),
    ] = DEFAULT_TERMINAL_STATE_ROLLOUTS,
    embedding_dimensions: typing.Annotated[
        int | None,
        typer.Option(help="Override checkpoint embedding_dimensions."),
    ] = None,
    global_state_embedding_dimensions: typing.Annotated[
        int | None,
        typer.Option(help="Override checkpoint global_state_embedding_dimensions."),
    ] = None,
    num_heads: typing.Annotated[
        int | None,
        typer.Option(help="Override checkpoint num_heads."),
    ] = None,
    show_progress: typing.Annotated[
        bool,
        typer.Option("--progress/--no-progress", help="Show game progress."),
    ] = True,
    workers: typing.Annotated[
        int,
        typer.Option(help="CPU worker processes; 1 runs games sequentially."),
    ] = 1,
) -> None:
    """Run a deterministic, paired-seat MCTS checkpoint faceoff."""
    try:
        with tqdm(
            total=games,
            desc="Faceoff",
            unit="game",
            disable=not show_progress,
        ) as progress_bar:
            raw_result = run_faceoff(
                candidate_checkpoint=candidate_checkpoint,
                champion_checkpoint=champion_checkpoint,
                games=games,
                seed=seed,
                device_name=device_name,
                mcts_iterations=mcts_iterations,
                terminal_state_rollouts=terminal_state_rollouts,
                embedding_dimensions=embedding_dimensions,
                global_state_embedding_dimensions=global_state_embedding_dimensions,
                num_heads=num_heads,
                game_completed_callback=progress_bar.update,
                workers=workers,
                detailed=True,
            )
    except (OSError, RuntimeError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error

    if isinstance(raw_result, tuple):
        result = faceoff.MCTSFaceoffResult(
            candidate_wins=raw_result[0],
            champion_wins=raw_result[1],
        )
    else:
        result = raw_result
    typer.echo(f"candidate: {candidate_checkpoint}")
    typer.echo(f"champion: {champion_checkpoint}")
    typer.echo(f"games: {games}")
    typer.echo(f"workers: {workers}")
    typer.echo(f"candidate_wins: {result.candidate_wins}")
    typer.echo(f"champion_wins: {result.champion_wins}")
    typer.echo(f"candidate_win_rate: {result.candidate_wins / games:.1%}")
    typer.echo(f"candidate_mean_score: {result.candidate_mean_score:.3f}")
    typer.echo(f"champion_mean_score: {result.champion_mean_score:.3f}")
    typer.echo(
        "candidate_mean_score_differential: "
        f"{result.candidate_mean_score - result.champion_mean_score:.3f}"
    )
    typer.echo(f"candidate_mean_clears: {result.candidate_mean_clears:.3f}")
    typer.echo(f"champion_mean_clears: {result.champion_mean_clears:.3f}")
    if result.candidate_wins == result.champion_wins:
        if result.candidate_mean_score == result.champion_mean_score:
            typer.echo("winner: tie")
        else:
            winner = (
                "candidate"
                if result.candidate_mean_score < result.champion_mean_score
                else "champion"
            )
            typer.echo(f"winner: {winner} (score tiebreak)")
    else:
        winner = (
            "candidate"
            if result.candidate_wins > result.champion_wins
            else "champion"
        )
        typer.echo(f"winner: {winner}")


def main() -> None:
    typer.run(faceoff_checkpoints)


if __name__ == "__main__":
    main()
