"""Model factory for loading and saving models with consistent behavior across
processes."""

import pathlib
import typing
import datetime

import torch

from . import game as sj
from . import skynet
from . import checkpoint


class SkyNetModelFactory:
    def __init__(
        self,
        model_callable: typing.Callable[[], skynet.SkyNet],
        players: int = 2,
        device: torch.device = torch.device("cpu"),
        models_dir: pathlib.Path = pathlib.Path("models"),
        model_kwargs: dict[str, typing.Any] = {},
        initial_model: skynet.SkyNet | None = None,
    ):
        self.models_dir = models_dir
        self.players = players
        self.device = device
        self.model_kwargs = model_kwargs
        self.model_callable = model_callable
        if list(self.models_dir.glob("checkpoint_*.pth")):
            return
        legacy_models = list(self.models_dir.glob("model_*.pth"))
        if legacy_models:
            raise checkpoint.CheckpointFormatError(
                f"{self.models_dir} contains raw model files but no versioned checkpoints"
            )
        if initial_model is None:
            initial_model = self.model_callable(
                spatial_input_shape=(
                    players,
                    sj.ROW_COUNT,
                    sj.COLUMN_COUNT,
                    sj.FINGER_SIZE,
                ),
                non_spatial_input_shape=(sj.GAME_SIZE,),
                value_output_shape=(players,),
                policy_output_shape=(sj.MASK_SIZE,),
                device=self.device,
                **self.model_kwargs,
            )
        # Save initial model
        self.save_model(initial_model)

    def __str__(self) -> str:
        return f"SkyNetModelFactory(model_callable={self.model_callable}, players={self.players}, device={self.device}, models_dir={self.models_dir}, model_kwargs={self.model_kwargs})"

    def _get_latest_model_path(self) -> pathlib.Path:
        """Find the newest versioned checkpoint; raw model files are ignored."""
        checkpoint_files = sorted(self.models_dir.glob("checkpoint_*.pth"))
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoints found in {self.models_dir}")
        return checkpoint_files[-1]

    def get_latest_checkpoint_path(self) -> pathlib.Path:
        return self._get_latest_model_path()

    def save_model(
        self,
        model: skynet.SkyNet,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        configuration: typing.Any = None,
        progress: checkpoint.TrainingProgress | None = None,
    ) -> pathlib.Path:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now(tz=datetime.timezone.utc).strftime(
            "%Y%m%d_%H%M%S_%f"
        )
        return checkpoint.save_checkpoint(
            self.models_dir / f"checkpoint_{timestamp}.pth",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            configuration=configuration,
            progress=progress,
        )

    def get_latest_model(self) -> skynet.SkyNet:
        latest_model_path = self._get_latest_model_path()
        model = self.model_callable(
            spatial_input_shape=(
                self.players,
                sj.ROW_COUNT,
                sj.COLUMN_COUNT,
                sj.FINGER_SIZE,
            ),
            non_spatial_input_shape=(sj.GAME_SIZE,),
            value_output_shape=(self.players,),
            policy_output_shape=(sj.MASK_SIZE,),
            device=self.device,
            **self.model_kwargs,
        )
        checkpoint.load_checkpoint(
            latest_model_path,
            model=model,
            restore_rng=False,
            map_location=self.device,
        )
        model.to(self.device)
        return model
