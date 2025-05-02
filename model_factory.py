"""Model factory for loading and saving models with consistent behavior across
processes."""

import pathlib
import re
import typing

import torch

import skyjo as sj
import skynet


# Model factory
class SkyNetModelFactory:
    def __init__(
        self,
        players: int = 2,
        dropout_rate: float = 0.5,
        device: torch.device = torch.device("cpu"),
        models_dir: pathlib.Path = pathlib.Path("models"),
        model_callable: typing.Callable[[], skynet.SkyNet] = skynet.SkyNet1D,
        model_kwargs: dict[str, typing.Any] = {},
    ):
        self.models_dir = models_dir
        self.players = players
        self.dropout_rate = dropout_rate
        self.device = device
        self.model_kwargs = model_kwargs
        self.model_callable = model_callable
        self.training_model = self.model_callable(
            spatial_input_shape=(
                players,
                sj.ROW_COUNT,
                sj.COLUMN_COUNT,
                sj.FINGER_SIZE,
            ),
            non_spatial_input_shape=(sj.GAME_SIZE,),
            value_output_shape=(players,),
            policy_output_shape=(sj.MASK_SIZE,),
            dropout_rate=self.dropout_rate,
            device=self.device,
            **self.model_kwargs,
        )
        # Save initial model
        self.save_model(self.training_model)

    def _get_latest_model_path(self) -> pathlib.Path:
        """Finds the model file with the latest timestamp in the filename."""
        model_files = list(self.models_dir.glob("model_*.pth"))
        if not model_files:
            # If no models saved yet, return path for the initial model
            # Assuming the initial model is saved with a specific name or timestamp 0
            # For simplicity, let's assume save_model ensures one exists or handles it.
            # Re-evaluating: It's better to raise an error if called before first save.
            raise FileNotFoundError(f"No model files found in {self.models_dir}")

        # Regex to extract the timestamp YYYYMMDD_HHMMSS
        pattern = re.compile(r"model_(\d{8}_\d{6})\.pth")

        latest_file = None
        latest_timestamp_str = ""

        for file_path in model_files:
            match = pattern.match(file_path.name)
            if match:
                timestamp_str = match.group(1)
                # String comparison works for YYYYMMDD_HHMMSS format
                if timestamp_str > latest_timestamp_str:
                    latest_timestamp_str = timestamp_str
                    latest_file = file_path

        if latest_file is None:
            # This case should ideally not happen if files exist and saving adheres to format
            raise FileNotFoundError(
                f"No model files matching the pattern 'model_YYYYMMDD_HHMMSS.pth' found in {self.models_dir}"
            )

        return latest_file

    def save_model(self, model: skynet.SkyNet) -> pathlib.Path:
        return model.save(self.models_dir)

    def get_latest_model(self) -> skynet.SkyNet:
        latest_model_path = self._get_latest_model_path()
        model = skynet.SkyNet1D(
            spatial_input_shape=(
                self.players,
                sj.ROW_COUNT,
                sj.COLUMN_COUNT,
                sj.FINGER_SIZE,
            ),
            non_spatial_input_shape=(sj.GAME_SIZE,),
            value_output_shape=(self.players,),
            policy_output_shape=(sj.MASK_SIZE,),
            dropout_rate=self.dropout_rate,
            device=self.device,
            **self.model_kwargs,
        )
        model.load_state_dict(torch.load(latest_model_path, weights_only=True))
        model.to(self.device)
        return model
