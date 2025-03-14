import abc
import pathlib
import typing

import numpy as np
import torch


class AbstractGameAction(abc.ABC):
    @abc.abstractmethod
    def from_numpy(cls, numpy_action: np.ndarray) -> typing.Self:
        pass

    @abc.abstractmethod
    def numpy(self) -> np.ndarray:
        pass


class AbstractImmutableGameState(abc.ABC):
    @abc.abstractmethod
    def next_state(self, action: AbstractGameAction) -> typing.Self:
        pass

    @abc.abstractmethod
    def numpy(self) -> np.ndarray:
        pass

    @abc.abstractmethod
    def game_ended(self) -> bool:
        pass


class AbstractModel(abc.ABC):
    @abc.abstractmethod
    def predict(self, state_repr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pass

    @abc.abstractmethod
    def create_valid_actions_mask(
        self, valid_actions: list[AbstractGameAction]
    ) -> np.ndarray:
        pass

    def save_checkpoint(
        self, folder: str = "checkpoint", filename: str = "checkpoint.pth.tar"
    ):
        filepath = pathlib.Path(folder) / filename
        if not filepath.parent.exists():
            print(
                "Checkpoint Directory does not exist! Making directory {}".format(
                    folder
                )
            )
            filepath.parent.mkdir()
        else:
            print("Checkpoint Directory exists! ")
        torch.save(
            {
                "state_dict": self.state_dict(),
            },
            filepath,
        )

    def load_checkpoint(
        self,
        folder: str = "checkpoint",
        filename: str = "checkpoint.pth.tar",
        map_location: typing.Callable | None = None,
    ):
        filepath = pathlib.Path(folder) / filename
        if not filepath.exists():
            raise ("No model in path {}".format(filepath))
        checkpoint = torch.load(filepath, map_location=map_location)
        self.load_state_dict(checkpoint["state_dict"])
