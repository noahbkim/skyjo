import abc
from typing import Self

import numpy as np


class AbstractGameAction(abc.ABC):
    @abc.abstractmethod
    def from_numpy(cls, numpy_action: np.ndarray) -> Self:
        pass

    @abc.abstractmethod
    def numpy(self) -> np.ndarray:
        pass


class AbstractImmutableGameState(abc.ABC):
    @abc.abstractmethod
    def next_state(self, action: AbstractGameAction) -> Self:
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
