"""Abstract base classes for game state and actions"""

import typing
from abc import ABC, abstractmethod

import numpy as np


class AbstractGameAction(ABC):
    """Abstract base class for game actions"""

    @abstractmethod
    def numpy(self) -> np.ndarray:
        pass


class AbstractGameStateValue(ABC):
    """Abstract base class for game outcomes"""

    @abstractmethod
    def numpy(self) -> np.ndarray:
        pass

    @abstractmethod
    def value_from_perspective_of(self, player: int) -> float:
        """Returns the value of the outcome from the perspective of the given player"""
        pass


class AbstractImmutableGameState(ABC):
    """Abstract base class for the actual game state"""

    @property
    @abstractmethod
    def game_outcome(self) -> AbstractGameStateValue | None:
        pass

    @property
    @abstractmethod
    def valid_actions(self) -> list[AbstractGameAction]:
        pass

    @property
    @abstractmethod
    def curr_player(self) -> int:
        pass

    @property
    @abstractmethod
    def num_players(self) -> int:
        pass

    @property
    def game_is_over(self) -> bool:
        return self.game_outcome is not None

    @abstractmethod
    def numpy(self) -> np.ndarray:
        pass

    @abstractmethod
    def apply_action(self, action: AbstractGameAction) -> typing.Self:
        pass


# Model Interface
class AbstractModelValueOutput(ABC):
    @property
    @abstractmethod
    def game_state_value(self) -> AbstractGameStateValue:
        pass


class AbstractModelPolicyOutput(ABC):
    @abstractmethod
    def probabilities(self) -> dict[typing.Any, float]:
        pass


class AbstractModelPrediction(ABC):
    @property
    @abstractmethod
    def value_output(self) -> AbstractModelValueOutput:
        pass

    # might include other outputs like policy, point differential, etc.


class AbstractModel(ABC):
    @abstractmethod
    def predict(self, state_repr: np.ndarray) -> AbstractModelPrediction:
        pass

    @abstractmethod
    def afterstate_predict(
        self, state: AbstractImmutableGameState
    ) -> AbstractModelPrediction:
        pass

    @abstractmethod
    def create_valid_actions_mask(
        self, valid_actions: list[AbstractGameAction]
    ) -> np.ndarray:
        pass


class AbstractPlayer(ABC):
    """Abstract base class for players"""

    @abstractmethod
    def select_action(self, state: AbstractImmutableGameState) -> AbstractGameAction:
        pass
