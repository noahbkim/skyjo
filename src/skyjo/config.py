import dataclasses
import json
import pathlib
import typing


class Config:
    def kwargs(self, prefix: str = "") -> dict[str, typing.Any]:
        kwargs = dataclasses.asdict(self)
        if prefix:
            return {f"{prefix}_{k}": v for k, v in kwargs.items()}
        return kwargs


@dataclasses.dataclass
class LearningObjectivesConfig(Config):
    """Objective and target-generation settings, independent of training budget."""

    auxiliary_objectives: dict[str, float] = dataclasses.field(default_factory=dict)
    outcome_rollouts: int = 32
    target_seed: int = 0

    def __post_init__(self):
        from . import objectives

        self.auxiliary_objectives = objectives.resolve(self.auxiliary_objectives).weights
        if type(self.outcome_rollouts) is not int or self.outcome_rollouts < 1:
            raise ValueError("outcome_rollouts must be a positive integer")

    @classmethod
    def load(cls, path: pathlib.Path) -> typing.Self:
        return cls(**json.loads(path.read_text()))
