import dataclasses
import typing


class Config:
    def kwargs(self, prefix: str = "") -> dict[str, typing.Any]:
        kwargs = dataclasses.asdict(self)
        if prefix:
            return {f"{prefix}_{k}": v for k, v in kwargs.items()}
        return kwargs
