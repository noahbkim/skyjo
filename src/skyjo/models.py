"""Named model construction shared by training and checkpoint readers."""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from . import game, skynet


def _equivariant_settings(settings):
    defaults = {
        "embedding_dimensions": 16,
        "global_state_embedding_dimensions": 32,
        "num_heads": 2,
    }
    unknown = settings.keys() - defaults.keys()
    if unknown:
        raise ValueError(f"Unknown model settings: {sorted(unknown)}")
    result = defaults | settings
    for key, value in result.items():
        if type(value) is not int or value <= 0:
            raise ValueError(f"model.{key} must be a positive integer")
    for key in ("embedding_dimensions", "global_state_embedding_dimensions"):
        if result[key] % result["num_heads"]:
            raise ValueError(f"model.{key} must be divisible by num_heads")
    return result


@dataclass(frozen=True)
class ModelDefinition:
    constructor: Callable
    resolve_settings: Callable


REGISTRY = {
    skynet.EQUIVARIANT_ARCHITECTURE_NAME: ModelDefinition(
        skynet.EquivariantSkyNet, _equivariant_settings
    ),
}


def resolve(settings):
    settings = dict(settings)
    name = settings.pop("name", skynet.EQUIVARIANT_ARCHITECTURE_NAME)
    if name not in REGISTRY:
        raise ValueError(f"Unsupported model architecture: {name}")
    return {"name": name, **REGISTRY[name].resolve_settings(settings)}


def constructor_and_kwargs(settings):
    resolved = resolve(settings)
    return REGISTRY[resolved.pop("name")].constructor, resolved


def build(settings, *, players, device, auxiliary_objectives=None):
    constructor, kwargs = constructor_and_kwargs(settings)
    return constructor(
        spatial_input_shape=(
            players,
            game.ROW_COUNT,
            game.COLUMN_COUNT,
            game.FINGER_SIZE,
        ),
        non_spatial_input_shape=skynet.get_non_spatial_input_shape(players),
        value_output_shape=(players,),
        policy_output_shape=(game.MASK_SIZE,),
        device=torch.device(device),
        auxiliary_objectives=auxiliary_objectives,
        **kwargs,
    )
