"""Named model construction shared by training and checkpoint readers."""

import torch

from skyjo.learning import observations, skynet


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


def resolve(settings):
    settings = dict(settings)
    name = settings.pop("name", skynet.EQUIVARIANT_ARCHITECTURE_NAME)
    if name != skynet.EQUIVARIANT_ARCHITECTURE_NAME:
        raise ValueError(f"Unsupported model architecture: {name}")
    return {"name": name, **_equivariant_settings(settings)}


def build(settings, *, players, device, auxiliary_objectives=None):
    kwargs = resolve(settings)
    kwargs.pop("name")
    return skynet.EquivariantSkyNet(
        spatial_input_shape=observations.spatial_input_shape(players),
        non_spatial_input_shape=observations.get_non_spatial_input_shape(players),
        value_output_shape=(players,),
        policy_output_shape=observations.action_mask_shape(),
        device=torch.device(device),
        auxiliary_objectives=auxiliary_objectives,
        **kwargs,
    )
