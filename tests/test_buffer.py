import pathlib

from skyjo import buffer, train_utils


def test_replay_buffer_defaults_to_core_target_specs():
    replay_buffer = buffer.ReplayBuffer(
        max_size=16,
        spatial_input_shape=(3, 4, 4, 17),
        non_spatial_input_shape=(10,),
        action_mask_shape=(23,),
    )

    assert replay_buffer.target_names == train_utils.CORE_TARGET_NAMES
    assert replay_buffer.target_specs == (
        buffer.TargetShapeSpec(
            name=train_utils.VALUE_TARGET_NAME,
            shape=(3,),
        ),
        buffer.TargetShapeSpec(
            name=train_utils.POLICY_TARGET_NAME,
            shape=(23,),
        ),
    )


def test_replay_buffer_from_config_preserves_path_with_default_target_specs():
    path = pathlib.Path("data/training_data/test-buffer.pkl")
    config = buffer.Config(
        max_size=16,
        spatial_input_shape=(2, 4, 4, 17),
        non_spatial_input_shape=(10,),
        action_mask_shape=(12,),
        path=path,
    )

    replay_buffer = buffer.ReplayBuffer.from_config(config)

    assert replay_buffer.path == path
    assert replay_buffer.target_names == train_utils.CORE_TARGET_NAMES


def test_replay_buffer_accepts_round_score_aux_target_spec():
    replay_buffer = buffer.ReplayBuffer(
        max_size=16,
        spatial_input_shape=(2, 4, 4, 17),
        non_spatial_input_shape=(10,),
        action_mask_shape=(23,),
        target_specs=(
            buffer.TargetShapeSpec(
                name=train_utils.VALUE_TARGET_NAME,
                shape=(2,),
            ),
            buffer.TargetShapeSpec(
                name=train_utils.POLICY_TARGET_NAME,
                shape=(23,),
            ),
            buffer.TargetShapeSpec(
                name=train_utils.ROUND_SCORE_TARGET_NAME,
                shape=(2,),
            ),
        ),
    )

    assert replay_buffer.target_names == (
        train_utils.VALUE_TARGET_NAME,
        train_utils.POLICY_TARGET_NAME,
        train_utils.ROUND_SCORE_TARGET_NAME,
    )
