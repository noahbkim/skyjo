import unittest

import numpy as np
import torch

import skyjo_immutable as sj
import skynet


class TestPolicyOutput(unittest.TestCase):
    def test_post_init_shape_assertions(self):
        # Test wrong dimensionality
        with self.assertRaises(AssertionError):
            skynet.PolicyOutput(np.ones(sj.ACTION_SHAPE))  # 2D instead of 3D

        # Test wrong inner dimensions
        with self.assertRaises(AssertionError):
            skynet.PolicyOutput(np.ones((1, 5, 12)))  # Should be (n, 4, 12)

        # Test probabilities not summing to 1
        with self.assertRaises(AssertionError):
            skynet.PolicyOutput(np.ones((1, *sj.ACTION_SHAPE)))  # Sum will be 48, not 1

    def test_from_tensor_output(self):
        # Create a valid probability distribution as a tensor
        probs = torch.zeros((1, sj.ACTION_SHAPE[0] * sj.ACTION_SHAPE[1]))
        probs[0, 0] = 1.0  # Set one probability to 1, rest 0

        policy_output = skynet.PolicyOutput.from_tensor_output(probs)

        # Check shape is correct
        self.assertEqual(policy_output.probabilities.shape, (1, *sj.ACTION_SHAPE))
        # Check sum is 1
        self.assertAlmostEqual(policy_output.probabilities.sum(), 1.0)
        # Check the 1.0 is in the right place
        self.assertAlmostEqual(policy_output.probabilities[0, 0, 0], 1.0)

    def test_mask_invalid_and_renormalize(self):
        # Create a simple probability distribution
        probs = np.zeros((1, 4, 12))
        probs[0, 0, 0] = 0.5
        probs[0, 1, 1] = 0.5
        policy = skynet.PolicyOutput(probs)

        # Test assertion on wrong mask shape
        with self.assertRaises(AssertionError):
            policy.mask_invalid_and_renormalize(np.ones((4, 12)))

        # Test masking and renormalization
        mask = np.zeros((1, 4, 12))
        mask[0, 0, 0] = 1  # Only first action is valid

        masked_policy = policy.mask_invalid_and_renormalize(mask)

        self.assertTrue(masked_policy.is_renormalized)
        self.assertAlmostEqual(masked_policy.probabilities[0, 0, 0], 1.0)
        self.assertAlmostEqual(masked_policy.probabilities[0, 1, 1], 0.0)

        # Test uniform distribution when no valid actions have probability
        probs = np.zeros((1, 4, 12))
        probs[0, 0, 0] = 1.0
        policy = skynet.PolicyOutput(probs)

        mask = np.zeros((1, 4, 12))
        mask[0, 1, 1] = 1  # Only action at (1,1) is valid
        mask[0, 2, 2] = 1  # Only action at (2,2) is valid

        masked_policy = policy.mask_invalid_and_renormalize(mask)

        self.assertAlmostEqual(masked_policy.probabilities[0, 1, 1], 0.5)
        self.assertAlmostEqual(masked_policy.probabilities[0, 2, 2], 0.5)

    def test_get_action_probability(self):
        # Create probability distribution
        probs = np.zeros((1, 4, 12))
        probs[0, 0, :] = 1 / 24  # Uniform distribution for draw actions
        nonzero_action = sj.SkyjoAction(
            action_type=sj.SkyjoActionType.PLACE_DRAWN, row_idx=1, col_idx=2
        )
        probs[
            0,
            nonzero_action.action_type.value,
            sj.Hand.flat_index(nonzero_action.row_idx, nonzero_action.col_idx),
        ] = 0.5  # 0.5 probability for specific board action
        policy = skynet.PolicyOutput(probs)

        # Test DRAW action type
        draw_action = sj.SkyjoAction(action_type=sj.SkyjoActionType.DRAW)
        self.assertAlmostEqual(
            policy.get_action_probability(draw_action), 0.5
        )  # Sum of row 0 should be 0.5

        # Test nonzero prob action
        self.assertAlmostEqual(policy.get_action_probability(nonzero_action), 0.5)

        # Test action with zero probability
        zero_action = sj.SkyjoAction(
            action_type=sj.SkyjoActionType.PLACE_FROM_DISCARD, row_idx=2, col_idx=2
        )
        self.assertAlmostEqual(policy.get_action_probability(zero_action), 0.0)


class TestValueOutput(unittest.TestCase):
    def test_post_init_shape_assertions(self):
        # Test wrong dimensionality (1D)
        with self.assertRaises(AssertionError):
            skynet.ValueOutput(np.ones(3))

        # Test wrong dimensionality (3D)
        with self.assertRaises(AssertionError):
            skynet.ValueOutput(np.ones((2, 3, 4)))

    def test_from_tensor_output(self):
        # Create a sample value tensor (batch_size=2, num_players=3)
        values = [[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]]

        value_output = skynet.ValueOutput.from_tensor_output(torch.tensor(values))

        # Check shape is preserved
        self.assertEqual(value_output.values.shape, (2, 3))

        # Check values are correctly converted
        np.testing.assert_array_almost_equal(value_output.values, np.array(values))


class TestResBlock(unittest.TestCase):
    def test_forward_shape_preservation(self):
        # Test that output shape matches input shape
        batch_size, channels, seq_len = 2, 64, 17
        input_tensor = torch.randn(batch_size, channels, seq_len)

        res_block = skynet.ResBlock(
            in_channels=channels,
            out_channels=(
                channels,
                channels,
            ),  # Same input/output channels for residual
        )

        output = res_block(input_tensor)

        self.assertEqual(output.shape, input_tensor.shape)

    def test_residual_connection(self):
        # Test that the residual connection is working
        # by checking output != 0 when conv weights are 0
        batch_size, channels, seq_len = 2, 64, 17
        input_tensor = torch.randn(batch_size, channels, seq_len)

        res_block = skynet.ResBlock(
            in_channels=channels, out_channels=(channels, channels)
        )

        # Zero out weights of convolutions
        with torch.no_grad():
            res_block.conv1.weight.zero_()
            res_block.conv2.weight.zero_()

        output = res_block(input_tensor)

        # Output should equal input due to residual connection
        torch.testing.assert_close(output, torch.nn.ReLU()(input_tensor))


class TestSkyNet(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.in_channels = 29  # Typical input channels for Skyjo state
        self.seq_length = 17  # Typical sequence length
        self.num_players = 3

        self.model = skynet.SkyNet(
            in_channels=self.in_channels, num_players=self.num_players
        )

        self.test_input = torch.randn(
            self.batch_size, self.in_channels, self.seq_length
        )

    def test_initialization(self):
        # Test model creates with expected parameters
        self.assertEqual(self.model.conv1.in_channels, self.in_channels)
        self.assertEqual(self.model.value_head.linear1.out_features, self.num_players)

        # Test policy head output shape
        policy_out_features = np.prod(sj.ACTION_SHAPE)
        self.assertEqual(
            self.model.policy_head.linear1.out_features, policy_out_features
        )

    def test_forward(self):
        # Test forward pass returns expected shapes
        policy_probs, values = self.model(self.test_input)

        # Check policy output shape
        expected_policy_shape = (self.batch_size, *sj.ACTION_SHAPE)
        self.assertEqual(policy_probs.shape, expected_policy_shape)

        # Check value output shape
        expected_value_shape = (self.batch_size, self.num_players)
        self.assertEqual(values.shape, expected_value_shape)

        # Check outputs are valid probabilities
        self.assertTrue(torch.allclose(policy_probs.sum(dim=(1, 2)), torch.tensor(1.0)))
        self.assertTrue(torch.allclose(values.sum(dim=1), torch.tensor(1.0)))

    def test_predict(self):
        # Test predict method with numpy input
        state_repr = np.random.randn(self.in_channels, self.seq_length)

        output = self.model.predict(state_repr)

        # Check output types
        self.assertIsInstance(output, skynet.SkyNetOutput)
        self.assertIsInstance(output.policy_output, skynet.PolicyOutput)
        self.assertIsInstance(output.value_output, skynet.ValueOutput)

        # Check policy output shape
        self.assertEqual(
            output.policy_output.probabilities.shape, (1, *sj.ACTION_SHAPE)
        )

        # Check value output shape
        self.assertEqual(output.value_output.values.shape, (1, self.num_players))


if __name__ == "__main__":
    unittest.main()
