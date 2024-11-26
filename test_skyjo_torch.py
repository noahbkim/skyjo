import unittest

import torch

import skyjo


class TestFinger(unittest.TestCase):
    def setUp(self):
        self.num_categories = 15 + 2

    """Test Tensor representations of Fingers"""
    def test__face_down_tensor_repr(self) -> None:
        finger = skyjo.Finger(_card=2)
        expected_tensor = torch.zeros(1, self.num_categories)
        expected_tensor[0, -2] = 1
        torch.testing.assert_close(finger.tensor_repr(), expected_tensor)

    def test__cleared_tensor_repr(self) -> None:
        finger = skyjo.Finger(_card=2)
        finger._clear()
        expected_tensor = torch.zeros(1,  self.num_categories)
        expected_tensor[0, -1] = 1
        torch.testing.assert_close(finger.tensor_repr(), expected_tensor)

    def test__flipped_card_tensor_repr(self) -> None:
        finger = skyjo.Finger(_card=2)
        finger._flip_card()
        expected_tensor = torch.zeros(1,  self.num_categories)
        expected_tensor[0, 2 + 2] = 1
        torch.testing.assert_close(finger.tensor_repr(), expected_tensor)


class TestHand(unittest.TestCase):
    """Test Tensor representations of Hands"""
    def setUp(self) -> None:
        self.num_cards = skyjo.HAND_ROWS * skyjo.HAND_COLUMNS 
        self.num_categories = 15 + 2 # number of different cards, face down card, cleared card
        
    def test__all_flipped(self) -> None:
        hand = skyjo.Hand([skyjo.Finger(_card=val) for val in range(self.num_cards)])
        hand._flip_all_cards()
        
        expected_finger_tensors = []
        for i in range(self.num_cards):
            finger_tensor = torch.zeros(1, self.num_categories)
            finger_tensor[0, i + 2] = 1
            expected_finger_tensors.append(finger_tensor)
        expected_tensor = torch.cat(expected_finger_tensors)
        torch.testing.assert_close(hand.tensor_repr(), expected_tensor)

    def test__face_down(self) -> None:
        hand = skyjo.Hand([skyjo.Finger(_card=val) for val in range(self.num_cards)])
        face_down_finger_tensor = torch.zeros(1, self.num_categories)
        face_down_finger_tensor[0, -2] = 1
        expected_finger_tensors = [face_down_finger_tensor for _ in range(self.num_cards)]
        expected_tensor = torch.cat(expected_finger_tensors)
        torch.testing.assert_close(hand.tensor_repr(), expected_tensor)

    def test__cleared_column(self) -> None:
        hand = skyjo.Hand([skyjo.Finger(_card=4) for val in range(self.num_cards)])
        hand._flip_all_cards() # includes a _try_clear

        cleared_tensor = torch.zeros(1, self.num_categories)
        cleared_tensor[0, -1] = 1
        expected_tensor = torch.cat([cleared_tensor for _ in range(self.num_cards)])
        torch.testing.assert_close(hand.tensor_repr(), expected_tensor)


if __name__ == '__main__':
    unittest.main()
