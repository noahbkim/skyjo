import unittest

import numpy as np

import skyjo_immutable as sj


class TestSkyjoAction(unittest.TestCase):
    def test_action_equality(self):
        """Test that identical actions are equal"""
        action1 = sj.SkyjoAction(sj.SkyjoActionType.DRAW)
        action2 = sj.SkyjoAction(sj.SkyjoActionType.DRAW)
        self.assertEqual(action1.action_type, action2.action_type)
        self.assertEqual(action1.row_idx, action2.row_idx)
        self.assertEqual(action1.col_idx, action2.col_idx)

        action1 = sj.SkyjoAction(sj.SkyjoActionType.PLACE_DRAWN, 1, 2)
        action2 = sj.SkyjoAction(sj.SkyjoActionType.PLACE_DRAWN, 1, 2)
        self.assertEqual(action1.action_type, action2.action_type)
        self.assertEqual(action1.row_idx, action2.row_idx)
        self.assertEqual(action1.col_idx, action2.col_idx)

    def test_action_inequality(self):
        """Test that different actions are not equal"""
        action1 = sj.SkyjoAction(sj.SkyjoActionType.DRAW)
        action2 = sj.SkyjoAction(sj.SkyjoActionType.END_ROUND)
        self.assertNotEqual(action1.action_type, action2.action_type)

        action1 = sj.SkyjoAction(sj.SkyjoActionType.PLACE_DRAWN, 1, 2)
        action2 = sj.SkyjoAction(sj.SkyjoActionType.PLACE_DRAWN, 1, 3)
        self.assertEqual(action1.action_type, action2.action_type)
        self.assertEqual(action1.row_idx, action2.row_idx)
        self.assertNotEqual(action1.col_idx, action2.col_idx)

        action1 = sj.SkyjoAction(sj.SkyjoActionType.PLACE_DRAWN, 1, 2)
        action2 = sj.SkyjoAction(sj.SkyjoActionType.PLACE_FROM_DISCARD, 1, 2)
        self.assertNotEqual(action1.action_type, action2.action_type)
        self.assertEqual(action1.row_idx, action2.row_idx)
        self.assertEqual(action1.col_idx, action2.col_idx)

    def test_numpy(self):
        """Test the numpy representation of an action."""
        action = sj.SkyjoAction(sj.SkyjoActionType.PLACE_DRAWN, 1, 2)
        np_repr = action.numpy()

        # Check shape and dtype
        self.assertEqual(np_repr.shape, sj.ACTION_SHAPE)
        self.assertEqual(np_repr.dtype, np.int8)

        # Check it's one-hot encoded
        self.assertEqual(np.sum(np_repr), 1)

        # Check the correct value is set
        self.assertEqual(np_repr[sj.SkyjoActionType.PLACE_DRAWN.value, 1, 2], 1)

        # Test DRAW action
        action_draw = sj.SkyjoAction(sj.SkyjoActionType.DRAW)
        np_repr_draw = action_draw.numpy()
        self.assertEqual(np_repr_draw.shape, sj.ACTION_SHAPE)
        self.assertEqual(np.sum(np_repr_draw), 1)
        # Draw action sets (0, 0) for its type
        self.assertEqual(np_repr_draw[sj.SkyjoActionType.DRAW.value, 0, 0], 1)


class TestCard(unittest.TestCase):
    def test_card_point_values(self):
        """Test that cards have the correct point values"""
        self.assertEqual(sj.Card.NEGATIVE_TWO.point_value, -2)
        self.assertEqual(sj.Card.NEGATIVE_ONE.point_value, -1)
        self.assertEqual(sj.Card.ZERO.point_value, 0)
        self.assertEqual(sj.Card.ONE.point_value, 1)
        self.assertEqual(sj.Card.TWO.point_value, 2)
        self.assertEqual(sj.Card.THREE.point_value, 3)
        self.assertEqual(sj.Card.FOUR.point_value, 4)
        self.assertEqual(sj.Card.FIVE.point_value, 5)
        self.assertEqual(sj.Card.SIX.point_value, 6)
        self.assertEqual(sj.Card.SEVEN.point_value, 7)
        self.assertEqual(sj.Card.EIGHT.point_value, 8)
        self.assertEqual(sj.Card.NINE.point_value, 9)
        self.assertEqual(sj.Card.TEN.point_value, 10)
        self.assertEqual(sj.Card.ELEVEN.point_value, 11)
        self.assertEqual(sj.Card.TWELVE.point_value, 12)
        self.assertEqual(sj.Card.FACE_DOWN.point_value, 0)
        self.assertEqual(sj.Card.CLEARED.point_value, 0)

    def test_card_clearable(self):
        """Test that cards have the correct clearable property"""
        self.assertTrue(sj.Card.NEGATIVE_TWO.is_clearable)
        self.assertTrue(sj.Card.NEGATIVE_ONE.is_clearable)
        self.assertTrue(sj.Card.ZERO.is_clearable)
        self.assertTrue(sj.Card.ONE.is_clearable)
        self.assertTrue(sj.Card.TWO.is_clearable)
        self.assertTrue(sj.Card.THREE.is_clearable)
        self.assertTrue(sj.Card.FOUR.is_clearable)
        self.assertTrue(sj.Card.FIVE.is_clearable)
        self.assertTrue(sj.Card.SIX.is_clearable)
        self.assertTrue(sj.Card.SEVEN.is_clearable)
        self.assertTrue(sj.Card.EIGHT.is_clearable)
        self.assertTrue(sj.Card.NINE.is_clearable)
        self.assertTrue(sj.Card.TEN.is_clearable)
        self.assertTrue(sj.Card.ELEVEN.is_clearable)
        self.assertTrue(sj.Card.TWELVE.is_clearable)
        self.assertFalse(sj.Card.FACE_DOWN.is_clearable)
        self.assertFalse(sj.Card.CLEARED.is_clearable)

    def test_one_hot_encoding(self):
        """Test card one-hot encoding"""
        # Test ZERO card
        one_hot = sj.Card.ZERO.one_hot_encoding()
        self.assertEqual(one_hot.shape, (17,))  # 15 card types + face_down + cleared
        self.assertEqual(np.sum(one_hot), 1)
        self.assertEqual(one_hot[sj.Card.ZERO.one_hot_encoding_index], 1)

        # Test FACE_DOWN card
        one_hot = sj.Card.FACE_DOWN.one_hot_encoding()
        self.assertEqual(np.sum(one_hot), 1)
        self.assertEqual(one_hot[sj.Card.FACE_DOWN.one_hot_encoding_index], 1)

    def test_from_one_hot_encoding(self):
        """Test creating a card from one-hot encoding"""
        one_hot = np.zeros(sj.CARD_TYPES, dtype=np.int8)
        one_hot[sj.Card.ZERO.one_hot_encoding_index] = 1  # ZERO card
        card = sj.Card.from_one_hot_encoding(one_hot)
        self.assertEqual(card, sj.Card.ZERO)

        one_hot = np.zeros(sj.CARD_TYPES, dtype=np.int8)
        one_hot[sj.Card.FACE_DOWN.one_hot_encoding_index] = 1  # FACE_DOWN card
        card = sj.Card.from_one_hot_encoding(one_hot)
        self.assertEqual(card, sj.Card.FACE_DOWN)


class TestCardCounts(unittest.TestCase):
    def test_create_initial_deck_counts(self):
        """Test that initial deck counts are correct"""
        counts = sj.CardCounts.create_initial_deck_counts()
        self.assertEqual(counts.get_card_count(sj.Card.NEGATIVE_TWO), 5)
        self.assertEqual(counts.get_card_count(sj.Card.NEGATIVE_ONE), 10)
        self.assertEqual(counts.get_card_count(sj.Card.ZERO), 15)
        self.assertEqual(counts.get_card_count(sj.Card.ONE), 10)
        # Special cards should have count 0
        self.assertEqual(counts.get_card_count(sj.Card.FACE_DOWN), 0)
        self.assertEqual(counts.get_card_count(sj.Card.CLEARED), 0)

    def test_copy(self):
        """Test that copying counts creates a new object with the same values"""
        counts = sj.CardCounts.create_initial_deck_counts()
        counts_copy = counts.copy()
        self.assertIsNot(counts, counts_copy)

        # Check values are the same
        for card in sj.Card:
            self.assertEqual(
                counts.get_card_count(card), counts_copy.get_card_count(card)
            )

    def test_add_card(self):
        """Test adding a card increments the count"""
        counts = sj.CardCounts.create_initial_deck_counts()
        initial_count = counts.get_card_count(sj.Card.ZERO)
        new_counts = counts.add_card(sj.Card.ZERO)

        # Original object should be unchanged
        self.assertEqual(counts.get_card_count(sj.Card.ZERO), initial_count)

        # New object should have incremented count
        self.assertEqual(new_counts.get_card_count(sj.Card.ZERO), initial_count + 1)

    def test_remove_card(self):
        """Test removing a card decrements the count"""
        counts = sj.CardCounts.create_initial_deck_counts()
        initial_count = counts.get_card_count(sj.Card.ZERO)
        new_counts = counts.remove_card(sj.Card.ZERO)

        # Original object should be unchanged
        self.assertEqual(counts.get_card_count(sj.Card.ZERO), initial_count)

        # New object should have decremented count
        self.assertEqual(new_counts.get_card_count(sj.Card.ZERO), initial_count - 1)

    def test_remove_cards(self):
        """Test removing multiple cards at once"""
        # Create a card counts object with known values
        counts = sj.CardCounts.create_initial_deck_counts()

        # Record initial counts for cards we'll remove
        initial_zero_count = counts.get_card_count(sj.Card.ZERO)
        initial_one_count = counts.get_card_count(sj.Card.ONE)
        initial_five_count = counts.get_card_count(sj.Card.FIVE)

        # Create a list of cards to remove
        cards_to_remove = [
            sj.Card.ZERO,
            sj.Card.ZERO,  # Remove ZERO twice
            sj.Card.ONE,
            sj.Card.FIVE,
            sj.Card.FIVE,  # Remove FIVE twice
        ]

        # Call remove_cards
        new_counts = counts.remove_cards(cards_to_remove)

        # Original object should be unchanged
        self.assertEqual(counts.get_card_count(sj.Card.ZERO), initial_zero_count)
        self.assertEqual(counts.get_card_count(sj.Card.ONE), initial_one_count)
        self.assertEqual(counts.get_card_count(sj.Card.FIVE), initial_five_count)

        # New object should have decremented counts
        self.assertEqual(
            new_counts.get_card_count(sj.Card.ZERO), initial_zero_count - 2
        )
        self.assertEqual(new_counts.get_card_count(sj.Card.ONE), initial_one_count - 1)
        self.assertEqual(
            new_counts.get_card_count(sj.Card.FIVE), initial_five_count - 2
        )

        # Cards not in the list should remain unchanged
        for card in sj.Card:
            if card not in cards_to_remove:
                self.assertEqual(
                    new_counts.get_card_count(card), counts.get_card_count(card)
                )

    def test_total_points(self):
        """Test calculating total points"""
        # Create a simple counts object with known cards
        counts_array = np.zeros(sj.CARD_TYPES, dtype=np.int8)
        counts_array[sj.Card.NEGATIVE_TWO.one_hot_encoding_index] = (
            2  # 2 cards of -2 points
        )
        counts_array[sj.Card.THREE.one_hot_encoding_index] = 3  # 3 cards of 3 points
        counts = sj.CardCounts(_counts=counts_array)

        # Expected: 2*(-2) + 3*3 = -4 + 9 = 5
        self.assertEqual(counts.total_points, 5)

    def test_numpy(self):
        """Test the numpy representation of card counts."""
        counts = sj.CardCounts.create_initial_deck_counts()
        np_repr = counts.numpy()

        # Check shape and dtype
        self.assertEqual(np_repr.shape, (sj.CARD_TYPES,))
        self.assertEqual(np_repr.dtype, np.int8)

        # Check some specific counts
        self.assertEqual(np_repr[sj.Card.ZERO.one_hot_encoding_index], 15)
        self.assertEqual(np_repr[sj.Card.FACE_DOWN.one_hot_encoding_index], 0)

        # Check that it returns a copy
        np_repr[0] = 99  # Modify the returned array
        self.assertNotEqual(counts.numpy()[0], 99)  # Original should be unchanged


class TestDiscardPile(unittest.TestCase):
    def test_initialization(self):
        """Test discard pile initialization"""
        pile = sj.DiscardPile()
        self.assertIsNone(pile.top_card)

    def test_discard(self):
        """Test discarding a card"""
        pile = sj.DiscardPile()
        new_pile = pile.discard(sj.Card.THREE)

        # Original pile should be unchanged
        self.assertIsNone(pile.top_card)

        # New pile should have the card on top
        self.assertEqual(new_pile.top_card, sj.Card.THREE)
        self.assertEqual(
            new_pile.discarded_card_counts.get_card_count(sj.Card.THREE), 1
        )

    def test_replace_top_card(self):
        """Test replacing the top card"""
        pile = sj.DiscardPile()
        pile = pile.discard(sj.Card.THREE)
        new_pile = pile.replace_top_card(sj.Card.FIVE)

        # Original pile should be unchanged
        self.assertEqual(pile.top_card, sj.Card.THREE)
        self.assertEqual(pile.discarded_card_counts.get_card_count(sj.Card.THREE), 1)
        self.assertEqual(pile.discarded_card_counts.get_card_count(sj.Card.FIVE), 0)

        # New pile should have the new card on top
        self.assertEqual(new_pile.top_card, sj.Card.FIVE)
        self.assertEqual(
            new_pile.discarded_card_counts.get_card_count(sj.Card.THREE), 0
        )
        self.assertEqual(new_pile.discarded_card_counts.get_card_count(sj.Card.FIVE), 1)

    def test_copy(self):
        """Test copying the discard pile"""
        pile = sj.DiscardPile()
        pile = pile.discard(sj.Card.THREE)
        pile_copy = pile.copy()

        # Copy should be a different object but with the same values
        self.assertIsNot(pile, pile_copy)
        self.assertEqual(pile.top_card, pile_copy.top_card)
        for card in sj.Card:
            self.assertEqual(
                pile.discarded_card_counts.get_card_count(card),
                pile_copy.discarded_card_counts.get_card_count(card),
            )

    def test_numpy(self):
        """Test the numpy representation of a discard pile."""
        pile = sj.DiscardPile()
        # Test empty pile
        np_repr_empty = pile.numpy()
        self.assertEqual(np_repr_empty.shape, (2 * sj.CARD_TYPES,))
        self.assertEqual(np_repr_empty.dtype, np.int8)
        np.testing.assert_array_equal(
            np_repr_empty, np.zeros(2 * sj.CARD_TYPES, dtype=np.int8)
        )

        # Test pile with one card
        pile = pile.discard(sj.Card.THREE)
        np_repr = pile.numpy()

        # Check shape and dtype
        self.assertEqual(np_repr.shape, (2 * sj.CARD_TYPES,))
        self.assertEqual(np_repr.dtype, np.int8)

        # Check counts part (first half)
        self.assertEqual(np_repr[sj.Card.THREE.one_hot_encoding_index], 1)
        self.assertEqual(np.sum(np_repr[: sj.CARD_TYPES]), 1)  # Only one card discarded

        # Check top card part (second half)
        top_card_one_hot = sj.Card.THREE.one_hot_encoding()
        np.testing.assert_array_equal(np_repr[sj.CARD_TYPES :], top_card_one_hot)

        # Check that it returns a copy
        np_repr_copy = pile.numpy()
        np_repr_copy[0] = 99  # Modify the returned array
        self.assertNotEqual(pile.numpy()[0], 99)  # Original should be unchanged


class TestHand(unittest.TestCase):
    def test_create_initial_hand(self):
        """Test creating an initial hand"""
        hand = sj.Hand.create_initial_hand()

        # All cards should be face down
        for row in range(sj.NUM_ROWS):
            for col in range(sj.NUM_COLUMNS):
                self.assertEqual(hand.get_card(row, col), sj.Card.FACE_DOWN)

        # Check face down indices
        face_down_indices = hand.face_down_indices
        self.assertEqual(len(face_down_indices), sj.NUM_ROWS * sj.NUM_COLUMNS)

        # Check cleared indices (should be empty)
        cleared_indices = hand.cleared_indices
        self.assertEqual(len(cleared_indices), 0)

    def test_flip_card(self):
        """Test flipping a card"""
        hand = sj.Hand.create_initial_hand()
        next_hand = hand.flip(sj.Card.ZERO, 0, 0)

        # Check the card was flipped
        self.assertEqual(next_hand.get_card(0, 0), sj.Card.ZERO)

        # Original hand should be unchanged
        self.assertEqual(hand.get_card(0, 0), sj.Card.FACE_DOWN)

        # Check face down indices decreased
        self.assertEqual(
            len(next_hand.face_down_indices), sj.NUM_ROWS * sj.NUM_COLUMNS - 1
        )

    def test_flip_card_assertion(self):
        """Test that flipping an already flipped card raises an assertion"""
        hand = sj.Hand.create_initial_hand()
        next_hand = hand.flip(sj.Card.ZERO, 0, 0)

        # Try to flip the already flipped card
        with self.assertRaises(AssertionError):
            next_hand.flip(sj.Card.ONE, 0, 0)

    def test_column_clearing(self):
        """Test column clearing when 3 of the same card are in a column"""
        hand = sj.Hand.create_initial_hand()

        # Flip 3 of the same card in a column
        next_hand = hand.flip(sj.Card.ZERO, 0, 0)
        next_hand = next_hand.flip(sj.Card.ZERO, 1, 0)
        next_hand = next_hand.flip(sj.Card.ZERO, 2, 0)

        # Check the column was cleared
        self.assertEqual(next_hand.get_card(0, 0), sj.Card.CLEARED)
        self.assertEqual(next_hand.get_card(1, 0), sj.Card.CLEARED)
        self.assertEqual(next_hand.get_card(2, 0), sj.Card.CLEARED)

        # Check that the cleared_card field is set
        self.assertEqual(next_hand.cleared_card, sj.Card.ZERO)

        # Check cleared indices
        cleared_indices = next_hand.cleared_indices
        self.assertEqual(len(cleared_indices), 3)  # One column of 3 cards

        # Check face down indices (12 - 3 = 9)
        self.assertEqual(
            len(next_hand.face_down_indices), sj.NUM_ROWS * sj.NUM_COLUMNS - 3
        )

    def test_replace_card(self):
        """Test replacing a card"""
        hand = sj.Hand.create_initial_hand()

        # Replace a face-down card
        next_hand, replaced_card = hand.replace(0, 0, sj.Card.ONE)

        # Check the replacement happened
        self.assertEqual(next_hand.get_card(0, 0), sj.Card.ONE)
        self.assertEqual(replaced_card, sj.Card.FACE_DOWN)

        # Original hand should be unchanged
        self.assertEqual(hand.get_card(0, 0), sj.Card.FACE_DOWN)

    def test_replace_card_assertion(self):
        """Test that replacing a cleared card raises an assertion"""
        hand = sj.Hand.create_initial_hand()

        # Create a cleared column
        next_hand = hand.flip(sj.Card.ZERO, 0, 0)
        next_hand = next_hand.flip(sj.Card.ZERO, 1, 0)
        next_hand = next_hand.flip(sj.Card.ZERO, 2, 0)

        # Try to replace a cleared card
        with self.assertRaises(AssertionError):
            next_hand.replace(0, 0, sj.Card.ONE)

    def test_replace_card_clearing(self):
        """Test that replacing a card can trigger column clearing"""
        hand = sj.Hand.create_initial_hand()

        # Set up two of the same card in a column
        next_hand = hand.flip(sj.Card.ONE, 0, 0)
        next_hand = next_hand.flip(sj.Card.ONE, 1, 0)

        # Replace a card to complete the column
        next_hand, _ = next_hand.replace(2, 0, sj.Card.ONE)

        # Check the column was cleared
        self.assertEqual(next_hand.get_card(0, 0), sj.Card.CLEARED)
        self.assertEqual(next_hand.get_card(1, 0), sj.Card.CLEARED)
        self.assertEqual(next_hand.get_card(2, 0), sj.Card.CLEARED)
        self.assertEqual(next_hand.cleared_card, sj.Card.ONE)

    def test_get_card_assertion(self):
        """Test that get_card raises assertions for invalid indices"""
        hand = sj.Hand.create_initial_hand()

        # Test invalid row
        with self.assertRaises(AssertionError):
            hand.get_card(-1, 0)

        with self.assertRaises(AssertionError):
            hand.get_card(sj.NUM_ROWS, 0)

        # Test invalid column
        with self.assertRaises(AssertionError):
            hand.get_card(0, -1)

        with self.assertRaises(AssertionError):
            hand.get_card(0, sj.NUM_COLUMNS)

    def test_property_face_down_indices(self):
        """Test the face_down_indices property"""
        hand = sj.Hand.create_initial_hand()

        # Initially all cards are face down
        self.assertEqual(len(hand.face_down_indices), sj.NUM_ROWS * sj.NUM_COLUMNS)

        # Flip some cards
        next_hand = hand.flip(sj.Card.ONE, 0, 0)
        next_hand = next_hand.flip(sj.Card.TWO, 1, 1)

        # Check face down indices
        face_down_indices = next_hand.face_down_indices
        self.assertEqual(len(face_down_indices), sj.NUM_ROWS * sj.NUM_COLUMNS - 2)

        # Check specific indices are not in face_down_indices
        self.assertNotIn((0, 0), face_down_indices)
        self.assertNotIn((1, 1), face_down_indices)

    def test_property_cleared_indices(self):
        """Test the cleared_indices property"""
        hand = sj.Hand.create_initial_hand()

        # Initially no cards are cleared
        self.assertEqual(len(hand.cleared_indices), 0)

        # Create a hand with cleared column by flipping same cards
        next_hand = hand.flip(sj.Card.ZERO, 0, 0)
        next_hand = next_hand.flip(sj.Card.ZERO, 1, 0)
        next_hand = next_hand.flip(sj.Card.ZERO, 2, 0)

        # Check cleared indices
        cleared_indices = next_hand.cleared_indices
        self.assertEqual(len(cleared_indices), 3)

        # Indices are now (row_idx, col_idx) rather than (col_idx, row_idx)
        self.assertIn((0, 0), cleared_indices)
        self.assertIn((1, 0), cleared_indices)
        self.assertIn((2, 0), cleared_indices)

    def test_property_non_cleared_indices(self):
        """Test the non_cleared_indices property"""
        hand = sj.Hand.create_initial_hand()

        # Initially all cards are non-cleared
        self.assertEqual(len(hand.non_cleared_indices), sj.NUM_ROWS * sj.NUM_COLUMNS)

        # Clear a column
        next_hand = hand.flip(sj.Card.ZERO, 0, 0)
        next_hand = next_hand.flip(sj.Card.ZERO, 1, 0)
        next_hand = next_hand.flip(sj.Card.ZERO, 2, 0)

        # Check non-cleared indices
        non_cleared_indices = next_hand.non_cleared_indices
        self.assertEqual(len(non_cleared_indices), sj.NUM_ROWS * sj.NUM_COLUMNS - 3)

        # Check specific indices are not in non_cleared_indices
        # Indices are now (row_idx, col_idx) rather than (col_idx, row_idx)
        self.assertNotIn((0, 0), non_cleared_indices)
        self.assertNotIn((1, 0), non_cleared_indices)
        self.assertNotIn((2, 0), non_cleared_indices)

    def test_compute_valid_place_from_discard_actions(self):
        """Test computing valid place from discard actions"""
        hand = sj.Hand.create_initial_hand()
        valid_actions = hand.compute_valid_place_from_discard_actions()

        # Initially all positions are valid
        self.assertEqual(len(valid_actions), sj.NUM_ROWS * sj.NUM_COLUMNS)

        # Clear a column
        next_hand = hand.flip(sj.Card.ZERO, 0, 0)
        next_hand = next_hand.flip(sj.Card.ZERO, 1, 0)
        next_hand = next_hand.flip(sj.Card.ZERO, 2, 0)

        valid_actions = next_hand.compute_valid_place_from_discard_actions()

        # Cleared positions should not be valid
        self.assertEqual(len(valid_actions), sj.NUM_ROWS * sj.NUM_COLUMNS - 3)

        # Check action types are correct
        for action in valid_actions:
            self.assertEqual(action.action_type, sj.SkyjoActionType.PLACE_FROM_DISCARD)

        # Verify no action exists for column 0 (which is the cleared column)
        for action in valid_actions:
            if action.row_idx == 0 and action.col_idx == 0:
                self.fail("Should not have action for cleared position (0,0)")
            if action.row_idx == 1 and action.col_idx == 0:
                self.fail("Should not have action for cleared position (1,0)")
            if action.row_idx == 2 and action.col_idx == 0:
                self.fail("Should not have action for cleared position (2,0)")

    def test_compute_valid_discard_and_flip_actions(self):
        """Test computing valid discard and flip actions"""
        hand = sj.Hand.create_initial_hand()
        valid_actions = hand.compute_valid_discard_and_flip_actions()

        # Initially all positions are valid for flipping
        self.assertEqual(len(valid_actions), sj.NUM_ROWS * sj.NUM_COLUMNS)

        # Flip a card
        next_hand = hand.flip(sj.Card.ONE, 0, 0)
        valid_actions = next_hand.compute_valid_discard_and_flip_actions()

        # Flipped card is no longer valid for flipping
        self.assertEqual(len(valid_actions), sj.NUM_ROWS * sj.NUM_COLUMNS - 1)

        # Check action types and positions
        for action in valid_actions:
            self.assertEqual(action.action_type, sj.SkyjoActionType.DISCARD_AND_FLIP)
            self.assertNotEqual(
                (action.row_idx, action.col_idx), (0, 0)
            )  # (0,0) is flipped

    def test_compute_valid_place_drawn_actions(self):
        """Test computing valid place drawn actions"""
        hand = sj.Hand.create_initial_hand()
        valid_actions = hand.compute_valid_place_drawn_actions()

        # Initially all positions are valid
        self.assertEqual(len(valid_actions), sj.NUM_ROWS * sj.NUM_COLUMNS)

        # Clear a column
        next_hand = hand.flip(sj.Card.ZERO, 0, 0)
        next_hand = next_hand.flip(sj.Card.ZERO, 1, 0)
        next_hand = next_hand.flip(sj.Card.ZERO, 2, 0)

        valid_actions = next_hand.compute_valid_place_drawn_actions()

        # Cleared positions should not be valid
        self.assertEqual(len(valid_actions), sj.NUM_ROWS * sj.NUM_COLUMNS - 3)

        self.assertTrue(
            all(
                action.action_type == sj.SkyjoActionType.PLACE_DRAWN
                for action in valid_actions
            )
        )

        # Verify no action exists for column 0 (which is the cleared column)
        for action in valid_actions:
            if action.row_idx == 0 and action.col_idx == 0:
                self.fail("Should not have action for cleared position (0,0)")
            if action.row_idx == 1 and action.col_idx == 0:
                self.fail("Should not have action for cleared position (1,0)")
            if action.row_idx == 2 and action.col_idx == 0:
                self.fail("Should not have action for cleared position (2,0)")

    def test_clearable_columns(self):
        """Test the clearable_columns static method"""
        hand = sj.Hand.create_initial_hand()

        # Create test data with a clearable column
        card_one_hots = hand._card_one_hots.copy()

        # Set up a column with all the same card (e.g., ONE)
        for row_idx in range(sj.NUM_ROWS):
            card_one_hots[0, row_idx] = sj.Card.ONE.one_hot_encoding()

        # Check clearable columns
        clearable_columns = sj.Hand.clearable_columns(card_one_hots)
        self.assertEqual(len(clearable_columns), 1)
        self.assertEqual(clearable_columns[0], 0)

        # Check with multiple clearable columns
        for row_idx in range(sj.NUM_ROWS):
            card_one_hots[1, row_idx] = sj.Card.TWO.one_hot_encoding()

        clearable_columns = sj.Hand.clearable_columns(card_one_hots)
        self.assertEqual(len(clearable_columns), 2)
        self.assertIn(0, clearable_columns)
        self.assertIn(1, clearable_columns)

    def test_visible_points(self):
        """Test calculating visible points"""
        # Create a hand with known cards
        hand = sj.Hand.create_initial_hand()

        # Flip some cards with known values
        hand = hand.flip(sj.Card.NEGATIVE_TWO, 0, 0)  # -2
        hand = hand.flip(sj.Card.ONE, 1, 0)  # 1
        hand = hand.flip(sj.Card.THREE, 2, 0)  # 3
        hand = hand.flip(sj.Card.FIVE, 0, 1)  # 5

        # Calculate visible points (should be -2 + 1 + 3 + 5 = 7)
        self.assertEqual(hand.visible_points, 7)

    def test_reveal_all_face_down_cards_with_clearing(self):
        """Test calculating total points with column clearing"""
        hand = sj.Hand.create_initial_hand()

        # First, flip all cards except for column 0
        for row_idx in range(sj.NUM_ROWS):
            for col_idx in range(1, sj.NUM_COLUMNS):
                # Assign different card values to each row so no clearing occurs
                hand = hand.flip(sj.Card(row_idx), row_idx, col_idx)

        # Verify we have 3 face-down cards left (all in column 0)
        self.assertEqual(len(hand.face_down_indices), 3)
        self.assertEqual(
            hand.face_down_indices[0][1], 0
        )  # First face-down card is in column 0
        self.assertEqual(
            hand.face_down_indices[1][1], 0
        )  # Second face-down card is in column 0
        self.assertEqual(
            hand.face_down_indices[2][1], 0
        )  # Third face-down card is in column 0

        # Calculate current visible points (sum of non-column-0 cards)
        expected_visible_points = sum(
            row_idx for row_idx in range(sj.NUM_ROWS) for _ in range(1, sj.NUM_COLUMNS)
        )
        self.assertEqual(hand.visible_points, expected_visible_points)

        # Create cards to be revealed - all THREE for column 0 to make it clear
        remaining_face_down = [
            sj.Card.THREE,  # will be at (0, 0)
            sj.Card.THREE,  # will be at (1, 0)
            sj.Card.THREE,  # will be at (2, 0)
        ]

        # Use reveal_all_face_down_cards to reveal all cards and calculate total
        revealed_hand = hand.reveal_all_face_down_cards(remaining_face_down)

        # Expected points:
        # - Initial points from visible cards outside column 0
        # - Column 0 should be cleared (all THREEs) so it contributes 0 points
        self.assertEqual(revealed_hand.visible_points, expected_visible_points)

        # Check that column 0 was cleared
        for row_idx in range(sj.NUM_ROWS):
            self.assertEqual(
                hand.get_card(row_idx, 0), sj.Card.FACE_DOWN
            )  # Still face down in original hand
            self.assertEqual(
                revealed_hand.get_card(row_idx, 0), sj.Card.CLEARED
            )  # Now revealed as CLEARED

    def test_numpy(self):
        """Test the numpy representation of a hand."""
        hand = sj.Hand.create_initial_hand()
        np_repr = hand.numpy()

        # Check shape and dtype
        self.assertEqual(np_repr.shape, sj.HAND_SHAPE)
        self.assertEqual(np_repr.dtype, np.int8)

        # Check initial state (all face down)
        face_down_one_hot = sj.Card.FACE_DOWN.one_hot_encoding()
        for r in range(sj.NUM_ROWS):
            for c in range(sj.NUM_COLUMNS):
                np.testing.assert_array_equal(np_repr[c, r, :], face_down_one_hot)

        # Flip a card and check again
        next_hand = hand.flip(sj.Card.FIVE, 1, 2)
        np_repr_flipped = next_hand.numpy()
        five_one_hot = sj.Card.FIVE.one_hot_encoding()
        np.testing.assert_array_equal(np_repr_flipped[2, 1, :], five_one_hot)
        # Check original numpy representation is unchanged
        np.testing.assert_array_equal(np_repr[2, 1, :], face_down_one_hot)


class TestImmutableState(unittest.TestCase):
    def test_start_round(self):
        """Test setting up a new round"""
        initial_state = sj.ImmutableState(
            num_players=2,
            player_scores=np.zeros(2, dtype=np.int16),
            remaining_card_counts=sj.CardCounts.create_initial_deck_counts(),
        )

        # Setup round with a specific top card
        top_card = sj.Card.ZERO
        state = initial_state.start_round(top_card)

        # Check round setup
        self.assertEqual(state.turn_count, 1)
        self.assertFalse(state.is_round_ending)
        self.assertIsNone(state.round_ending_player)
        self.assertIsNone(state.drawn_card)

        # Check discard pile
        self.assertEqual(state.discard_pile.top_card, top_card)
        self.assertEqual(
            state.discard_pile.discarded_card_counts.get_card_count(top_card), 1
        )
        self.assertEqual(
            state.discard_pile.discarded_card_counts.total_points,
            top_card.point_value,
        )

        # Check remaining cards
        self.assertEqual(
            state.remaining_card_counts.get_card_count(top_card),
            initial_state.remaining_card_counts.get_card_count(top_card) - 1,
        )
        self.assertEqual(
            state.remaining_card_counts.get_card_count(top_card),
            sj.CardCounts.create_initial_deck_counts().get_card_count(top_card) - 1,
        )

        # Check valid actions are initial flips
        self.assertTrue(
            all(
                action.action_type == sj.SkyjoActionType.INITIAL_FLIP
                for action in state.valid_actions
            )
        )

    def test_initial_flip(self):
        """Test initial flip action"""
        initial_state = sj.ImmutableState(
            num_players=2,
            player_scores=np.zeros(2, dtype=np.int16),
            remaining_card_counts=sj.CardCounts.create_initial_deck_counts(),
        )

        # Setup round and perform initial flip
        state = initial_state.start_round(sj.Card.ZERO)
        flipped_card = sj.Card.ONE
        state = state.initial_flip(flipped_card, 0, 0)

        # Check card was flipped
        self.assertEqual(state.hands[0].get_card(0, 0), flipped_card)

        # Check remaining cards updated
        self.assertEqual(
            state.remaining_card_counts.get_card_count(flipped_card),
            initial_state.remaining_card_counts.get_card_count(flipped_card) - 1,
        )

        # Check turn moved to next player
        self.assertEqual(state.curr_player, 1)

        # Check valid actions are still initial flips
        self.assertTrue(
            all(
                action.action_type == sj.SkyjoActionType.INITIAL_FLIP
                for action in state.valid_actions
            )
        )

    def test_initial_flip_start_round(self):
        """Test that round starts after all players have done initial flips"""
        initial_state = sj.ImmutableState(
            num_players=2,
            player_scores=np.zeros(2, dtype=np.int16),
            remaining_card_counts=sj.CardCounts.create_initial_deck_counts(),
        )

        # Setup round and do initial flips for both players
        state = initial_state.start_round(sj.Card.ZERO)

        state = state.initial_flip(sj.Card.ONE, 0, 0)
        state = state.initial_flip(sj.Card.TWO, 0, 0)
        state = state.initial_flip(sj.Card.ONE, 1, 0)
        state = state.initial_flip(sj.Card.TWO, 1, 0)

        # Check round has started
        self.assertTrue(
            all(
                action.action_type
                in [sj.SkyjoActionType.DRAW, sj.SkyjoActionType.PLACE_FROM_DISCARD]
                for action in state.valid_actions
            )
        )

        # Check player with highest points goes first
        self.assertEqual(state.curr_player, 1)  # Player 1 has higher points (2+2 > 1+1)

    def test_initial_flip_with_round_ending_player(self):
        """Test that new round starts with last round's ending player"""
        # Create state with a round_ending_player already set
        initial_state = sj.ImmutableState(
            num_players=2,
            player_scores=np.zeros(2, dtype=np.int16),
            remaining_card_counts=sj.CardCounts.create_initial_deck_counts(),
            round_ending_player=1,  # Player 1 ended the previous round
        )

        # Setup new round
        state = initial_state.start_round(sj.Card.ZERO)

        # Check that player 1 goes first in the new round
        self.assertEqual(state.curr_player, 1)

        # Check valid actions are initial flips for player 1
        self.assertTrue(
            all(
                action.action_type == sj.SkyjoActionType.INITIAL_FLIP
                for action in state.valid_actions
            )
        )

    def test_draw_card(self):
        """Test drawing a card"""
        # Create a state ready for drawing
        initial_state = sj.ImmutableState(
            num_players=2,
            player_scores=np.zeros(2, dtype=np.int16),
            remaining_card_counts=sj.CardCounts.create_initial_deck_counts(),
            hands=[sj.Hand.create_initial_hand() for _ in range(2)],
            discard_pile=sj.DiscardPile().discard(sj.Card.ZERO),
            valid_actions=[sj.SkyjoAction(sj.SkyjoActionType.DRAW)],
        )

        # Draw a card
        drawn_card = sj.Card.THREE
        state = initial_state.draw_card(drawn_card)

        # Check drawn card is set
        self.assertEqual(state.drawn_card, drawn_card)

        # Check remaining cards updated
        self.assertEqual(
            state.remaining_card_counts.get_card_count(drawn_card),
            initial_state.remaining_card_counts.get_card_count(drawn_card) - 1,
        )

        # Check valid actions are place_drawn and discard_and_flip
        action_types = [action.action_type for action in state.valid_actions]
        self.assertIn(sj.SkyjoActionType.PLACE_DRAWN, action_types)
        self.assertIn(sj.SkyjoActionType.DISCARD_AND_FLIP, action_types)

    def test_place_drawn(self):
        """Test placing a drawn card"""
        # Create a state with a drawn card
        hand = sj.Hand.create_initial_hand()
        hand = hand.flip(sj.Card.ONE, 0, 0)  # Flip one card so we have a known state

        initial_state = sj.ImmutableState(
            num_players=2,
            player_scores=np.zeros(2, dtype=np.int16),
            remaining_card_counts=sj.CardCounts.create_initial_deck_counts(),
            hands=[hand, sj.Hand.create_initial_hand()],
            discard_pile=sj.DiscardPile().discard(sj.Card.ZERO),
            drawn_card=sj.Card.THREE,
            valid_actions=[sj.SkyjoAction(sj.SkyjoActionType.PLACE_DRAWN, 1, 0)],
        )

        # Place drawn card on a face-down card (requires face_down_card)
        face_down_card = sj.Card.TWO
        state = initial_state.place_drawn(1, 0, face_down_card)

        # Check card was placed
        self.assertEqual(state.hands[0].get_card(1, 0), sj.Card.THREE)

        # Check face-down card was added to discard pile
        self.assertEqual(state.discard_pile.top_card, face_down_card)

        # Check remaining cards updated for face-down card
        self.assertEqual(
            state.remaining_card_counts.get_card_count(face_down_card),
            initial_state.remaining_card_counts.get_card_count(face_down_card) - 1,
        )

        # Check drawn card is cleared
        self.assertIsNone(state.drawn_card)

        # Check turn moved to next player
        self.assertEqual(state.curr_player, 1)

        # Check turn count incremented
        self.assertEqual(state.turn_count, initial_state.turn_count + 1)

    def test_place_drawn_on_visible_card(self):
        """Test placing a drawn card on a visible (already flipped) card"""
        # Create a hand with a visible card
        hand = sj.Hand.create_initial_hand()
        hand = hand.flip(sj.Card.ONE, 0, 0)

        initial_state = sj.ImmutableState(
            num_players=2,
            player_scores=np.zeros(2, dtype=np.int16),
            remaining_card_counts=sj.CardCounts.create_initial_deck_counts(),
            hands=[hand, sj.Hand.create_initial_hand()],
            discard_pile=sj.DiscardPile().discard(sj.Card.ZERO),
            drawn_card=sj.Card.THREE,
            valid_actions=[sj.SkyjoAction(sj.SkyjoActionType.PLACE_DRAWN, 0, 0)],
        )

        # Place drawn card on visible card
        state = initial_state.place_drawn(0, 0)

        # Check card was placed
        self.assertEqual(state.hands[0].get_card(0, 0), sj.Card.THREE)

        # Check replaced card was added to discard pile
        self.assertEqual(state.discard_pile.top_card, sj.Card.ONE)

        # Check remaining cards not updated for visible card
        self.assertEqual(
            state.remaining_card_counts.get_card_count(sj.Card.ONE),
            initial_state.remaining_card_counts.get_card_count(sj.Card.ONE),
        )

    def test_place_from_discard(self):
        """Test placing a card from the discard pile"""
        # Create a state ready for place_from_discard
        hand = sj.Hand.create_initial_hand()
        hand = hand.flip(sj.Card.ONE, 0, 0)  # Flip one card

        initial_state = sj.ImmutableState(
            num_players=2,
            player_scores=np.zeros(2, dtype=np.int16),
            remaining_card_counts=sj.CardCounts.create_initial_deck_counts(),
            hands=[hand, sj.Hand.create_initial_hand()],
            discard_pile=sj.DiscardPile().discard(sj.Card.THREE),
            valid_actions=[sj.SkyjoAction(sj.SkyjoActionType.PLACE_FROM_DISCARD, 1, 0)],
        )

        # Place from discard on a face-down card
        face_down_card = sj.Card.TWO
        state = initial_state.place_from_discard(1, 0, face_down_card)

        # Check card was placed
        self.assertEqual(state.hands[0].get_card(1, 0), sj.Card.THREE)

        # Check face-down card became new top of discard pile
        self.assertEqual(state.discard_pile.top_card, face_down_card)

        # Check remaining cards updated for face-down card
        self.assertEqual(
            state.remaining_card_counts.get_card_count(face_down_card),
            initial_state.remaining_card_counts.get_card_count(face_down_card) - 1,
        )

        # Check turn moved to next player
        self.assertEqual(state.curr_player, 1)

        # Check turn count incremented
        self.assertEqual(state.turn_count, initial_state.turn_count + 1)

    def test_place_from_discard_on_visible_card(self):
        """Test placing a card from the discard pile on a visible card"""
        # Create a hand with a visible card
        hand = sj.Hand.create_initial_hand()
        hand = hand.flip(sj.Card.ONE, 0, 0)

        initial_state = sj.ImmutableState(
            num_players=2,
            player_scores=np.zeros(2, dtype=np.int16),
            remaining_card_counts=sj.CardCounts.create_initial_deck_counts(),
            hands=[hand, sj.Hand.create_initial_hand()],
            discard_pile=sj.DiscardPile().discard(sj.Card.THREE),
            valid_actions=[sj.SkyjoAction(sj.SkyjoActionType.PLACE_FROM_DISCARD, 0, 0)],
        )

        # Place from discard on visible card
        state = initial_state.place_from_discard(0, 0)

        # Check card was placed
        self.assertEqual(state.hands[0].get_card(0, 0), sj.Card.THREE)

        # Check replaced card became new top of discard pile
        self.assertEqual(state.discard_pile.top_card, sj.Card.ONE)

        # Check remaining cards not updated for visible card
        self.assertEqual(
            state.remaining_card_counts.get_card_count(sj.Card.ONE),
            initial_state.remaining_card_counts.get_card_count(sj.Card.ONE),
        )

    def test_discard_and_flip(self):
        """Test discarding a drawn card and flipping a face-down card"""
        # Create a state with a drawn card
        hand = sj.Hand.create_initial_hand()

        initial_state = sj.ImmutableState(
            num_players=2,
            player_scores=np.zeros(2, dtype=np.int16),
            remaining_card_counts=sj.CardCounts.create_initial_deck_counts(),
            hands=[hand, sj.Hand.create_initial_hand()],
            discard_pile=sj.DiscardPile().discard(sj.Card.ZERO),
            drawn_card=sj.Card.THREE,
            valid_actions=[sj.SkyjoAction(sj.SkyjoActionType.DISCARD_AND_FLIP, 0, 0)],
        )

        # Discard drawn card and flip a face-down card
        flipped_card = sj.Card.TWO
        state = initial_state.discard_and_flip(flipped_card, 0, 0)

        # Check face-down card was flipped
        self.assertEqual(state.hands[0].get_card(0, 0), flipped_card)

        # Check drawn card was added to discard pile
        self.assertEqual(state.discard_pile.top_card, sj.Card.THREE)

        # Check remaining cards updated for flipped card
        self.assertEqual(
            state.remaining_card_counts.get_card_count(flipped_card),
            initial_state.remaining_card_counts.get_card_count(flipped_card) - 1,
        )

        # Check drawn card is cleared
        self.assertIsNone(state.drawn_card)

        # Check turn moved to next player
        self.assertEqual(state.curr_player, 1)

        # Check turn count incremented
        self.assertEqual(state.turn_count, initial_state.turn_count + 1)

    def test_end_round(self):
        """Test ending a round with player 0 having face-down cards and player 1 having all cards revealed"""
        # Set up specific scenario:
        # - Player 0 has 5 face-down cards
        # - Player 1 has all cards revealed (0 face-down)
        # - 3 of player 0's revealed cards will clear (same column)
        # - Player 1 will have score doubled since they have equal total score

        # Create hand for player 0 with 5 face-down cards
        hand0 = sj.Hand.create_initial_hand()

        # Define the card values and positions to flip
        # We'll flip 7 cards total, leaving 5 face-down
        p0_cards = [
            (sj.Card(v), v % sj.NUM_ROWS, v // sj.NUM_ROWS)
            for v in range(sj.NUM_COLUMNS * sj.NUM_ROWS - sj.NUM_ROWS - 1)
        ]

        # Flip the cards
        for card, row, col in p0_cards:
            hand0 = hand0.flip(card, row, col)

        # Verify we have 5 face-down cards
        self.assertEqual(len(hand0.face_down_indices), sj.NUM_ROWS + 1)

        # Create hand for player 1 with all cards revealed
        hand1 = sj.Hand.create_initial_hand()

        # Define card values to create a score around 14
        # (player 0 has 5+5+1+2+3+1+2 = 19, after revealing will be 10)
        p1_cards = [
            (sj.Card(v), v % sj.NUM_ROWS, v // sj.NUM_ROWS)
            for v in range(sj.NUM_COLUMNS * sj.NUM_ROWS)
        ]
        # Flip all cards for player 1
        for card, row, col in p1_cards:
            hand1 = hand1.flip(card, row, col)

        # Verify player 1 has no face-down cards
        self.assertEqual(len(hand1.face_down_indices), 0)

        # Create cards to be revealed for player 0 - 5 cards:
        # - The third FIVE in column 0 (will trigger column clearing)
        # - 4 other cards for remaining positions
        p0_cards_to_reveal = [sj.Card.ONE for _ in range(sj.NUM_ROWS + 1)]

        # Set up the game state
        initial_scores = np.ones(2, dtype=np.int16)
        initial_scores[0] = 19
        state = sj.ImmutableState(
            num_players=2,
            player_scores=initial_scores,
            remaining_card_counts=sj.CardCounts.create_initial_deck_counts(),
            hands=[hand0, hand1],
            discard_pile=sj.DiscardPile().discard(sj.Card.ZERO),
            turn_count=20,
            round_turn_counts=[],
            is_round_ending=True,
            round_ending_player=1,  # Player 1 ended the round
            curr_player=1,  # Last player before round ending player
            valid_actions=[sj.SkyjoAction(sj.SkyjoActionType.END_ROUND)],
        )

        # End the round
        result_state = state.end_round(p0_cards_to_reveal)

        # Check all cards were flipped
        for hand in result_state.hands:
            self.assertEqual(len(hand.face_down_indices), 0)

        # Check turn count was added to round_turn_counts
        self.assertEqual(len(result_state.round_turn_counts), 1)
        self.assertEqual(result_state.round_turn_counts[0], 20)
        self.assertEqual(
            result_state.player_scores[0], initial_scores[0] + hand0.visible_points + 1
        )
        self.assertEqual(
            result_state.player_scores[0],
            initial_scores[0] + result_state.hands[0].visible_points,
        )
        self.assertEqual(
            result_state.player_scores[1], initial_scores[1] + hand1.visible_points * 2
        )
        self.assertEqual(
            result_state.player_scores[1],
            initial_scores[1] + result_state.hands[1].visible_points * 2,
        )

    def test_compute_round_scores_assertion(self):
        """Test that compute_round_scores raises an assertion error when not all cards are revealed"""
        # Create a hand with face-down cards
        hand1 = sj.Hand.create_initial_hand()
        hand1 = hand1.flip(sj.Card.ONE, 0, 0)

        # Create another fully revealed hand
        hand2 = sj.Hand.create_initial_hand()
        for r in range(sj.NUM_ROWS):
            for c in range(sj.NUM_COLUMNS):
                hand2 = hand2.flip(sj.Card.TWO, r, c)

        hands = [hand1, hand2]

        # This should fail because hand1 still has face-down cards
        with self.assertRaises(AssertionError):
            sj.ImmutableState.compute_round_scores(hands, 0)

    def test_compute_round_scores_basic(self):
        """Test basic score computation case"""
        # Create fully revealed hands with known scores
        hand1 = sj.Hand.create_initial_hand()
        for r in range(sj.NUM_ROWS):
            hand1 = hand1.flip(sj.Card.ONE, r, 0)
        for v in range((sj.NUM_COLUMNS - 1) * sj.NUM_ROWS):
            hand1 = hand1.flip(sj.Card(v), v % sj.NUM_ROWS, v // sj.NUM_ROWS + 1)

        hand2 = sj.Hand.create_initial_hand()
        for v in range(sj.NUM_COLUMNS * sj.NUM_ROWS):
            hand2 = hand2.flip(sj.Card(v), v % sj.NUM_ROWS, v // sj.NUM_ROWS)

        hands = [hand1, hand2]
        self.assertTrue(hands[0].visible_points < hands[1].visible_points)
        self.assertTrue(len(hands[0].cleared_indices) == sj.NUM_ROWS)
        # Check that we get the expected array of scores
        # No doubling expected since player 0 has lower visible points than player 1
        scores = sj.ImmutableState.compute_round_scores(hands, 0)
        self.assertEqual(scores.shape, (2,))
        self.assertEqual(scores[0], hands[0].visible_points)
        self.assertEqual(scores[1], hands[1].visible_points)

        # Doubling expected since player 1 has higher board points than player 0
        scores = sj.ImmutableState.compute_round_scores(hands, 1)
        self.assertEqual(scores.shape, (2,))
        self.assertEqual(scores[0], hands[0].visible_points)
        self.assertEqual(scores[1], hands[1].visible_points * 2)

    def test_compute_round_scores_doubling_equal_scores(self):
        """Test score computation doubles on equal scores"""
        # Create fully revealed hands with known scores
        hand1 = sj.Hand.create_initial_hand()
        for v in range(sj.NUM_COLUMNS * sj.NUM_ROWS):
            hand1 = hand1.flip(sj.Card(v), v % sj.NUM_ROWS, v // sj.NUM_ROWS)

        hand2 = sj.Hand.create_initial_hand()
        for v in range(sj.NUM_COLUMNS * sj.NUM_ROWS):
            hand2 = hand2.flip(sj.Card(v), v % sj.NUM_ROWS, v // sj.NUM_ROWS)

        hands = [hand1, hand2]
        self.assertTrue(hands[0].visible_points == hands[1].visible_points)
        # Check that we get the expected array of scores
        # Doubling expected since player 0 has equal visible points as player 1
        scores = sj.ImmutableState.compute_round_scores(hands, 0)
        self.assertEqual(scores.shape, (2,))
        self.assertEqual(scores[0], hands[0].visible_points * 2)
        self.assertEqual(scores[1], hands[1].visible_points)

        # Doubling expected since player 1 has equal visible points as player 0
        scores = sj.ImmutableState.compute_round_scores(hands, 1)
        self.assertEqual(scores.shape, (2,))
        self.assertEqual(scores[0], hands[0].visible_points)
        self.assertEqual(scores[1], hands[1].visible_points * 2)

    def _setup_state_for_numpy_test(self) -> sj.ImmutableState:
        """Helper to set up a consistent state for numpy representation tests."""
        state = sj.ImmutableState(
            num_players=2,
            player_scores=np.array([10, 20], dtype=np.int16),
            remaining_card_counts=sj.CardCounts.create_initial_deck_counts(),
        )
        state = state.start_round(sj.Card.ZERO)  # Discard = 0
        state = state.initial_flip(sj.Card.ONE, 0, 0)  # P0 flips (0,0) -> 1
        state = state.initial_flip(sj.Card.TWO, 1, 1)  # P1 flips (1,1) -> 2
        state = state.initial_flip(sj.Card.THREE, 1, 0)  # P0 flips (1,0) -> 3
        state = state.initial_flip(sj.Card.FOUR, 0, 1)  # P1 flips (0,1) -> 4
        # Round starts, P1 is current player (4+2 > 1+3)
        self.assertEqual(state.curr_player, 1)
        return state

    def test_spatial_numpy(self):
        """Test the spatial_numpy representation of the game state."""
        state = self._setup_state_for_numpy_test()
        spatial_np = state.spatial_numpy()

        # Shape should be (NUM_PLAYERS * NUM_COLUMNS, NUM_ROWS, CARD_TYPES)
        expected_spatial_shape = (
            state.num_players * sj.NUM_COLUMNS,
            sj.NUM_ROWS,
            sj.CARD_TYPES,
        )
        self.assertEqual(spatial_np.shape, expected_spatial_shape)
        self.assertEqual(spatial_np.dtype, np.int8)

        # Player perspective: spatial_np should have P1's hand first, then P0's
        # P1's hand (state.hands[1]) columns 0-3
        # P0's hand (state.hands[0]) columns 4-7 (assuming 4 cols)
        p1_hand_np = state.hands[1].numpy()
        p0_hand_np = state.hands[0].numpy()

        np.testing.assert_array_equal(spatial_np[: sj.NUM_COLUMNS], p1_hand_np)
        np.testing.assert_array_equal(spatial_np[sj.NUM_COLUMNS :], p0_hand_np)

        # Check specific flipped cards for P1 (index 1) at (1,1) -> 2 and (0,1) -> 4
        np.testing.assert_array_equal(
            spatial_np[1, 1, :], sj.Card.TWO.one_hot_encoding()
        )  # col=1, row=1
        np.testing.assert_array_equal(
            spatial_np[1, 0, :], sj.Card.FOUR.one_hot_encoding()
        )  # col=1, row=0

        # Check specific flipped cards for P0 (index 0) at (0,0) -> 1 and (1,0) -> 3
        np.testing.assert_array_equal(
            spatial_np[sj.NUM_COLUMNS + 0, 0, :], sj.Card.ONE.one_hot_encoding()
        )  # P0 hand (offset), col=0, row=0
        np.testing.assert_array_equal(
            spatial_np[sj.NUM_COLUMNS + 0, 1, :], sj.Card.THREE.one_hot_encoding()
        )  # P0 hand (offset), col=0, row=1

    def test_non_spatial_numpy(self):
        """Test the non_spatial_numpy representation of the game state."""
        state = self._setup_state_for_numpy_test()
        non_spatial_np = state.non_spatial_numpy()

        self.assertEqual(non_spatial_np.dtype, np.int16)  # Should be int16

        # Expected size: scores + remaining + discard_counts + top_card + drawn_card + is_ending
        expected_non_spatial_size = (
            state.num_players
            + sj.CARD_TYPES
            + sj.CARD_TYPES
            + sj.CARD_TYPES
            + sj.CARD_TYPES
            + 1
        )
        self.assertEqual(non_spatial_np.shape[0], expected_non_spatial_size)

        # Check player scores (rolled, P1 first)
        self.assertEqual(non_spatial_np[0], 20)  # P1 score
        self.assertEqual(non_spatial_np[1], 10)  # P0 score

        # Check remaining card counts (e.g., ZERO count should be 14)
        remaining_offset = state.num_players
        self.assertEqual(
            non_spatial_np[remaining_offset + sj.Card.ZERO.one_hot_encoding_index], 14
        )  # 15 initial - 1 discard
        self.assertEqual(
            non_spatial_np[remaining_offset + sj.Card.ONE.one_hot_encoding_index], 9
        )  # 10 initial - 1 flip
        self.assertEqual(
            non_spatial_np[remaining_offset + sj.Card.TWO.one_hot_encoding_index], 9
        )  # 10 initial - 1 flip
        self.assertEqual(
            non_spatial_np[remaining_offset + sj.Card.THREE.one_hot_encoding_index], 9
        )  # 10 initial - 1 flip
        self.assertEqual(
            non_spatial_np[remaining_offset + sj.Card.FOUR.one_hot_encoding_index], 9
        )  # 10 initial - 1 flip

        # Check discard pile counts (only ZERO should be 1)
        discard_offset = remaining_offset + sj.CARD_TYPES
        self.assertEqual(
            non_spatial_np[discard_offset + sj.Card.ZERO.one_hot_encoding_index], 1
        )
        self.assertEqual(
            np.sum(non_spatial_np[discard_offset : discard_offset + sj.CARD_TYPES]), 1
        )

        # Check top card one-hot (should be ZERO)
        top_card_offset = discard_offset + sj.CARD_TYPES
        expected_top_card_one_hot = sj.Card.ZERO.one_hot_encoding().astype(np.int16)
        np.testing.assert_array_equal(
            non_spatial_np[top_card_offset : top_card_offset + sj.CARD_TYPES],
            expected_top_card_one_hot,
        )

        # Check drawn card (should be all zeros as it's None)
        drawn_card_offset = top_card_offset + sj.CARD_TYPES
        self.assertEqual(
            np.sum(
                non_spatial_np[drawn_card_offset : drawn_card_offset + sj.CARD_TYPES]
            ),
            0,
        )

        # Check round ending flag (should be 0)
        self.assertEqual(non_spatial_np[-1], 0)

    # TODO: review tests for numpy representations
    # TODO: Add tests for round end.
    # Specifically, making sure round end detection and new start round work as intended.


if __name__ == "__main__":
    unittest.main()
