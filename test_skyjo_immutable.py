import unittest

import numpy as np
import numpy.testing as npt
import skyjo_immutable as sj


class TestHand(unittest.TestCase):
    def test_visible_indices_empty(self):
        hand = sj.Hand()
        self.assertListEqual(hand.visible_indices, [])

    def test_visible_indices_face_down(self):
        flipped = np.zeros(12)
        flipped[10] = 1
        flipped[4] = 1
        hand = sj.Hand(is_flipped=flipped)
        self.assertListEqual(hand.visible_indices, [(1, 0), (2, 2)])

    def test_visible_indices_cleared(self):
        cleared = np.zeros(12)
        cleared[0] = 1
        cleared[1] = 1
        cleared[2] = 1
        flipped = np.zeros(12)
        flipped[0] = 1
        flipped[1] = 1
        flipped[2] = 1
        hand = sj.Hand(is_cleared=cleared, is_flipped=flipped)
        self.assertListEqual(hand.visible_indices, [])

    def test_face_down_indices_empty(self):
        hand = sj.Hand(is_flipped=np.ones(12))
        self.assertListEqual(hand.face_down_indices, [])

    def test_face_down_indices_flipped(self):
        flipped = np.ones(12)
        flipped[10] = 0
        hand = sj.Hand(is_flipped=flipped)
        self.assertListEqual(hand.face_down_indices, [(2, 2)])

    def test_face_down_indices_flipped_cleared(self):
        flipped = np.ones(12)
        flipped[10] = 0
        flipped[11] = 0
        flipped[9] = 0
        cleared = np.zeros(12)
        cleared[10] = 1
        cleared[11] = 1
        cleared[9] = 1
        hand = sj.Hand(is_cleared=cleared, is_flipped=flipped)
        self.assertListEqual(hand.face_down_indices, [])


class TestDeck(unittest.TestCase):
    def test_draw_top_card(self):
        init_deck = sj.Deck(sj.DECK)
        top_card, next_deck = init_deck.draw_top_card()
        self.assertEqual(top_card, init_deck._cards[0])
        self.assertEqual(len(next_deck._cards), len(init_deck._cards) - 1)

    def test_deal_hands(self):
        p0_expected_hand, p1_expected_hand = (
            [0 for _ in range(sj.NUM_COLUMNS * sj.NUM_ROWS)],
            [1 for _ in range(sj.NUM_COLUMNS * sj.NUM_ROWS)],
        )
        init_deck = sj.Deck(_cards=(p0_expected_hand + p1_expected_hand + [10]))
        hands, next_deck = init_deck.deal_hands(2)
        npt.assert_allclose(hands[0].cards, np.array(p0_expected_hand))
        npt.assert_allclose(hands[1].cards, np.array(p1_expected_hand))
        self.assertEqual(len(next_deck._cards), 1)
        self.assertEqual(next_deck._cards[0], 10)


class TestDiscardPile(unittest.TestCase):
    def test_discard(self):
        init_discard = sj.DiscardPile()
        new_discard = init_discard.discard(10)
        self.assertEqual(new_discard.top_card, 10)
        expected_counts = np.zeros(sj.NUM_CARD_TYPES)
        expected_counts[10 + 2] = 1
        npt.assert_allclose(new_discard.discarded_card_counts, expected_counts)

    def test_replace_top_card(self):
        init_discard = sj.DiscardPile()
        init_discard = init_discard.discard(10)
        new_discard = init_discard.replace_top_card(9)
        self.assertEqual(new_discard.top_card, 9)
        expected_counts = np.zeros(sj.NUM_CARD_TYPES)
        expected_counts[9 + 2] = 1
        npt.assert_allclose(new_discard.discarded_card_counts, expected_counts)

# TODO: Add tests for end of round and end of game cases
class TestImmutableSkyjoState(unittest.TestCase):
    def test_setup_round(self):
        init_state = sj.ImmutableSkyjoState(
            num_players=2, player_scores=np.zeros(2), deck=sj.Deck(_cards=sj.DECK)
        )
        setup_state = init_state.setup_round()

        # verify starting player is player with highest shown value and only two cards flipped
        for i in range(setup_state.num_players):
            self.assertEqual(len(setup_state.hands[i].visible_indices), 2)
            self.assertTrue(
                setup_state.hands[setup_state.curr_player].visible_points()
                >= setup_state.hands[i].visible_points()
            )

        # verify discard pile is not empty
        self.assertEqual(setup_state.discard_pile.discarded_card_counts.sum(), 1)
        self.assertTrue(setup_state.discard_pile.top_card is not None)

        # verify valid_actions contains correct actions
        self.assertEqual(len(setup_state.valid_actions), 13)

    def test_draw(self):
        setup_state = sj.ImmutableSkyjoState(
            num_players=2, player_scores=np.zeros(2), deck=sj.Deck(_cards=sj.DECK)
        ).setup_round()
        drawn_state = setup_state.take_action(
            sj.SkyjoAction(action_type=sj.SkyjoActionType.DRAW)
        )
        self.assertTrue(drawn_state.drawn_card is not None)

    def test_discard_and_flip(self):
        setup_state = sj.ImmutableSkyjoState(
            num_players=2, player_scores=np.zeros(2), deck=sj.Deck(_cards=sj.DECK)
        ).setup_round()
        drawn_state = setup_state.take_action(
            sj.SkyjoAction(action_type=sj.SkyjoActionType.DRAW)
        )
        face_down_idxs = drawn_state.hands[drawn_state.curr_player].face_down_indices
        discard_state = drawn_state.take_action(
            sj.SkyjoAction(
                sj.SkyjoActionType.DISCARD_AND_FLIP,
                face_down_idxs[0][0],
                face_down_idxs[0][1],
            )
        )
        self.assertEqual(
            discard_state.curr_player,
            (drawn_state.curr_player + 1) % drawn_state.num_players,
        )
        self.assertEqual(
            discard_state.hands[drawn_state.curr_player].is_flipped[
                sj.Hand.flat_index(face_down_idxs[0][0], face_down_idxs[0][1])
            ],
            1,
        )
        self.assertEqual(
            len(discard_state.hands[drawn_state.curr_player].visible_indices),
            len(drawn_state.hands[drawn_state.curr_player].visible_indices) + 1,
        )
        self.assertEqual(discard_state.discard_pile.top_card, drawn_state.drawn_card)

    def test_place_from_discard(self):
        setup_state = sj.ImmutableSkyjoState(
            num_players=2, player_scores=np.zeros(2), deck=sj.Deck(_cards=sj.DECK)
        ).setup_round()
        face_down_idxs = setup_state.hands[setup_state.curr_player].face_down_indices

        replace_state = setup_state.take_action(
            sj.SkyjoAction(
                sj.SkyjoActionType.PLACE_FROM_DISCARD,
                face_down_idxs[0][0],
                face_down_idxs[0][1],
            )
        )
        (
            self.assertEqual(
                replace_state.hands[setup_state.curr_player].cards[
                    sj.Hand.flat_index(face_down_idxs[0][0], face_down_idxs[0][1])
                ],
                setup_state.discard_pile.top_card,
            ),
        )

        self.assertEqual(
            replace_state.discard_pile.top_card,
            setup_state.hands[setup_state.curr_player].cards[
                sj.Hand.flat_index(face_down_idxs[0][0], face_down_idxs[0][1])
            ],
        )

        self.assertEqual(
            replace_state.discard_pile.discarded_card_counts[
                setup_state.discard_pile.top_card + 2
            ],
            0,
        )
        self.assertEqual(
            replace_state.discard_pile.discarded_card_counts[
                replace_state.discard_pile.top_card + 2
            ],
            1,
        )


if __name__ == "__main__":
    unittest.main()
