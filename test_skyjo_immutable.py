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

    def test_visible_points(self):
        hand_size = sj.NUM_ROWS * sj.NUM_COLUMNS
        flipped = np.zeros(hand_size)
        cleared = np.zeros(hand_size)
        hand = sj.Hand(cards=np.ones(hand_size), is_flipped=flipped, is_cleared=cleared)
        self.assertEqual(hand.visible_points(), 0)
        hand = hand.flip(0, 0)
        hand = hand.flip(1, 0)
        self.assertEqual(hand.visible_points(), 2)
        # clears column
        for r in range(2, sj.NUM_ROWS):
            hand = hand.flip(r, 0)
        self.assertEqual(hand.visible_points(), 0)

    def test_total_points_basic(self):
        hand_size = sj.NUM_ROWS * sj.NUM_COLUMNS
        flipped = np.zeros(hand_size)
        cleared = np.zeros(hand_size)
        cards = np.array(range(hand_size))
        self.assertEqual(
            sj.Hand(cards=cards, is_flipped=flipped, is_cleared=cleared).total_points(),
            sum(cards),
        )
        flipped = np.ones(hand_size)
        self.assertEqual(
            sj.Hand(cards=cards, is_flipped=flipped, is_cleared=cleared).total_points(),
            sum(cards),
        )

    def test_total_points_uncleared(self):
        hand_size = sj.NUM_ROWS * sj.NUM_COLUMNS
        flipped = np.zeros(hand_size)
        cleared = np.zeros(hand_size)

        hand = sj.Hand(cards=np.ones(hand_size), is_flipped=flipped, is_cleared=cleared)
        # everthing would be cleared when revealed
        self.assertEqual(hand.total_points(), 0)

    def test_valid_place_drawn_actions(self):
        hand_size = sj.NUM_ROWS * sj.NUM_COLUMNS
        flipped = np.zeros(hand_size)
        cleared = np.zeros(hand_size)

        hand = sj.Hand(cards=np.ones(hand_size), is_flipped=flipped, is_cleared=cleared)
        self.assertEqual(len(hand.valid_place_drawn_actions), hand_size)

        # clear first column
        for r in range(sj.NUM_ROWS):
            hand = hand.flip(r, 0)
        self.assertEqual(
            len(hand.valid_place_drawn_actions), hand_size - hand.is_cleared.sum()
        )

    def test_valid_place_from_discard_actions(self):
        hand_size = sj.NUM_ROWS * sj.NUM_COLUMNS
        flipped = np.zeros(hand_size)
        cleared = np.zeros(hand_size)

        hand = sj.Hand(cards=np.ones(hand_size), is_flipped=flipped, is_cleared=cleared)
        self.assertEqual(len(hand.valid_place_from_discard_actions), hand_size)

        # clear first column
        for r in range(sj.NUM_ROWS):
            hand = hand.flip(r, 0)
        self.assertEqual(
            len(hand.valid_place_from_discard_actions),
            hand_size - hand.is_cleared.sum(),
        )

    def test_valid_discard_and_flip_actions(self):
        hand_size = sj.NUM_ROWS * sj.NUM_COLUMNS
        flipped = np.zeros(hand_size)
        cleared = np.zeros(hand_size)

        hand = sj.Hand(cards=np.ones(hand_size), is_flipped=flipped, is_cleared=cleared)
        self.assertEqual(
            len(hand.valid_discard_and_flip_actions), len(hand.face_down_indices)
        )

        # clear first column
        for r in range(sj.NUM_ROWS):
            hand = hand.flip(r, 0)
        self.assertEqual(
            len(hand.valid_discard_and_flip_actions),
            len(hand.face_down_indices),
        )

        # flip one more card
        hand = hand.flip(0, 1)
        self.assertEqual(
            len(hand.valid_discard_and_flip_actions),
            len(hand.face_down_indices),
        )


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

    def test_round_end(self):
        curr_state = sj.ImmutableSkyjoState(
            num_players=2,
            player_scores=np.zeros(2),
            deck=sj.Deck(_cards=sj.DECK),
            round_ending_player=0,  # enforce first player starts
        ).setup_round()

        while (
            len(curr_state.hands[(curr_state.curr_player + 1) % 2].face_down_indices)
            >= 1
        ):
            curr_state = curr_state.take_action(
                sj.SkyjoAction(
                    action_type=sj.SkyjoActionType.PLACE_FROM_DISCARD,
                    row_idx=curr_state.hands[curr_state.curr_player].face_down_indices[
                        0
                    ][0],
                    col_idx=curr_state.hands[curr_state.curr_player].face_down_indices[
                        0
                    ][1],
                )
            )

        # check round ending attributes set correctly
        self.assertEqual(curr_state.is_round_ending, True)
        self.assertEqual(curr_state.round_ending_player, 0)

        end_round_state = curr_state.take_action(
            sj.SkyjoAction(
                action_type=sj.SkyjoActionType.PLACE_FROM_DISCARD,
                row_idx=curr_state.hands[curr_state.curr_player].face_down_indices[0][
                    0
                ],
                col_idx=curr_state.hands[curr_state.curr_player].face_down_indices[0][
                    1
                ],
            )
        )
        # check round ending attributes correctly reset after finishing round
        self.assertEqual(end_round_state.is_round_ending, False)
        self.assertEqual(end_round_state.round_ending_player is None, True)

    def test_compute_round_scores_basic(self):
        hand_size = sj.NUM_COLUMNS * sj.NUM_ROWS
        hands = [
            sj.Hand(
                cards=np.zeros(hand_size),
                is_flipped=np.ones(hand_size),
                is_cleared=np.zeros(hand_size),
            ),
            sj.Hand(
                cards=np.ones(hand_size),
                is_flipped=np.ones(hand_size),
                is_cleared=np.zeros(hand_size),
            ),
        ]
        round_scores = sj.ImmutableSkyjoState.compute_round_scores(hands, 0)
        npt.assert_allclose(round_scores, np.array([0, hand_size]))

        # test that score gets doubled for round ender if not lowest score
        round_scores = sj.ImmutableSkyjoState.compute_round_scores(hands, 1)
        npt.assert_allclose(round_scores, np.array([0, hand_size * 2]))

    def test_compute_round_scores_tie(self):
        hand_size = sj.NUM_COLUMNS * sj.NUM_ROWS
        hands = [
            sj.Hand(
                cards=np.ones(hand_size),
                is_flipped=np.ones(hand_size),
                is_cleared=np.zeros(hand_size),
            ),
            sj.Hand(
                cards=np.ones(hand_size),
                is_flipped=np.ones(hand_size),
                is_cleared=np.zeros(hand_size),
            ),
        ]
        round_scores = sj.ImmutableSkyjoState.compute_round_scores(hands, 0)
        npt.assert_allclose(round_scores, np.array([hand_size * 2, hand_size]))

        round_scores = sj.ImmutableSkyjoState.compute_round_scores(hands, 1)
        npt.assert_allclose(round_scores, np.array([hand_size, hand_size * 2]))


if __name__ == "__main__":
    unittest.main()
