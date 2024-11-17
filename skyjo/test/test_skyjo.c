#include <stdio.h>
#include <stdlib.h>

#include "cards.h"
#include "hand.h"
#include "test.h"

TESTS_BEGIN()

// MARK: Cards

TEST(cards_restore)
{
    cards_t cards;
    cards_restore(&cards);
    ASSERT(cards.discard_index == 0);
    ASSERT(cards.draw_index == 1);
    ASSERT(cards.buffer[0] == -2);
    ASSERT(cards.buffer[1] == -2);
    ASSERT(cards.buffer[2] == -2);
    ASSERT(cards.buffer[3] == -2);
    ASSERT(cards.buffer[4] == -2);
    ASSERT(cards.buffer[DECK_SIZE - 1] == 12);
}

TEST(cards_shuffle)
{
    srand(0);

    cards_t cards;
    cards_restore(&cards);
    cards_shuffle(&cards);
    ASSERT(cards.discard_index == 0);
    ASSERT(cards.draw_index == 1);
    ASSERT(cards.buffer[0] == 1);
    ASSERT(cards.buffer[1] == -2);
    ASSERT(cards.buffer[2] == 3);
    ASSERT(cards.buffer[3] == 4);
    ASSERT(cards.buffer[4] == -1);
    ASSERT(cards.buffer[DECK_SIZE - 1] == 2);
}

TEST(cards_shuffle_restore)
{
    cards_t cards;
    cards_restore(&cards);
    cards_shuffle(&cards);
    cards_restore(&cards);
    ASSERT(cards.discard_index == 0);
    ASSERT(cards.draw_index == 1);
    ASSERT(cards.buffer[0] == -2);
    ASSERT(cards.buffer[1] == -2);
    ASSERT(cards.buffer[2] == -2);
    ASSERT(cards.buffer[3] == -2);
    ASSERT(cards.buffer[4] == -2);
    ASSERT(cards.buffer[DECK_SIZE - 1] == 12);
}

TEST(cards_deal)
{
    cards_t cards;
    cards_restore(&cards);
    ASSERT(cards_deal(&cards) == -2);
    ASSERT(cards.discard_index == 1);
    ASSERT(cards.draw_index == 2);
}

TEST(cards_get_last_discard)
{
    cards_t cards;
    cards_restore(&cards);
    ASSERT(cards_get_last_discard(&cards) == -2);
}

TEST(cards_get_next_draw)
{
    cards_t cards;
    cards_restore(&cards);
    ASSERT(cards_get_next_draw(&cards) == -2);
}

TEST(cards_draw)
{
    cards_t cards;
    cards_restore(&cards);
    cards_shuffle(&cards);
    card_t next_draw = cards_get_next_draw(&cards);
    card_t draw = cards_draw(&cards);
    ASSERT(next_draw == draw);
    ASSERT(cards.draw_index == 2);
}

TEST(cards_discard)
{
    cards_t cards;
    cards_restore(&cards);
    cards_shuffle(&cards);
    card_t draw = cards_draw(&cards);
    cards_discard(&cards, draw);
    ASSERT(cards_get_last_discard(&cards) == draw);
    ASSERT(cards.discard_index == 1);
    ASSERT(cards.draw_index == 2);
}

TEST(cards_replace_discard)
{
    cards_t cards;
    cards_restore(&cards);
    cards_shuffle(&cards);
    card_t last_discard = cards_get_last_discard(&cards);
    card_t draw = cards_draw(&cards);
    card_t replaced_discard = cards_replace_discard(&cards, draw);
    ASSERT(replaced_discard == last_discard);
    ASSERT(cards_get_last_discard(&cards) == draw);
    ASSERT(cards.discard_index == 0);
    ASSERT(cards.draw_index == 2);
}

// MARK: Hand

TEST(finger_restore)
{
    finger_t finger;
    finger_restore(&finger);
    ASSERT(finger_matches(finger, C));
}

TEST(finger_deal)
{
    finger_t finger;
    ASSERT(finger_deal(&finger, 12) == 12);
    ASSERT(finger_matches(finger, 12 H));
}

TEST(finger_reveal)
{
    finger_t finger;
    finger_deal(&finger, 12);
    finger_reveal(&finger);
    ASSERT(finger_matches(finger, 12));
}

TEST(finger_replace_hidden)
{
    finger_t finger;
    finger_deal(&finger, 12);
    ASSERT(finger_replace(&finger, -2) == 12);
    ASSERT(finger_matches(finger, -2));
}

TEST(finger_replace_revealed)
{
    finger_t finger;
    finger_deal(&finger, 12);
    finger_reveal(&finger);
    ASSERT(finger_replace(&finger, -2) == 12);
    ASSERT(finger_matches(finger, -2));
}

TEST(hand_restore)
{
    hand_t hand;
    hand_restore(&hand);
    ASSERT(hand_matches(hand,
                        C, C, C, C,
                        C, C, C, C,
                        C, C, C, C));
}

TEST(hand_try_clear_single)
{
    hand_t hand;
    hand_restore(&hand);
    hand_assign(hand,
                12, C, C, C,
                12, C, C, C,
                12, C, C, C);
    ASSERT(hand_try_clear(&hand, 0));
    ASSERT(hand_matches(hand,
                        C, C, C, C,
                        C, C, C, C,
                        C, C, C, C));
}

TEST(hand_try_clear_hidden)
{
    hand_t hand;
    hand_restore(&hand);
    hand_assign(hand,
                12 H, C, C, C,
                12 H, C, C, C,
                12 H, C, C, C);
    ASSERT(!hand_try_clear(&hand, 0));
    ASSERT(hand_matches(hand,
                        12 H, C, C, C,
                        12 H, C, C, C,
                        12 H, C, C, C));
}

TEST(hand_try_clear_hidden_one)
{
    hand_t hand;
    hand_restore(&hand);
    hand_assign(hand,
                12, C, C, C,
                12, C, C, C,
                12 H, C, C, C);
    ASSERT(!hand_try_clear(&hand, 0));
    ASSERT(hand_matches(hand,
                        12, C, C, C,
                        12, C, C, C,
                        12 H, C, C, C));
}

TEST(hand_try_clear_hidden_two)
{
    hand_t hand;
    hand_restore(&hand);
    hand_assign(hand,
                12, C, C, C,
                12 H, C, C, C,
                12 H, C, C, C);
    ASSERT(!hand_try_clear(&hand, 0));
    ASSERT(hand_matches(hand,
                        12, C, C, C,
                        12 H, C, C, C,
                        12 H, C, C, C));
}

TEST(hand_try_clear_double_right)
{
    hand_t hand;
    hand_restore(&hand);
    hand_assign(hand,
                12, 11, C, C,
                12, 11, C, C,
                12, 11, C, C);
    ASSERT(hand_try_clear(&hand, 1));
    ASSERT(hand_matches(hand,
                        12, C, C, C,
                        12, C, C, C,
                        12, C, C, C));
}

TEST(hand_try_clear_double_left)
{
    hand_t hand;
    hand_restore(&hand);
    hand_assign(hand,
                12, 11, C, C,
                12, 11, C, C,
                12, 11, C, C);
    ASSERT(hand_try_clear(&hand, 0));
    ASSERT(hand_matches(hand,
                        11, C, C, C,
                        11, C, C, C,
                        11, C, C, C));
}

TEST(hand_try_clear_triple_right)
{
    hand_t hand;
    hand_restore(&hand);
    hand_assign(hand,
                12, 11, 10, C,
                12, 11, 10, C,
                12, 11, 10, C);
    ASSERT(hand_try_clear(&hand, 2));
    ASSERT(hand_matches(hand,
                        12, 11, C, C,
                        12, 11, C, C,
                        12, 11, C, C));
}

TEST(hand_try_clear_triple_middle)
{
    hand_t hand;
    hand_restore(&hand);
    hand_assign(hand,
                12, 11, 10, C,
                12, 11, 10, C,
                12, 11, 10, C);
    ASSERT(hand_try_clear(&hand, 1));
    ASSERT(hand_matches(hand,
                        12, 10, C, C,
                        12, 10, C, C,
                        12, 10, C, C));
}

TEST(hand_try_clear_triple_right)
{
    hand_t hand;
    hand_restore(&hand);
    hand_assign(hand,
                12, 11, 10, C,
                12, 11, 10, C,
                12, 11, 10, C);
    ASSERT(hand_try_clear(&hand, 0));
    ASSERT(hand_matches(hand,
                        11, 10, C, C,
                        11, 10, C, C,
                        11, 10, C, C));
}

TEST(hand_try_clear_four_left)
{
    hand_t hand;
    hand_restore(&hand);
    hand_assign(hand,
                12, 11, 10, 9,
                12, 11, 10, 9,
                12, 11, 10, 9);
    ASSERT(hand_try_clear(&hand, 0));
    ASSERT(hand_matches(hand,
                        11, 10, 9, C,
                        11, 10, 9, C,
                        11, 10, 9, C));
}

TEST(hand_try_clear_four_left_middle)
{
    hand_t hand;
    hand_restore(&hand);
    hand_assign(hand,
                12, 11, 10, 9,
                12, 11, 10, 9,
                12, 11, 10, 9);
    ASSERT(hand_try_clear(&hand, 1));
    ASSERT(hand_matches(hand,
                        12, 10, 9, C,
                        12, 10, 9, C,
                        12, 10, 9, C));
}

TEST(hand_try_clear_four_right_middle)
{
    hand_t hand;
    hand_restore(&hand);
    hand_assign(hand,
                12, 11, 10, 9,
                12, 11, 10, 9,
                12, 11, 10, 9);
    ASSERT(hand_try_clear(&hand, 2));
    ASSERT(hand_matches(hand,
                        12, 11, 9, C,
                        12, 11, 9, C,
                        12, 11, 9, C));
}

TEST(hand_try_clear_four_right)
{
    hand_t hand;
    hand_restore(&hand);
    hand_assign(hand,
                12, 11, 10, 9,
                12, 11, 10, 9,
                12, 11, 10, 9);
    ASSERT(hand_try_clear(&hand, 3));
    ASSERT(hand_matches(hand,
                        12, 11, 10, C,
                        12, 11, 10, C,
                        12, 11, 10, C));
}

TEST(hand_try_clear_different_first)
{
    hand_t hand;
    hand_restore(&hand);
    hand_assign(hand,
                11, C, C, C,
                12, C, C, C,
                12, C, C, C);
    ASSERT(!hand_try_clear(&hand, 0));
    ASSERT(hand_matches(hand,
                        11, C, C, C,
                        12, C, C, C,
                        12, C, C, C));
}

TEST(hand_try_clear_different_second)
{
    hand_t hand;
    hand_restore(&hand);
    hand_assign(hand,
                12, C, C, C,
                11, C, C, C,
                12, C, C, C);
    ASSERT(!hand_try_clear(&hand, 0));
    ASSERT(hand_matches(hand,
                        12, C, C, C,
                        11, C, C, C,
                        12, C, C, C));
}

TEST(hand_try_clear_different_third)
{
    hand_t hand;
    hand_restore(&hand);
    hand_assign(hand,
                12, C, C, C,
                12, C, C, C,
                11, C, C, C);
    ASSERT(!hand_try_clear(&hand, 0));
    ASSERT(hand_matches(hand,
                        12, C, C, C,
                        12, C, C, C,
                        11, C, C, C));
}

TESTS_END()
