#include <stdio.h>
#include <stdlib.h>

#include "cards.h"
#include "hand.h"
#include "test.h"

TESTS_BEGIN()

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

TESTS_END()
