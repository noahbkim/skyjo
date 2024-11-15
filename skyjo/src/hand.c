#include "hand.h"

#include <assert.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

void finger_restore(finger_t *finger)
{
    memset(finger, 0, sizeof(finger_t));
}

inline card_t finger_deal(finger_t *finger, card_t card)
{
    finger->state = CARD_HIDDEN;
    finger->card = card;
    return card;
}

inline card_t finger_reveal(finger_t *finger)
{
    assert(finger->state == CARD_HIDDEN);

    finger->state = CARD_REVEALED;
    return finger->card;
}

inline card_t finger_replace(finger_t *finger, card_t card)
{
    assert(finger->state == CARD_HIDDEN || finger->state == CARD_REVEALED);

    card_t replaced_card = finger->card;
    finger->state = CARD_REVEALED;
    finger->card = card;
    return replaced_card;
}

inline bool finger_equal(finger_t a, finger_t b)
{
    return memcmp(&a, &b, sizeof(finger_t)) == 0;
}

inline void hand_restore(hand_t *hand)
{
    memset(hand, 0, sizeof(hand_t));
}

bool hand_try_clear(hand_t *hand, handsize_t column)
{
    assert(column < hand->columns);

    // Check if the first finger in the column is revealed.
    finger_t *finger = hand->fingers + column * HAND_ROWS;
    if (finger->state != CARD_REVEALED)
    {
        return false;
    }

    // Check if every subsequent finger has the same card and is revealed.
    finger_t *cursor = finger;
    for (handsize_t r = 1; r < HAND_ROWS; ++r, ++cursor)
    {
        if (cursor->state != CARD_REVEALED || cursor->card != finger->card)
        {
            return false;
        }
    }

    // Shift all cards after the cleared column over (by one column).
    handsize_t island = (hand->columns - (column + 1)) * HAND_ROWS;
    memmove(finger, cursor, island * sizeof(finger_t));

    // Zero leftover space at the end of the hand.
    memset(finger + island, 0, sizeof(finger_t) * HAND_ROWS);

    hand->columns -= 1;
    return true;
}

inline bool hand_get_is_revealed(const hand_t *hand)
{
    for (handsize_t i = 0; i < hand->columns * HAND_ROWS; ++i)
    {
        if (hand->fingers[i].state != CARD_REVEALED)
        {
            return false;
        }
    }

    return true;
}

inline score_t hand_get_score(const hand_t *hand)
{
    score_t score = 0;
    for (handsize_t i = 0; i < hand->columns * HAND_ROWS; ++i)
    {
        score += hand->fingers[i].card;
    }

    return score;
}

inline score_t hand_get_revealed_score(const hand_t *hand)
{
    score_t score = 0;
    for (handsize_t i = 0; i < hand->columns * HAND_ROWS; ++i)
    {
        if (hand->fingers[i].state == CARD_REVEALED)
        {
            score += hand->fingers[i].card;
        }
    }

    return score;
}

inline bool hand_equal(hand_t a, hand_t b)
{
    return memcmp(&a, &b, sizeof(hand_t)) == 0;
}
