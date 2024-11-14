#include "hand.h"

#include <assert.h>
#include <string.h>

card_t finger_reveal(finger_t *finger)
{
    assert(finger->state == CARD_HIDDEN);

    finger->state = CARD_REVEALED;
    return finger->card;
}

card_t finger_replace(finger_t *finger, card_t card)
{
    assert(finger->state == CARD_HIDDEN || finger->state == CARD_REVEALED);

    card_t replaced_card = finger->card;
    finger->card = card;
    return replaced_card;
}

void hand_restore(hand_t *hand)
{
    hand->columns = 0;
    memset(hand->fingers, 0, HAND_SIZE * sizeof(finger_t));
}

bool hand_try_clear(hand_t *hand, handsize_t column)
{
    assert(column < hand->columns);

    finger_t *finger = hand->fingers + column * HAND_HEIGHT;
    if (finger->state != CARD_REVEALED)
    {
        return false;
    }

    finger_t *cursor = finger;
    for (handsize_t r = 1; r < HAND_HEIGHT; ++r, ++cursor)
    {
        if (cursor->state != CARD_REVEALED || cursor->card != finger->card)
        {
            return false;
        }
    }

    handsize_t island = (hand->columns - (column + 1)) * HAND_HEIGHT;
    memmove(finger, cursor, island * sizeof(finger_t));
    memset(finger + island, 0, sizeof(finger_t));
    return true;
}

bool hand_get_is_revealed(const hand_t *hand)
{
    for (handsize_t i = 0; i < hand->columns * HAND_HEIGHT; ++i)
    {
        if (hand->fingers[i].state != CARD_REVEALED)
        {
            return false;
        }
    }

    return true;
}

score_t hand_get_revealed_score(const hand_t *hand)
{
    score_t score = 0;
    for (handsize_t i = 0; i < hand->columns * HAND_HEIGHT; ++i)
    {
        if (hand->fingers[i].state == CARD_REVEALED)
        {
            score += hand->fingers[i].card;
        }
    }

    return score;
}

score_t hand_get_score(const hand_t *hand)
{
    score_t score = 0;
    for (handsize_t i = 0; i < hand->columns * HAND_HEIGHT; ++i)
    {
        score += hand->fingers[i].card;
    }

    return score;
}
