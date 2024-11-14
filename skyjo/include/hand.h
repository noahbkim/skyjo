#ifndef HAND_H
#define HAND_H

#include "cards.h"

#include <stdbool.h>
#include <stddef.h>

#define HAND_WIDTH 4
#define HAND_HEIGHT 3
#define HAND_SIZE 12

/** A safe integer width for an index in a hand. */
typedef uint8_t handsize_t;

/** A safe integer width for score values. */
typedef ptrdiff_t score_t;

/** The slot for a card in a hand.
 *
 * Includes the card value and a bit indicating whether said card has
 * has been flipped (if its value is visible). Packs into a byte.
 */
typedef struct
{
    card_t card : CARD_BITS;
    enum
    {
        CARD_CLEARED = 0,
        CARD_HIDDEN,
        CARD_REVEALED,
    } state : 2;
} finger_t;

card_t finger_reveal(finger_t *finger);
card_t finger_replace(finger_t *finger, card_t card);

/** A player's set of cards arranged in a grid.
 *
 * The `hand_t` represents its `finger_t`'s as a flat array for both
 * convenience (easier to compact) and performance (inline).
 *
 *              0   1   2   3 < columns
 *             -4  -3  -2  -1
 *
 *     0 -1  [  0,  3,  6,  9,
 *     1 -2     1,  4,  7, 10,
 *     2 -3     2,  5,  8, 11, ]
 *     ^                    ^
 *     rows                 indices
 */
typedef struct
{
    handsize_t columns;
    finger_t fingers[HAND_SIZE];
} hand_t;

void hand_restore(hand_t *hand);
bool hand_try_clear(hand_t *hand, handsize_t column);
bool hand_get_is_revealed(const hand_t *hand);
score_t hand_get_revealed_score(const hand_t *hand);
score_t hand_get_score(const hand_t *hand);

#endif // HAND_H