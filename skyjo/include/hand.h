#ifndef HAND_H
#define HAND_H

#include "cards.h"

#define HAND_WIDTH 4
#define HAND_HEIGHT 3
#define HAND_SIZE 12

/** A safe integer width for an index in a hand. */
typedef uint8_t handsize_t;

/** A safe integer width for score values. */
typedef ssize_t score_t;

/** The slot for a card in a hand.
 * 
 * Includes the card value and a bit indicating whether said card has
 * has been flipped (if its value is visible). Packs into a byte.
 */
typedef struct
{
    card_t card : CARD_BITS;
    bool is_flipped : 1;
} finger_t;

void finger_flip(finger_t* finger);
void finger_replace(finger_t* finger, card_t card);

/** A player's set of cards arranged in a grid.
 *
 * The `Hand` represents its `Finger`'s as a flat `list` for both
 * performance (less indirection) and convenience (easy to slice). The
 * number of rows is constant, so the number of columns can always be
 * determined by `len(_fingers) / rows`. Fingers can also be indexed by
 * row and column, in which case they are arranged as follows:
 *
 *              0   1   2   3 < columns
 *             -4  -3  -2  -1
 *
 *     0 -1  [  0,  1,  2,  3,
 *     1 -2     4,  5,  6,  7,
 *     2 -3     8,  9, 10, 11, ]
 *     ^                    ^
 *     rows                 indices
 */
typedef struct
{
    handsize_t width;
    handsize_t height;
    finger_t fingers[HAND_SIZE];
} hand_t;

void hand_restore(hand_t* hand);
void hand_try_clear(hand_t* hand, handsize_t column);
void hand_get_size(const hand_t* hand);
bool hand_get_is_flipped(const hand_t* hand);
card_t hand_get_highest_flipped_card(const hand_t* hand);
score_t hand_get_total_score(const hand_t* hand);

#endif  // HAND_H