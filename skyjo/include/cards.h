#ifndef CARDS_H
#define CARDS_H

#include <stddef.h>
#include <stdint.h>

#define CARD_BITS 5
#define DECK_SIZE 150

/** Sane wrapping for the `rand_r` type. */

/** The integer type for individual cards. */
typedef int8_t card_t;

/** A safe integer width for an index in a deck. */
typedef uint8_t decksize_t;

/** The discard and draw piles in the center of the table.
 *
 * Because there are a fixed number of cards for the duration of a
 * game, we can represent both the discard and draw piles using a list
 * of constant size, `buffer`, and two indices `discard_index` and
 * `draw_index`. A (dubiously) shuffled `buffer` look like:
 *
 *       discard_index
 *       v
 *     {-2, -1, 0, 1, 2, 3, 4, ...} < buffer
 *           ^
 *           draw_index
 *
 * This represents a discard pile with just -2 facing up and a draw
 * pile containing everything else (with a -1 on top).
 *
 * When a card is drawn, the card at `draw_index` is retrieved then
 * the index is incremented. We don't bother overwriting draw cards
 * with a sentinel because they should never be accessed. If we called
 * `cards_draw` four times, our indices would now be:
 *
 *       discard_index
 *       v
 *     {-2, -1, 0, 1, 2, 3, 4, ...} < buffer
 *                       ^
 *                       draw_index
 *
 * When a card is discarded, we increment `discard_index` and set the
 * corresponding element in `buffer` to its value. If we discarded our
 * four cards in reverse order, we would have:
 *
 *                    discard_index
 *                    v
 *     {-2, 2, 1, 0, -1, 3, 4, ...} < buffer
 *                       ^
 *                       draw_index
 *
 * Once we've exhausted our draw pile, we can shuffle the buried
 * discards and slice them back into the draw pile.
 */
typedef struct
{
    decksize_t discard_index;
    decksize_t draw_index;
    card_t buffer[DECK_SIZE];
} cards_t;

void cards_restore(cards_t* cards);
void cards_shuffle(cards_t* cards);
card_t cards_get_last_discard(const cards_t* cards);
card_t cards_get_next_draw(const cards_t* cards);
card_t cards_draw(cards_t* cards);
void cards_discard(cards_t* cards, card_t card);
card_t cards_replace_discard(cards_t* cards, card_t card);

#endif  // CARDS_H
