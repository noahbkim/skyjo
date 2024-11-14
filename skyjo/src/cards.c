#include "cards.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

void cards_restore(cards_t *cards)
{
    cards->discard_index = 0;
    cards->draw_index = 1;
    memcpy(cards->buffer, DECK, DECK_SIZE * sizeof(card_t));
}

/** Randomly shuffle a buffer of cards.
 *
 * Assumes `size` is greater than 1 and reasonably smaller than
 * `RAND_MAX` so the shuffle will be sufficiently random. Discussed
 * further here: https://stackoverflow.com/a/6127606/3015219
 */
static inline void shuffle(card_t *buffer, size_t size)
{
    assert(size > 1);

    for (size_t i = 0; i < size - 1; ++i)
    {
        size_t j = i + rand() / (RAND_MAX / (size - i) + 1);
        card_t swap = buffer[j];
        buffer[j] = buffer[i];
        buffer[i] = swap;
    }
}

void cards_shuffle(cards_t *cards)
{
    shuffle(cards->buffer, DECK_SIZE);
}

card_t cards_get_last_discard(const cards_t *cards)
{
    assert(cards->discard_index < DECK_SIZE);
    return cards->buffer[cards->discard_index];
}

card_t cards_get_next_draw(const cards_t *cards)
{
    assert(cards->draw_index < DECK_SIZE);
    return cards->buffer[cards->draw_index];
}

card_t cards_draw(cards_t *cards)
{
    assert(cards->draw_index < DECK_SIZE);
    card_t card = cards->buffer[cards->draw_index++];

    if (cards->draw_index == DECK_SIZE)
    {
        card_t last_discard = cards->buffer[cards->discard_index];
        decksize_t buried_count = cards->discard_index;
        shuffle(cards->buffer, buried_count);
        cards->draw_index -= buried_count;
        size_t buried_size = (size_t)buried_count * sizeof(card_t);
        memmove(cards->buffer + cards->draw_index, cards->buffer, buried_size);
        cards->discard_index = 0;
        cards->buffer[0] = last_discard;
    }

    return card;
}

void cards_discard(cards_t *cards, card_t card)
{
    assert(cards->discard_index < DECK_SIZE);
    assert(cards->discard_index + 1 < cards->draw_index);
    cards->buffer[++cards->discard_index] = card;
}

card_t cards_replace_discard(cards_t *cards, card_t card)
{
    assert(cards->discard_index < DECK_SIZE);
    card_t last_discard = cards->buffer[cards->discard_index];
    cards->buffer[cards->discard_index] = card;
    return last_discard;
}
