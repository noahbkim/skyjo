#ifndef GAME_H
#define GAME_H

#include "cards.h"
#include "hand.h"

#define PLAYERS_MIN 3
#define PLAYERS_MAX 8

/** Safe width for the number of players in a game. */
typedef uint8_t playercount_t;

/** Represents the hand and score of a player in a Skyjo game. */
typedef struct
{
    score_t score;
    hand_t hand;
} player_t;

void player_restore(player_t *player);

/** Overall state for flipping cards then taking turns. */
typedef struct
{
    size_t index;
    enum
    {
        ROUND_FLIPS = 0, // Players flip their first two cards
        ROUND_TURNS,     // Players take turns drawing and placing
        ROUND_END        // A player revealed their hand
    } state : 2;
    playercount_t starter_index;
    playercount_t ender_index;
} round_t;

void round_restore(round_t *round);
void round_start(round_t *round, playercount_t starter_index);
void round_end(round_t *round, playercount_t ender_index);
void round_increment(round_t *round);

/** State for a single player flipping cards or taking a turn. */
typedef struct
{
    size_t index;
    enum
    {
        FLIP_FIRST = 0,        // Player to flip their first card at round start
        FLIP_SECOND,           // Player to flip their second card at round start
        FLIP_DONE,             // Player is done flipping cards'
        TURN_DRAW_OR_TAKE,     // Player to draw or pick up a discard
        TURN_PLACE_OR_DISCARD, // Player took from draw pile and must place or discard
        TURN_PLACE,            // Player stole from draw pile and must place
        TURN_DONE              // Player placed or discarded their card
    } state : 3;
    enum
    {
        TURN_FLIPPED = 0, // Player flipped a card in their hand
        TURN_PLACED_DRAW, // Player placed a card they drew
        TURN_PLACED_TAKE, // Player placed a card taken from the discard
    } result : 2;
    card_t card;
    handsize_t finger_index;
    handsize_t second_finger_index;
} turn_t;

void turn_restore(turn_t *turn);
void turn_flip(turn_t *turn);
void turn_start(turn_t *turn);
void turn_increment(turn_t *turn);

#endif // GAME_H
