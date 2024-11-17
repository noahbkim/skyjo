#include "game.h"

#include <string.h>

void player_restore(player_t *player)
{
    player->score = 0;
    hand_restore(&player->hand);
}

void round_restore(round_t *round)
{
    memset(round, 0, sizeof(round_t));
}

void round_increment(round_t *round)
{
    round->index += 1;
    round->state = ROUND_FLIPS;
}

void round_start(round_t *round, playercount_t starter_index)
{
    round->state = ROUND_TURNS;
    round->starter_index = starter_index;
}

void round_end(round_t *round, playercount_t ender_index)
{
    round->state = ROUND_END;
    round->ender_index = ender_index;
}

void turn_restore(turn_t *turn)
{
    memset(turn, 0, sizeof(turn_t));
}

void turn_flip(turn_t *turn)
{
    turn->state = FLIP_FIRST;
}

void turn_start(turn_t *turn)
{
    turn->state = TURN_DRAW_OR_TAKE;
}

void turn_increment(turn_t *turn)
{
    turn->index += 1;
    turn->state = TURN_DRAW_OR_TAKE;
}
