# Skyjo

The following is transcribed from the English section of the Skyjo manual with
minor tweaks to grammar, punctuation, and wording.

## Game Idea and Game Target

The game Skyjo runs over multiple rounds and ends as soon as a player reached
100 points or more. The player with the least points wins the game. Therefore,
the goal of the game is to get rid of as many high cards as possible.

## Preparing a Game Round

First, each player receives 12 cards facing down from the well-shuffled deck of
cards. Then, a card is placed openly in the middle of the table; it makes the
discard pile. The rest of cards are placed facing down next to it as the draw
pile. Each player places their 12 playing cards in four vertical rows with three
cards per row. Then, each player reveals two of their playing cards at will.

## The Start of a Game Round

The player with the highest sum (points) of the two open cards gets to start the
first game round. 

> Example: the cards of Player A show a 12 and a -2, which results in a total of
> 10. The cards of Player B show a 4 and a 2, which results in a total of 6.
> Player A gets to to start first because their cards totaled to a bigger sum.

The player who ended the current game round gets to start the new game round.

## Course of the Game

The play starts with the drawing of a card. The player may choose whether they
want to draw the top open card of the discard pile or the top hidden card of the
draw pile.

If they choose the open card from the discard pile, they have to exchange it
with one of their playing cards and openly display it. They may freely choose
among the open and hidden cards, and the hidden cards may not be turned around
and looked at. The exchanged card is openly placed on the discard pile.

If they choose a hidden card from the draw pile, they may look at the card and
choose whether they want to excahnge it for one of their hidden or open game
cards. If they keep it, it runs as described above. If they do not want to keep
the drawn card, they must place it on the discard pile and reveal one of their
hidden cards. Thus, their move ends, and it is the move of the next player in a
clockwise direction.

The game round ends as soon as a player has revealed all their cards. Then, each
subsequent player will have one more move. Afterwards, the points of each
player's open and hidden cards are added up and added to their current score.

The player who finishes the game round must have the smallest number of points
in the evaluation of the round. If they do not have the lowest points because
another player has reached fewer or the same amount, their points collected in
this round are doubled. Note that this rule only applies to positive points, not
zero or negative points.

Example: Player A shows all their cards and finishes the game round. They have
10 points at the end of the game round, Player B has 24 points, and Player C has
collected 10 points as well. Player A does not have the lowest score, so their
score is doubled to 20.

```
Player A     Player B     Player C
-----------  -----------  -----------
 0  2  1 -1   1  3  7  4   2 -2  5  3
 0 -2  1  3  -2  0 -1  0   0  0  2 -1
 5  3  0 -2   1  0 12 -1   2 -1  0  0

Player A: 0 + 0 + 5 + 2 - 2 + 3 + 1 + 1 + 0 - 1 + 3 - 2 = 10 -> 20
Player B: 1 - 2 + 1 + 3 + 0 + 0 + 7 - 1 + 12 + 4 + 0 - 1 = 24
Player C: 2 + 0 + 2 - 2 + 0 - 1 + 5 + 2 + 0 + 3 - 1 + 0 = 10
```

Special rule: if a player has uncovered or places three of the same playing card
in a vertical row, the player has to place the entire row of cards on the
discard pile. This rule also applies in the last turn of a game round when the
remaining cards are turned over and added up for evaluation. The triplet always
has to be placed on the discard pile after the exchanged card if the row of
cards is achieved due to an exchange with the draw pile or discard pile.

```
discard: 9                  discard: -1                 discard: -1
-----------    -------->    -----------    -------->    -----------
[]  9  0  4     swap -1     []  9  0  4     discard     []     0  4
12  9 [] []   for discard   12  9 [] []     column      12    [] []
[] -1 []  2                 []  9 []  2                 []    []  2
```

## End of the Game

At the end of each round the points of every player are added to their current
score. The game ends as soon as one of the players has reached 100 points or
more. The player with the lowest score wins the game.
