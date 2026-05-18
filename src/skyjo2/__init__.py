from __future__ import annotations

import random
from collections.abc import Iterable, Iterator, Sequence
from enum import IntEnum, auto
from typing import NamedTuple

DECK = {
    -2: 5,
    -1: 10,
    0: 15,
    1: 10,
    2: 10,
    3: 10,
    4: 10,
    5: 10,
    6: 10,
    7: 10,
    8: 10,
    9: 10,
    10: 10,
    11: 10,
    12: 10,
}

CARD_VALUES = tuple(DECK)
CARD_COUNTS = tuple(DECK.values())

HAND_ROWS = 3
HAND_COLUMNS = 4


class Finger(NamedTuple):
    """The state of an individual card on a player's board."""

    card_index: int | None
    """The card at this position, if present, by index."""

    is_revealed: bool
    """Whether the card is hidden."""

    @property
    def is_cleared(self) -> bool:
        return self.card_index is None

    @property
    def is_hidden(self) -> bool:
        return not self.is_revealed

    def with_deal(self, card_index: int, is_revealed: bool = False) -> None:
        return Finger(card_index, is_revealed=is_revealed)

    def with_revealed(self) -> Finger:
        assert self.card_index is not None
        assert not self.is_revealed
        return Finger(self.card_index, True)

    def with_replaced(self, card_index: int) -> Finger:
        assert self.card_index is not None
        return Finger(card_index, True)

    def with_cleared(self) -> Finger:
        assert self.card_index is not None
        assert self.is_revealed
        return Finger(None, True)


class Player(NamedTuple):
    """A snapshot of a Skyjo player during a round."""

    score: int
    """The player's cross-game score, not including hand values."""

    hand: tuple[Finger, ...]
    """The player's current hand of cards, column-contiguous."""

    # Extensions

    @property
    def hand_score(self) -> int:
        """The total score of the board, assuming all cards revealed."""

        return sum(
            CARD_VALUES[finger.card_index]
            for finger in self.hand
            if finger.card_index is not None
        )

    @property
    def hand_score_revealed(self) -> int:
        """The cumulative score of visible cards."""

        return sum(
            CARD_VALUES[finger.card_index]
            for finger in self.hand
            if finger.card_index is not None and finger.is_revealed
        )

    @property
    def hand_revealed_count(self) -> int:
        """Get the number of cards revealed but not cleared in the hand."""

        return sum(not finger.is_cleared and finger.is_revealed for finger in self.hand)

    @property
    def hand_cleared_count(self) -> int:
        """Get the number of cards revealed but not cleared in the hand."""

        return sum(finger.is_cleared for finger in self.hand)

    @property
    def is_hand_revealed(self) -> bool:
        """Whether all cards in the hand are cleared or revealed."""

        return all(finger.is_cleared or finger.is_revealed for finger in self.hand)

    def with_deal(self, card_indices: Iterable[int]) -> Player:
        """Deal a hand to the player."""

        # Always deal the top left card facing up. The rules do not specify
        # whether players should, at the start of the game, reveal their
        # initial two cards one at a time (requiring two turn cycles) or
        # simultaneously (requiring only one turn cycle). We opt for the
        # former, as it is more fair with regard to available information. To
        # simulate this efficiently, we deal with top left card face up then
        # allow the player to choose the second card to reveal. Only two
        # positions are commutatively distinct: in the same column and not.
        hand = tuple(
            finger.with_deal(card_index, is_revealed=i == 0)
            for i, (finger, card_index) in enumerate(zip(self.hand, card_indices))
        )

        return Player(score=self.score, hand=hand)

    def with_revealed_card(self, finger_index: int) -> Player:
        """Reveal a card present on the board."""

        assert not self.hand[finger_index].is_cleared
        assert self.hand[finger_index].is_hidden

        hand = (
            *self.hand[:finger_index],
            self.hand[finger_index].with_revealed(),
            *self.hand[finger_index + 1 :],
        )

        # Clear if the revealed card matches its column.
        card_index = self.hand[finger_index].card_index
        column_start = (finger_index // HAND_ROWS) * HAND_ROWS
        column_stop = column_start + HAND_ROWS
        if all(
            finger.is_revealed and finger.card_index == card_index
            for finger in hand[column_start:column_stop]
        ):
            hand = (
                *hand[:column_start],
                *map(Finger.with_cleared, hand[column_start:column_stop]),
                *hand[column_stop:],
            )

        return Player(score=self.score, hand=hand)

    def with_replaced_card(self, finger_index: int, card_index: int) -> Player:
        """Replace a card present on the board."""

        assert not self.hand[finger_index].is_cleared

        hand = (
            *self.hand[:finger_index],
            self.hand[finger_index].with_replaced(card_index),
            *self.hand[finger_index + 1 :],
        )

        # Clear if the replaced card matches its column.
        column_start = (finger_index // HAND_ROWS) * HAND_ROWS
        column_stop = column_start + HAND_ROWS
        if all(
            finger.is_revealed and finger.card_index == card_index
            for finger in hand[column_start:column_stop]
        ):
            hand = (
                *hand[:column_start],
                *map(Finger.with_cleared, hand[column_start:column_stop]),
                *hand[column_stop:],
            )

        return Player(score=self.score, hand=hand)


def pick_random_index(pile: Sequence[int], rng: random.Random = random) -> int:
    needle = rng.randint(0, sum(pile))
    for index, haystack in enumerate(pile):
        if needle < haystack:
            return index
        needle -= haystack
    assert False


def with_draw(pile: Sequence[int], index: int) -> tuple[int, ...]:
    assert pile[index] > 0
    return (*pile[:index], pile[index] - 1, *pile[index + 1 :])


def with_discard(pile: Sequence[int], index: int) -> tuple[int, ...]:
    return (*pile[:index], pile[index] + 1, *pile[index + 1 :])


class State(IntEnum):
    NULL = auto()
    REVEAL_SECOND_CARD = auto()
    DRAW_OR_REPLACE_WITH_DISCARD = auto()
    DISCARD_DRAW_AND_REVEAL_OR_REPLACE_WITH_DRAW = auto()


class Game(NamedTuple):
    """A snapshot of a Skyjo game."""

    turn: int
    """The number of turns that have been played, i.e. the turn index."""

    state: int | None
    """The current state of the game."""

    drawn_card_index: int | None
    """The card currently drawn from the draw pile, if requested."""

    draw_pile: tuple[int, ...]
    """The unordered contents of the draw pile excluding the drawn card."""

    discarded_card_index: int | None
    """The card visible on top of the draw pile."""

    discard_pile: tuple[int, ...]
    """The unordered contents of the discard pile."""

    players: tuple[Player, ...]
    """The complete array of player slots."""

    @property
    def player(self) -> Player:
        return self.players[0]

    @property
    def player_index(self) -> int:
        return 0

    @property
    def player_index_fixed(self) -> int:
        return self.turn % len(self.players)

    @property
    def players_fixed(self) -> tuple[Player]:
        offset = self.player_index_fixed
        return self.players[-offset:] + self.players[:-offset]

    @property
    def hand_scores(self) -> tuple[int, ...]:
        return tuple(player.hand_score for player in self.players)

    @property
    def hand_scores_final(self) -> tuple[int, ...]:
        """Get player scores as if the current player ended the round."""

        scores = [player.hand_score for player in self.players]
        # Penalize the round ender if they didn't have the lowest score.
        if scores[0] >= min(scores[1:]):
            scores[0] *= 2
        return tuple(scores)

    @property
    def winner(self) -> Player:
        return self.players[self.winner_index]

    @property
    def winner_index(self) -> Player:
        return min(range(len(self.players)), key=lambda i: self.players[i].hand_score)

    @property
    def winner_index_fixed(self) -> int:
        return (self.winner_index + self.turn) % len(self.players)

    @staticmethod
    def new(*, players: int) -> Game:
        """Construct a new game."""

        finger = Finger(card_index=None, is_revealed=True)
        player = Player(score=0, hand=(finger,) * HAND_ROWS * HAND_COLUMNS)

        return Game(
            turn=0,
            state=State.NULL,
            drawn_card_index=None,
            draw_pile=tuple(DECK.values()),
            discarded_card_index=None,
            discard_pile=(0,) * len(DECK),
            players=(player,) * players,
        )

    def with_deal(self, card_indices: Iterable[int]) -> Game:
        assert self.turn == 0
        assert self.state == State.NULL
        assert self.drawn_card_index is None
        assert self.discarded_card_index is None

        draw_pile = list(self.draw_pile)

        def deal_card_index_from_draw_pile() -> Iterator[int]:
            for card_index in card_indices:
                assert draw_pile[card_index] > 0
                draw_pile[card_index] -= 1
                yield card_index

        dealer = deal_card_index_from_draw_pile()

        # Exhaust the draw pile before we snapshot it as a `tuple` below.
        discarded_card_index = next(dealer)
        players = tuple(player.with_deal(dealer) for player in self.players)

        return Game(
            turn=self.turn,
            state=State.REVEAL_SECOND_CARD,
            drawn_card_index=None,
            draw_pile=tuple(draw_pile),
            discarded_card_index=discarded_card_index,
            discard_pile=self.discard_pile,
            players=players,
        )

    def with_random_deal(self, rng: random.Random = random) -> Game:
        """Deal players and set an initial discard."""

        assert self.turn == 0
        assert self.state == State.NULL
        assert self.drawn_card_index is None
        assert self.discarded_card_index is None

        draw_pile = list(self.draw_pile)

        def deal_card_index_from_draw_pile() -> Iterator[int]:
            while True:
                deal_card_index = pick_random_index(draw_pile, rng=rng)
                draw_pile[deal_card_index] -= 1
                yield deal_card_index

        dealer = deal_card_index_from_draw_pile()

        # Exhaust the draw pile before we snapshot it as a `tuple` below.
        discarded_card_index = next(dealer)
        players = tuple(player.with_deal(dealer) for player in self.players)

        return Game(
            turn=self.turn,
            state=State.REVEAL_SECOND_CARD,
            drawn_card_index=None,
            draw_pile=tuple(draw_pile),
            discarded_card_index=discarded_card_index,
            discard_pile=self.discard_pile,
            players=players,
        )

    def with_second_card_revealed(self, finger_index: int) -> Game:
        """Reveal a second card during initial board setup."""

        assert self.state == State.REVEAL_SECOND_CARD

        # Reveal the requested card.
        players = (
            self.players[0].with_revealed_card(finger_index),
            *self.players[1:],
        )

        # Rotate players. Left separate from reveal for clarity.
        players = (*players[1:], players[0])

        # Check if we're done revealing cards. If we are, we need to skip turns
        # until we've reached the player who revealed the lowest cards.
        turn = self.turn + 1
        state = self.state
        if turn == len(players):
            state = State.DRAW_OR_REPLACE_WITH_DISCARD
            turn += min(
                range(len(players)),
                key=lambda i: players[i].hand_score_revealed,
            )

        return Game(
            turn=turn,
            state=state,
            drawn_card_index=self.drawn_card_index,
            draw_pile=self.draw_pile,
            discarded_card_index=self.discarded_card_index,
            discard_pile=self.discard_pile,
            players=players,
        )

    def with_drawn_card(self, drawn_card_index: int) -> Game:
        """Draw a specific card from the pile but do nothing with it."""

        assert self.state == State.DRAW_OR_REPLACE_WITH_DISCARD
        assert self.drawn_card_index is None
        assert sum(self.draw_pile) > 0

        # Remove the drawn card from the draw pile.
        draw_pile = with_draw(self.draw_pile, drawn_card_index)

        # Shuffle the discards into the draw pile if there are no more draws.
        # Always do this at the end of states, if possible, so we don't have
        # to juggle reshuffles before draws.
        discard_pile = self.discard_pile
        if sum(draw_pile) == 0:
            assert sum(discard_pile) > 0
            draw_pile = discard_pile
            discard_pile = (0,) * len(discard_pile)

        return Game(
            turn=self.turn,
            state=State.DISCARD_DRAW_AND_REVEAL_OR_REPLACE_WITH_DRAW,
            drawn_card_index=drawn_card_index,
            draw_pile=draw_pile,
            discarded_card_index=self.discarded_card_index,
            discard_pile=discard_pile,
            players=self.players,
        )

    def with_random_drawn_card(self, rng: random.Random = random) -> Game:
        """Draw a random card from the pile but do nothing with it."""

        drawn_card_index = pick_random_index(self.draw_pile, rng=rng)
        return self.with_drawn_card(drawn_card_index)

    def with_draw_discarded_and_card_revealed(self, finger_index: int) -> Game:
        """Discard the drawn card and reveal a card in the current hand."""

        assert self.state == State.DISCARD_DRAW_AND_REVEAL_OR_REPLACE_WITH_DRAW
        assert self.drawn_card_index is not None
        assert self.discarded_card_index is not None

        # Put the current discarded card into the pile.
        discard_pile = with_discard(self.discard_pile, self.discarded_card_index)

        # Move the drawn card to be the discarded card.
        discarded_card_index = self.drawn_card_index
        drawn_card_index = None

        # Reveal the requested card.
        card_index = self.players[0].hand[finger_index].card_index
        players = (
            self.players[0].with_revealed_card(finger_index),
            *self.players[1:],
        )

        # If the card got cleared, we need to replace the current discarded
        # card and update the discard pile. This is a little wasteful but we'll
        # wait to profile before we inline it since it's rare.
        if players[0].hand[finger_index].is_cleared:
            discard_pile = with_discard(discard_pile, discarded_card_index)
            discard_pile = with_discard(discard_pile, card_index)
            discard_pile = with_discard(discard_pile, card_index)
            discarded_card_index = card_index

        # Check if the current player won. If so, stop the game. Otherwise,
        # rotate players so the next is in the first slot.
        if players[0].is_hand_revealed:
            state = State.NULL
        else:
            players = (*players[1:], players[0])
            state = State.DRAW_OR_REPLACE_WITH_DISCARD

        return Game(
            turn=self.turn + 1,
            state=state,
            drawn_card_index=drawn_card_index,
            draw_pile=self.draw_pile,
            discarded_card_index=discarded_card_index,
            discard_pile=discard_pile,
            players=players,
        )

    def with_card_replaced_with_draw(self, finger_index: int) -> Game:
        """Replace a card in the current hand with the drawn card."""

        assert self.state == State.DISCARD_DRAW_AND_REVEAL_OR_REPLACE_WITH_DRAW
        assert self.drawn_card_index is not None

        # Put the current discarded card into the pile.
        discard_pile = with_discard(self.discard_pile, self.discarded_card_index)

        # Move the replaced card to be the new discarded card.
        discarded_card_index = self.players[0].hand[finger_index].card_index

        # Replace the card in the player's hand.
        card_index = self.drawn_card_index
        drawn_card_index = None
        players = (
            self.players[0].with_replaced_card(finger_index, card_index),
            *self.players[1:],
        )

        # If the card got cleared, we need to replace the current discarded
        # card and update the discard pile. This is a little wasteful but we'll
        # wait to profile before we inline it since it's rare.
        if players[0].hand[finger_index].is_cleared:
            discard_pile = with_discard(discard_pile, discarded_card_index)
            discard_pile = with_discard(discard_pile, card_index)
            discard_pile = with_discard(discard_pile, card_index)
            discarded_card_index = card_index

        # Check if the current player won. If so, stop the game. Otherwise,
        # rotate players so the next is in the first slot.
        if players[0].is_hand_revealed:
            state = State.NULL
        else:
            players = (*players[1:], players[0])
            state = State.DRAW_OR_REPLACE_WITH_DISCARD

        return Game(
            turn=self.turn + 1,
            state=state,
            drawn_card_index=drawn_card_index,
            draw_pile=self.draw_pile,
            discarded_card_index=discarded_card_index,
            discard_pile=discard_pile,
            players=players,
        )

    def with_card_replaced_with_discard(self, finger_index: int) -> Game:
        """Replace a card in the current hand with the drawn card."""

        assert self.state == State.DRAW_OR_REPLACE_WITH_DISCARD
        assert self.drawn_card_index is None

        discard_pile = self.discard_pile

        # Put the current discarded card into the pile.
        card_index = self.discarded_card_index
        discarded_card_index = self.players[0].hand[finger_index].card_index

        # Replace the card in the player's hand.
        players = (
            self.players[0].with_replaced_card(finger_index, card_index),
            *self.players[1:],
        )

        # If the card got cleared, we need to replace the current discarded
        # card and update the discard pile. This is a little wasteful but we'll
        # wait to profile before we inline it since it's rare.
        if players[0].hand[finger_index].is_cleared:
            discard_pile = with_discard(discard_pile, discarded_card_index)
            discard_pile = with_discard(discard_pile, card_index)
            discard_pile = with_discard(discard_pile, card_index)
            discarded_card_index = card_index

        # Check if the current player won. If so, stop the game. Otherwise,
        # rotate players so the next is in the first slot.
        if players[0].is_hand_revealed:
            state = State.NULL
        else:
            players = (*players[1:], players[0])
            state = State.DRAW_OR_REPLACE_WITH_DISCARD

        return Game(
            turn=self.turn + 1,
            state=state,
            drawn_card_index=self.drawn_card_index,
            draw_pile=self.draw_pile,
            discarded_card_index=discarded_card_index,
            discard_pile=discard_pile,
            players=players,
        )
