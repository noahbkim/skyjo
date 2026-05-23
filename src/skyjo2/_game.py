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

# MARK: Finger


class Finger(NamedTuple):
    """The state of an individual card on a player's board."""

    card_index: int | None
    """The card at this position, if present, by index."""

    is_revealed: bool
    """Whether the card is face up (revealed) or face down (hidden)."""

    @property
    def is_cleared(self) -> bool:
        return self.is_revealed and self.card_index is None

    @property
    def is_hidden(self) -> bool:
        return not self.is_revealed

    def with_card(self, card_index: int | None) -> Finger:
        return Finger(card_index, card_index is not None)

    def with_cleared(self) -> Finger:
        assert self.card_index is not None
        assert self.is_revealed
        return Finger(None, True)


# MARK: Player


class Player(NamedTuple):
    """A snapshot of a Skyjo player during a round."""

    score: int
    """The player's cross-game score, not including hand values."""

    hand: tuple[Finger, ...]
    """The player's current hand of cards, column-contiguous."""

    @property
    def hand_score(self) -> int:
        """The total score of the hand, ignoring revealed."""

        assert self.is_hand_revealed
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
            if not finger.is_cleared and finger.is_revealed
        )

    @property
    def hand_hidden_count(self) -> int:
        """Get the number of cards revealed but not cleared in the hand."""

        return sum(not finger.is_cleared and finger.is_hidden for finger in self.hand)

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

    def with_first_card_dealt(self, card_index: int) -> Player:
        """Deal the top left card to the player."""

        # Always deal the top left card facing up. The rules do not specify
        # whether players should, at the start of the game, reveal their
        # initial two cards one at a time (requiring two turn cycles) or
        # simultaneously (requiring only one turn cycle). We opt for the
        # former, as it is more fair with regard to available information. To
        # simulate this efficiently, we deal the top-left card face-up then
        # allow the player to choose the second card to reveal. Only two
        # positions are commutatively distinct: in the same column and not.
        hand = tuple(
            finger.with_card(card_index) if i == 0 else finger.with_card(None)
            for i, finger in enumerate(self.hand)
        )

        return Player(score=self.score, hand=hand)

    def with_card(self, finger_index: int, card_index: int) -> Player:
        """Reveal a card present on the board."""

        assert not self.hand[finger_index].is_cleared

        hand = (
            *self.hand[:finger_index],
            self.hand[finger_index].with_card(card_index),
            *self.hand[finger_index + 1 :],
        )

        # Clear if the revealed card matches its column.
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

    def with_hidden_cards_revealed(
        self,
        revealed_card_indices: Iterable[int],
    ) -> Player:
        """Replace remaining hidden cards."""

        revealed_card_indices = iter(revealed_card_indices)
        hand = tuple(
            finger
            if not finger.is_hidden
            else finger.with_card(next(revealed_card_indices))
            for finger in self.hand
        )

        return Player(score=self.score, hand=hand)

    def with_score(self, score: int) -> Player:
        """Update the player's score."""

        return Player(score=score, hand=self.hand)


# MARK: Game


def _pick_random_card_index(
    draw_pile: Sequence[int],
    rng: random.Random,
) -> int:
    needle = rng.randint(0, sum(draw_pile) - 1)
    for card_index, haystack in enumerate(draw_pile):
        if needle < haystack:
            return card_index
        needle -= haystack
    assert False


def _with_card_drawn(
    draw_pile: Sequence[int],
    card_index: int,
) -> tuple[int, ...]:
    assert draw_pile[card_index] > 0
    return (
        *draw_pile[:card_index],
        draw_pile[card_index] - 1,
        *draw_pile[card_index + 1 :],
    )


def _with_card_discarded(
    discard_pile: Sequence[int],
    card_index: int,
) -> tuple[int, ...]:
    return (
        *discard_pile[:card_index],
        discard_pile[card_index] + 1,
        *discard_pile[card_index + 1 :],
    )


def _with_discard_pile_shuffled(
    draw_pile: tuple[int],
    discard_pile: tuple[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if sum(draw_pile) == 0:
        assert sum(discard_pile) > 0
        draw_pile = discard_pile
        discard_pile = (0,) * len(discard_pile)
    return draw_pile, discard_pile


class GameState(IntEnum):
    DEAL_FIRST_CARDS = auto()
    REVEAL_SECOND_CARD = auto()
    DRAW_OR_REPLACE_WITH_DISCARD = auto()
    DISCARD_DRAW_AND_REVEAL_OR_REPLACE_WITH_DRAW = auto()
    REVEAL_HIDDEN_CARDS = auto()


class Game(NamedTuple):
    """A snapshot of a Skyjo game."""

    state: GameState
    """The current state of the game."""

    turn_index: int
    """The number of turns that have been played, reset after reveal."""

    draw_pile: tuple[int, ...]
    """The unordered contents of the draw pile excluding the drawn card."""

    drawn_card_index: int | None
    """The card currently drawn from the draw pile, if requested."""

    discard_pile: tuple[int, ...]
    """The unordered contents of the discard pile."""

    discarded_card_index: int | None
    """The card visible on top of the draw pile."""

    players: tuple[Player, ...]
    """The complete array of player slots."""

    start_player_index: int
    """The index of the first player to go this game."""

    end_player_index: int | None
    """The index of the player who ended the last game."""

    @property
    def player(self) -> Player:
        return self.players[self.player_index]

    @property
    def player_index(self) -> int:
        return (self.start_player_index + self.turn_index) % len(self.players)

    @property
    def player_hand_hidden_counts(self) -> tuple[int]:
        return tuple(player.hand_hidden_count for player in self.players)

    @property
    def player_hand_revealed_counts(self) -> tuple[int]:
        return tuple(player.hand_revealed_count for player in self.players)

    @property
    def player_hand_cleared_counts(self) -> tuple[int]:
        return tuple(player.hand_cleared_count for player in self.players)

    # MARK: Construct

    @staticmethod
    def new(*, players: int) -> Game:
        """Construct a new game."""

        finger = Finger(card_index=None, is_revealed=True)
        player = Player(score=0, hand=(finger,) * HAND_ROWS * HAND_COLUMNS)
        return Game(
            state=GameState.DEAL_FIRST_CARDS,
            turn_index=0,
            draw_pile=tuple(DECK.values()),
            drawn_card_index=None,
            discard_pile=(0,) * len(DECK),
            discarded_card_index=None,
            players=(player,) * players,
            start_player_index=0,
            end_player_index=None,
        )

    # MARK: Deal

    def with_discard_and_first_cards_dealt(self, card_indices: Iterable[int]) -> Game:
        """Deal a set series of cards to the discard slot and players."""

        state = self.state
        players = self.players
        end_player_index = self.end_player_index
        del self

        assert state == GameState.DEAL_FIRST_CARDS

        state = GameState.REVEAL_SECOND_CARD
        turn_index = 0
        mutable_draw_pile = list(DECK.values())
        discard_pile = (0,) * len(DECK)
        start_player_index = end_player_index or 0
        drawn_card_index = None
        discarded_card_index = None

        def deal_card_index_from_draw_pile() -> Iterator[int]:
            for card_index in card_indices:
                assert mutable_draw_pile[card_index] > 0
                mutable_draw_pile[card_index] -= 1
                yield card_index

        # Update the draw pile before we snapshot it as a `tuple` below.
        dealer = deal_card_index_from_draw_pile()
        discarded_card_index = next(dealer)
        players = tuple(
            player.with_first_card_dealt(card_index)
            for player, card_index in zip(players, dealer, strict=True)
        )

        # Make the draw pile immutable again.
        draw_pile = tuple(mutable_draw_pile)

        return Game(
            state=state,
            turn_index=turn_index,
            draw_pile=draw_pile,
            drawn_card_index=drawn_card_index,
            discard_pile=discard_pile,
            discarded_card_index=discarded_card_index,
            players=players,
            start_player_index=start_player_index,
            end_player_index=end_player_index,
        )

    def with_random_discard_and_first_cards_dealt(self, rng: random.Random) -> Game:
        """Deal players and set an initial discard."""

        state = self.state
        players = self.players
        end_player_index = self.end_player_index
        del self

        assert state == GameState.DEAL_FIRST_CARDS

        state = GameState.REVEAL_SECOND_CARD
        turn_index = 0
        mutable_draw_pile = list(DECK.values())
        discard_pile = (0,) * len(DECK)
        start_player_index = end_player_index or 0
        drawn_card_index = None
        discarded_card_index = None

        def deal_card_index_from_draw_pile() -> Iterator[int]:
            while True:
                deal_card_index = _pick_random_card_index(mutable_draw_pile, rng)
                mutable_draw_pile[deal_card_index] -= 1
                yield deal_card_index

        # Update the draw pile before we snapshot it as a `tuple` below.
        dealer = deal_card_index_from_draw_pile()
        discarded_card_index = next(dealer)
        players = tuple(
            player.with_first_card_dealt(card_index)
            for player, card_index in zip(players, dealer)
        )

        # Make the draw pile immutable again.
        draw_pile = tuple(mutable_draw_pile)

        return Game(
            state=state,
            turn_index=turn_index,
            draw_pile=draw_pile,
            drawn_card_index=drawn_card_index,
            discard_pile=discard_pile,
            discarded_card_index=discarded_card_index,
            players=players,
            start_player_index=start_player_index,
            end_player_index=end_player_index,
        )

    # MARK: Second card

    def with_second_card_revealed(
        self,
        finger_index: int,
        revealed_card_index: int,
    ) -> Game:
        """Reveal a second card during initial board setup."""

        state = self.state
        turn_index = self.turn_index
        draw_pile = self.draw_pile
        drawn_card_index = self.drawn_card_index
        discard_pile = self.discard_pile
        discarded_card_index = self.discarded_card_index
        players = self.players
        start_player_index = self.start_player_index
        end_player_index = self.end_player_index
        del self

        assert state == GameState.REVEAL_SECOND_CARD
        assert turn_index < len(players)
        assert sum(draw_pile) > 0
        assert drawn_card_index is None
        assert discarded_card_index is not None
        assert start_player_index == (end_player_index or 0)

        player_index = (start_player_index + turn_index) % len(players)

        # Formally draw the card we're revealing.
        draw_pile = _with_card_drawn(draw_pile, revealed_card_index)

        # Reveal the requested card.
        assert players[player_index].hand[finger_index].is_hidden
        players = (
            *players[:player_index],
            players[player_index].with_card(finger_index, revealed_card_index),
            *players[player_index + 1 :],
        )

        # Check if we're done revealing cards. If we are, start the actual game
        # loop with the turn index reset to 0.
        turn_index += 1
        if turn_index == len(players):
            state = GameState.DRAW_OR_REPLACE_WITH_DISCARD
            turn_index = 0

            # If we have an `end_player_index` from a previous game, it should
            # already be assigned to `start_player_index` as well.
            if end_player_index is not None:
                assert start_player_index == end_player_index
                end_player_index = None

            # Otherwise, we're in the first round and choose the first player
            # based on who has the high sum of revealed cards.
            else:
                assert start_player_index == 0
                start_player_index = max(
                    range(len(players)),
                    key=lambda i: players[i].hand_score_revealed,
                )

        return Game(
            state=state,
            turn_index=turn_index,
            draw_pile=draw_pile,
            drawn_card_index=drawn_card_index,
            discard_pile=discard_pile,
            discarded_card_index=discarded_card_index,
            players=players,
            start_player_index=start_player_index,
            end_player_index=end_player_index,
        )

    def with_random_second_card_revealed(
        self,
        finger_index: int,
        rng: random.Random,
    ) -> Game:
        """Pick a random card from the draw pile to reveal."""

        return self.with_second_card_revealed(
            finger_index,
            _pick_random_card_index(self.draw_pile, rng),
        )

    # MARK: Draw

    def with_card_drawn(self, card_index: int) -> Game:
        """Draw a specific card from the pile but do nothing with it."""

        state = self.state
        turn_index = self.turn_index
        draw_pile = self.draw_pile
        drawn_card_index = self.drawn_card_index
        discard_pile = self.discard_pile
        discarded_card_index = self.discarded_card_index
        players = self.players
        start_player_index = self.start_player_index
        end_player_index = self.end_player_index
        del self

        assert state == GameState.DRAW_OR_REPLACE_WITH_DISCARD
        assert drawn_card_index is None
        assert sum(draw_pile) > 0
        assert discarded_card_index is not None

        # Remove the drawn card from the draw pile.
        draw_pile = _with_card_drawn(draw_pile, card_index)
        drawn_card_index = card_index
        state = GameState.DISCARD_DRAW_AND_REVEAL_OR_REPLACE_WITH_DRAW

        # Shuffle the discards into the draw pile if there are no more draws.
        # Always do this at the end of states, if possible, so we don't have
        # to juggle reshuffles before draws.
        draw_pile, discard_pile = _with_discard_pile_shuffled(draw_pile, discard_pile)

        return Game(
            state=state,
            turn_index=turn_index,
            draw_pile=draw_pile,
            drawn_card_index=drawn_card_index,
            discard_pile=discard_pile,
            discarded_card_index=discarded_card_index,
            players=players,
            start_player_index=start_player_index,
            end_player_index=end_player_index,
        )

    def with_random_drawn_card(self, rng: random.Random) -> Game:
        """Draw a random card from the pile but do nothing with it."""

        card_index = _pick_random_card_index(self.draw_pile, rng)
        return self.with_card_drawn(card_index)

    # MARK: Discard draw and reveal

    def with_draw_discarded_and_card_revealed(
        self,
        finger_index: int,
        revealed_card_index: int,
    ) -> Game:
        """Discard the drawn card and reveal a card in the current hand."""

        state = self.state
        turn_index = self.turn_index
        draw_pile = self.draw_pile
        drawn_card_index = self.drawn_card_index
        discard_pile = self.discard_pile
        discarded_card_index = self.discarded_card_index
        players = self.players
        start_player_index = self.start_player_index
        end_player_index = self.end_player_index
        del self

        assert state == GameState.DISCARD_DRAW_AND_REVEAL_OR_REPLACE_WITH_DRAW
        assert drawn_card_index is not None
        assert sum(draw_pile) > 0
        assert discarded_card_index is not None

        player_index = (start_player_index + turn_index) % len(players)
        assert players[player_index].hand[finger_index].is_hidden

        # Formally draw the revealed card.
        draw_pile = _with_card_drawn(draw_pile, revealed_card_index)

        # Reveal the requested card.
        finger_card_index = revealed_card_index
        players = (
            *players[:player_index],
            players[player_index].with_card(finger_index, finger_card_index),
            *players[player_index + 1 :],
        )

        # Put the current discarded card into the pile.
        discard_pile = _with_card_discarded(discard_pile, discarded_card_index)

        # Move the drawn card to be the discarded card.
        discarded_card_index = drawn_card_index
        drawn_card_index = None

        # If the card got cleared, we need to replace the current discarded
        # card and update the discard pile. This is a little wasteful but we'll
        # wait to profile before we inline it since it's rare.
        if players[player_index].hand[finger_index].is_cleared:
            discard_pile = _with_card_discarded(discard_pile, discarded_card_index)
            discard_pile = _with_card_discarded(discard_pile, finger_card_index)
            discard_pile = _with_card_discarded(discard_pile, finger_card_index)
            discarded_card_index = finger_card_index

        # Shuffle the discards into the draw pile if there are no more draws.
        # Always do this at the end of states, if possible, so we don't have
        # to juggle reshuffles before draws.
        draw_pile, discard_pile = _with_discard_pile_shuffled(draw_pile, discard_pile)

        # If we're not in the endgame and the current player has revealed all
        # their cards, start the endgame by setting `end_turn`.
        if end_player_index is None and players[player_index].is_hand_revealed:
            end_player_index = player_index

        turn_index += 1

        # If we're on the final turn of the endgame, transition states.
        if (player_index + 1) % len(players) == end_player_index:
            state = GameState.REVEAL_HIDDEN_CARDS
        else:
            state = GameState.DRAW_OR_REPLACE_WITH_DISCARD

        return Game(
            state=state,
            turn_index=turn_index,
            draw_pile=draw_pile,
            drawn_card_index=drawn_card_index,
            discard_pile=discard_pile,
            discarded_card_index=discarded_card_index,
            players=players,
            start_player_index=start_player_index,
            end_player_index=end_player_index,
        )

    def with_draw_discarded_and_random_card_revealed(
        self,
        finger_index: int,
        rng: random.Random,
    ) -> Game:
        """Pick a random card from the draw pile to reveal."""

        return self.with_draw_discarded_and_card_revealed(
            finger_index,
            _pick_random_card_index(self.draw_pile, rng),
        )

    # MARK: Replace with draw

    def with_card_replaced_with_draw(
        self,
        finger_index: int,
        revealed_card_index: int | None,  # only if finger is hidden
    ) -> Game:
        """Replace a card in the current hand with the drawn card."""

        state = self.state
        turn_index = self.turn_index
        draw_pile = self.draw_pile
        drawn_card_index = self.drawn_card_index
        discard_pile = self.discard_pile
        discarded_card_index = self.discarded_card_index
        players = self.players
        start_player_index = self.start_player_index
        end_player_index = self.end_player_index
        del self

        assert state == GameState.DISCARD_DRAW_AND_REVEAL_OR_REPLACE_WITH_DRAW
        assert drawn_card_index is not None
        assert discarded_card_index is not None

        player_index = (start_player_index + turn_index) % len(players)

        # Get the card being replaced. If it's hidden, we'll need to formally
        # draw the preordained revealed card.
        finger = players[player_index].hand[finger_index]
        old_finger_card_index = finger.card_index
        if old_finger_card_index is None:
            assert finger.is_hidden
            assert revealed_card_index is not None
            draw_pile = _with_card_drawn(draw_pile, revealed_card_index)
            old_finger_card_index = revealed_card_index
        else:
            assert revealed_card_index is None

        # Replace the card in the player's hand with the drawn card.
        finger_card_index = drawn_card_index
        drawn_card_index = None
        players = (
            *players[:player_index],
            players[player_index].with_card(finger_index, finger_card_index),
            *players[player_index + 1 :],
        )

        # Put the current discarded card into the pile.
        discard_pile = _with_card_discarded(discard_pile, discarded_card_index)

        # Move the card that was previously in hand to the discard.
        assert old_finger_card_index is not None
        discarded_card_index = old_finger_card_index

        # If the card got cleared, we need to replace the current discarded
        # card and update the discard pile. This is a little wasteful but we'll
        # wait to profile before we inline it since it's rare.
        if players[player_index].hand[finger_index].is_cleared:
            discard_pile = _with_card_discarded(discard_pile, discarded_card_index)
            discard_pile = _with_card_discarded(discard_pile, finger_card_index)
            discard_pile = _with_card_discarded(discard_pile, finger_card_index)
            discarded_card_index = finger_card_index

        # Shuffle the discards into the draw pile if there are no more draws.
        # Always do this at the end of states, if possible, so we don't have
        # to juggle reshuffles before draws.
        draw_pile, discard_pile = _with_discard_pile_shuffled(draw_pile, discard_pile)

        # If we're not in the endgame and the current player has revealed all
        # their cards, start the endgame by setting `end_turn`.
        if end_player_index is None and players[player_index].is_hand_revealed:
            end_player_index = player_index

        turn_index += 1

        # If we're on the final turn of the endgame, transition states.
        if (player_index + 1) % len(players) == end_player_index:
            state = GameState.REVEAL_HIDDEN_CARDS
        else:
            state = GameState.DRAW_OR_REPLACE_WITH_DISCARD

        return Game(
            state=state,
            turn_index=turn_index,
            draw_pile=draw_pile,
            drawn_card_index=drawn_card_index,
            discard_pile=discard_pile,
            discarded_card_index=discarded_card_index,
            players=players,
            start_player_index=start_player_index,
            end_player_index=end_player_index,
        )

    def with_random_card_replaced_with_draw(
        self,
        finger_index: int,
        rng: random.Random,
    ) -> Game:
        """Randomly pick if the replaced card is hidden."""

        return self.with_card_replaced_with_draw(
            finger_index,
            _pick_random_card_index(self.draw_pile, rng)
            if self.player.hand[finger_index].is_hidden
            else None,
        )

    # MARK: Replace with discard

    def with_card_replaced_with_discard(
        self,
        finger_index: int,
        revealed_card_index: int | None,
    ) -> Game:
        """Replace a card in the current hand with the drawn card."""

        state = self.state
        turn_index = self.turn_index
        draw_pile = self.draw_pile
        drawn_card_index = self.drawn_card_index
        discard_pile = self.discard_pile
        discarded_card_index = self.discarded_card_index
        players = self.players
        start_player_index = self.start_player_index
        end_player_index = self.end_player_index
        del self

        assert state == GameState.DRAW_OR_REPLACE_WITH_DISCARD
        assert drawn_card_index is None

        player_index = (start_player_index + turn_index) % len(players)

        # Move the replaced card to be the new discarded card. If the card was
        # hidden, it needs to be formally drawn.
        finger = players[player_index].hand[finger_index]
        old_finger_card_index = finger.card_index
        if old_finger_card_index is None:
            assert finger.is_hidden
            assert revealed_card_index is not None
            draw_pile = _with_card_drawn(draw_pile, revealed_card_index)
            old_finger_card_index = revealed_card_index
        else:
            assert revealed_card_index is None

        # Replace the card in the player's hand with the current discard.
        finger_card_index = discarded_card_index
        discarded_card_index = old_finger_card_index
        players = (
            *players[:player_index],
            players[player_index].with_card(finger_index, finger_card_index),
            *players[player_index + 1 :],
        )

        # If the card got cleared, we need to replace the current discarded
        # card and update the discard pile. This is a little wasteful but we'll
        # wait to profile before we inline it since it's rare.
        if players[player_index].hand[finger_index].is_cleared:
            discard_pile = _with_card_discarded(discard_pile, discarded_card_index)
            discard_pile = _with_card_discarded(discard_pile, finger_card_index)
            discard_pile = _with_card_discarded(discard_pile, finger_card_index)
            discarded_card_index = finger_card_index

        # Shuffle the discards into the draw pile if there are no more draws.
        # Always do this at the end of states, if possible, so we don't have
        # to juggle reshuffles before draws.
        draw_pile, discard_pile = _with_discard_pile_shuffled(draw_pile, discard_pile)

        # If we're not in the endgame and the current player has revealed all
        # their cards, start the endgame by setting `end_turn`.
        if end_player_index is None and players[player_index].is_hand_revealed:
            end_player_index = player_index

        turn_index += 1

        # If we're on the final turn of the endgame, transition states.
        if (player_index + 1) % len(players) == end_player_index:
            state = GameState.REVEAL_HIDDEN_CARDS
        else:
            state = GameState.DRAW_OR_REPLACE_WITH_DISCARD

        return Game(
            state=state,
            turn_index=turn_index,
            draw_pile=draw_pile,
            drawn_card_index=drawn_card_index,
            discard_pile=discard_pile,
            discarded_card_index=discarded_card_index,
            players=players,
            start_player_index=start_player_index,
            end_player_index=end_player_index,
        )

    def with_random_card_replaced_with_discard(
        self,
        finger_index: int,
        rng: random.Random,
    ) -> Game:
        """Randomly pick if the replaced card is hidden."""

        return self.with_card_replaced_with_discard(
            finger_index,
            _pick_random_card_index(self.draw_pile, rng)
            if self.player.hand[finger_index].is_hidden
            else None,
        )

    # MARK: Forfeit

    def with_forfeit(self) -> Game:
        """Give up as the current player, forcing a loss."""

        state = self.state
        turn_index = self.turn_index
        draw_pile = self.draw_pile
        drawn_card_index = self.drawn_card_index
        discard_pile = self.discard_pile
        discarded_card_index = self.discarded_card_index
        players = self.players
        start_player_index = self.start_player_index
        end_player_index = self.end_player_index
        del self

        assert state == GameState.DRAW_OR_REPLACE_WITH_DISCARD
        assert drawn_card_index is None
        assert end_player_index is None

        player_index = (start_player_index + turn_index) % len(players)

        # Set the end turn to begin the endgame.
        end_player_index = player_index

        # Rotate the players and increment the turn.
        turn_index += 1

        return Game(
            state=state,
            turn_index=turn_index,
            draw_pile=draw_pile,
            drawn_card_index=drawn_card_index,
            discard_pile=discard_pile,
            discarded_card_index=discarded_card_index,
            players=players,
            start_player_index=start_player_index,
            end_player_index=end_player_index,
        )

    # MARK: Reveal remaining

    def with_hidden_cards_revealed(self, revealed_card_indices: Iterable[int]) -> Game:
        """Draw hidden cards in player hands at the end of the game."""

        state = self.state
        turn_index = self.turn_index
        draw_pile = self.draw_pile
        drawn_card_index = self.drawn_card_index
        discard_pile = self.discard_pile
        discarded_card_index = self.discarded_card_index
        players = self.players
        start_player_index = self.start_player_index
        end_player_index = self.end_player_index
        del self

        assert state == GameState.REVEAL_HIDDEN_CARDS
        assert drawn_card_index is None

        player_index = (start_player_index + turn_index) % len(players)

        mutable_draw_pile = list(draw_pile)

        def deal_revealed_card_index_from_draw_pile() -> Iterator[int]:
            for revealed_card_index in revealed_card_indices:
                assert mutable_draw_pile[revealed_card_index] > 0
                mutable_draw_pile[revealed_card_index] -= 1
                yield revealed_card_index

        dealer = deal_revealed_card_index_from_draw_pile()

        is_forfeit = not players[player_index].is_hand_revealed
        players = tuple(player.with_hidden_cards_revealed(dealer) for player in players)
        draw_pile = tuple(mutable_draw_pile)

        scores = [player.hand_score for player in players]
        if is_forfeit or scores[player_index] >= min(scores[1:]):
            scores[player_index] *= 2

        players = tuple(
            player.with_score(player.score + score)
            for player, score in zip(players, scores)
        )

        state = GameState.DEAL_FIRST_CARDS

        return Game(
            state=state,
            turn_index=turn_index,
            draw_pile=draw_pile,
            drawn_card_index=drawn_card_index,
            discard_pile=discard_pile,
            discarded_card_index=discarded_card_index,
            players=players,
            start_player_index=start_player_index,
            end_player_index=end_player_index,
        )

    def with_random_hidden_cards_revealed(self, rng: random.Random) -> Game:
        """Draw hidden cards randomly."""

        state = self.state
        turn_index = self.turn_index
        draw_pile = self.draw_pile
        drawn_card_index = self.drawn_card_index
        discard_pile = self.discard_pile
        discarded_card_index = self.discarded_card_index
        players = self.players
        start_player_index = self.start_player_index
        end_player_index = self.end_player_index
        del self

        assert state == GameState.REVEAL_HIDDEN_CARDS
        assert drawn_card_index is None

        player_index = (start_player_index + turn_index) % len(players)

        mutable_draw_pile = list(draw_pile)

        def deal_revealed_card_index_from_draw_pile() -> Iterator[int]:
            while True:
                revealed_card_index = _pick_random_card_index(mutable_draw_pile, rng)
                mutable_draw_pile[revealed_card_index] -= 1
                yield revealed_card_index

        dealer = deal_revealed_card_index_from_draw_pile()

        is_forfeit = not players[player_index].is_hand_revealed
        players = tuple(player.with_hidden_cards_revealed(dealer) for player in players)
        draw_pile = tuple(mutable_draw_pile)

        scores = [player.hand_score for player in players]
        if is_forfeit or scores[player_index] >= min(scores[1:]):
            scores[player_index] *= 2

        players = tuple(
            player.with_score(player.score + score)
            for player, score in zip(players, scores)
        )

        state = GameState.DEAL_FIRST_CARDS

        return Game(
            state=state,
            turn_index=turn_index,
            draw_pile=draw_pile,
            drawn_card_index=drawn_card_index,
            discard_pile=discard_pile,
            discarded_card_index=discarded_card_index,
            players=players,
            start_player_index=start_player_index,
            end_player_index=end_player_index,
        )
