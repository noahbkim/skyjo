import dataclasses
import enum
import random
import typing
from functools import cached_property
from typing import Self

import numpy as np

NUM_CARD_TYPES = 15
NUM_ROWS = 3
NUM_COLUMNS = 4
DECK_SIZE = 150
DECK = (
    (-2,) * 5
    + (-1,) * 10
    + (0,) * 15
    + (1,) * 10
    + (2,) * 10
    + (3,) * 10
    + (4,) * 10
    + (5,) * 10
    + (6,) * 10
    + (7,) * 10
    + (8,) * 10
    + (9,) * 10
    + (10,) * 10
    + (11,) * 10
    + (12,) * 10
)


class SkyjoActionType(enum.Enum):
    DRAW = "DRAW"
    PLACE_DRAWN = "PLACE_DRAWN"
    PLACE_FROM_DISCARD = "PLACE_DISCARD"
    DISCARD_AND_FLIP = "DISCARD_AND_FLIP"


@dataclasses.dataclass(slots=True, frozen=True)
class SkyjoAction:
    action_type: SkyjoActionType = dataclasses.field()
    row_idx: int | None = None
    col_idx: int | None = None


@dataclasses.dataclass(slots=True, frozen=True)
class Hand:
    is_cleared: np.array = dataclasses.field(
        default_factory=lambda: np.zeros(NUM_ROWS * NUM_COLUMNS)
    )
    is_flipped: np.array = dataclasses.field(
        default_factory=lambda: np.zeros(NUM_ROWS * NUM_COLUMNS)
    )
    cards: np.array = dataclasses.field(
        default_factory=lambda: np.zeros(NUM_ROWS * NUM_COLUMNS)
    )

    @property
    def face_down_indices(self) -> list[tuple[int, int]]:
        return [
            (i // NUM_COLUMNS, i % NUM_COLUMNS)
            for i in np.where((self.is_cleared == 0) & (self.is_flipped == 0))[0]
        ]

    @property
    def visible_indices(self) -> list[tuple[int, int]]:
        return [
            (i // NUM_COLUMNS, i % NUM_COLUMNS)
            for i in np.where((self.is_cleared == 0) & (self.is_flipped == 1))[0]
        ]

    @property
    def valid_discard_and_flip_actions(self) -> list[SkyjoAction]:
        actions = [
            SkyjoAction(SkyjoActionType.DISCARD_AND_FLIP, row_idx, col_idx)
            for row_idx, col_idx in self.face_down_indices
        ]
        return actions

    @property
    def valid_place_from_discard_actions(self) -> list[SkyjoAction]:
        actions = [
            SkyjoAction(SkyjoActionType.PLACE_FROM_DISCARD, row_idx, col_idx)
            for row_idx, col_idx in self.visible_indices + self.face_down_indices
        ]
        return actions

    @property
    def valid_place_drawn_actions(self) -> list[SkyjoAction]:
        actions = [
            SkyjoAction(SkyjoActionType.PLACE_DRAWN, row_idx, col_idx)
            for row_idx, col_idx in self.visible_indices + self.face_down_indices
        ]
        return actions

    @staticmethod
    def clearable(cards: np.array, is_flipped: np.array, col_idx: int):
        """This is a static method so we can check whether the updated Hand
        we are about to create needs to have any cleared columns, preventing us
        from creating any intermediate Hands with incorrect is_cleared values."""
        all_same = True
        for r in range(NUM_ROWS):
            all_same = (
                all_same
                and is_flipped[Hand.flat_index(r, col_idx)]
                and cards[Hand.flat_index(0, col_idx)]
                == cards[Hand.flat_index(r, col_idx)]
            )
        return all_same

    def has_all_revealed(self) -> bool:
        # To be cleared the is_flipped value would also be set to 1, so we can just
        # count how many flipped cards there are without worrying about cleared counts.
        return self.is_flipped.sum() == NUM_COLUMNS * NUM_ROWS

    def flat_index(row_idx: int, col_idx: int) -> int:
        assert row_idx < NUM_ROWS and row_idx >= 0, "row index out of range"
        assert col_idx < NUM_COLUMNS and col_idx >= 0, "col index out of range"
        return row_idx * NUM_COLUMNS + col_idx

    def copy(self) -> Self:
        return Hand(
            is_cleared=self.is_cleared.copy(),
            is_flipped=self.is_flipped.copy(),
            cards=self.cards.copy(),
        )

    def flip(self, row_idx: int, col_idx: int) -> Self:
        assert (
            self.is_flipped[Hand.flat_index(row_idx, col_idx)] != 1
        ), "Cannot flip already flipped card"
        assert (
            self.is_cleared[Hand.flat_index(row_idx, col_idx)] != 1
        ), "Cannot flipped cleared card"
        flipped = self.is_flipped.copy()
        cleared = self.is_cleared.copy()
        flipped[Hand.flat_index(row_idx, col_idx)] = 1

        # Check if flipping led to clearable column
        if Hand.clearable(self.cards, flipped, col_idx):
            for r in range(NUM_ROWS):
                cleared[Hand.flat_index(r, col_idx)] = 1
        return Hand(is_flipped=flipped, is_cleared=cleared, cards=self.cards.copy())

    def replace(self, row_idx: int, col_idx: int, card: int) -> tuple[int, Self]:
        flipped = self.is_flipped.copy()
        cleared = self.is_cleared.copy()
        flat_idx = Hand.flat_index(row_idx, col_idx)
        cards = self.cards.copy()
        replaced_card = cards[flat_idx]
        cards[flat_idx] = card
        flipped[flat_idx] = 1

        if Hand.clearable(cards, flipped, col_idx):
            for r in range(NUM_ROWS):
                cleared[Hand.flat_index(r, col_idx)] = 1

        return replaced_card, Hand(is_flipped=flipped, is_cleared=cleared, cards=cards)

    def visible_points(self) -> int:
        return self.cards[
            np.where((self.is_cleared == 0) & (self.is_flipped == 1))
        ].sum()

    def total_points(self) -> int:
        """Returns the value of the hand after revealing all cards.
        This includes any previously undiscovered clearing columns"""
        assert (
            self.is_cleared.sum() % NUM_ROWS == 0
        ), f"somehow cleared cards not a multiple of {NUM_ROWS}"
        revealed_hand = self.copy()
        while len(revealed_hand.face_down_indices) > 0:
            revealed_hand = revealed_hand.flip(*revealed_hand.face_down_indices[0])
        return revealed_hand.cards[np.where(revealed_hand.is_cleared == 0)].sum()

    def initial_flips(self) -> Self:
        """This is just a temporary solution.
        So that we don't have to additionally code initial flip action/decisions"""
        assert (
            len(self.face_down_indices) == NUM_COLUMNS * NUM_ROWS
        ), "Cannot run initial_flips() when there are already flipped cards"
        return self.flip(0, 0).flip(0, 1)

    def _render(self, xray: bool = False) -> list[str]:
        card_width = max(max(len(str(card)) for card in self.cards), 2)
        empty = " " * card_width
        divider = ("+" + "-" * card_width) * NUM_COLUMNS + "+"
        template = f"|{{:>{card_width}}}" * NUM_COLUMNS + "|"
        lines = [divider] * (NUM_ROWS * 2 + 1)
        lines[1::2] = (
            template.format(
                *(
                    str(card)
                    if (
                        self.is_flipped[r * NUM_COLUMNS + c]
                        and not self.is_cleared[r * NUM_COLUMNS + c]
                    )
                    or xray
                    else empty
                    for c, card in enumerate(
                        self.cards[r * NUM_COLUMNS : (r + 1) * NUM_COLUMNS]
                    )
                )
            )
            for r in range(NUM_ROWS)
        )
        return lines


@dataclasses.dataclass(slots=True, frozen=True)
class Deck:
    _cards: typing.Collection[int] = dataclasses.field(default=None)
    _rng: random.Random = dataclasses.field(default_factory=random.Random)

    def deal_hands(self, num_hands: int) -> tuple[list[Hand], Self]:
        hands = []
        for i in range(num_hands):
            hands.append(
                Hand(
                    cards=np.array(
                        self._cards[
                            i * NUM_ROWS * NUM_COLUMNS : (i + 1)
                            * NUM_ROWS
                            * NUM_COLUMNS
                        ]
                    )
                )
            )
        next_cards = self._cards[num_hands * NUM_ROWS * NUM_COLUMNS :]
        return hands, Deck(_cards=next_cards, _rng=self._rng)

    def draw_top_card(self) -> tuple[int, Self]:
        if len(self._cards) == 1:
            next_cards = list(DECK)
            self._rng.shuffle(next_cards)
        else:
            next_cards = list(self._cards[1:])
        return self._cards[0], Deck(_cards=next_cards, _rng=self._rng)

    def shuffle(self) -> Self:
        next_cards = list(self._cards)
        self._rng.shuffle(next_cards)
        return Deck(_cards=next_cards, _rng=self._rng)

    def copy(self) -> Self:
        return Deck(_cards=list(self._cards), _rng=self._rng)


@dataclasses.dataclass(slots=True, frozen=True)
class DiscardPile:
    discarded_card_counts: np.array = dataclasses.field(
        default_factory=lambda: np.zeros(NUM_CARD_TYPES)
    )
    top_card: typing.Optional[int] = None

    def copy(self) -> Self:
        return DiscardPile(
            discarded_card_counts=self.discarded_card_counts.copy(),
            top_card=self.top_card,
        )

    def discard(self, card: int) -> Self:
        next_discarded_card_counts = self.discarded_card_counts.copy()
        next_discarded_card_counts[card + 2] += 1
        return dataclasses.replace(
            self, discarded_card_counts=next_discarded_card_counts, top_card=card
        )

    def replace_top_card(self, card: int) -> Self:
        next_discarded_card_counts = self.discarded_card_counts.copy()
        next_discarded_card_counts[self.top_card + 2] -= 1
        next_discarded_card_counts[card + 2] += 1
        return dataclasses.replace(
            self, discarded_card_counts=next_discarded_card_counts, top_card=card
        )


@dataclasses.dataclass(slots=True, frozen=True)
class ImmutableSkyjoState:
    num_players: int
    """Number of players in the game"""
    player_scores: np.array
    """Current player scores"""
    hands: list[Hand] = dataclasses.field(default=None)
    """The list of participating players."""
    deck: Deck = dataclasses.field(default=None)
    """Deck of cards"""
    discard_pile: DiscardPile = dataclasses.field(default_factory=DiscardPile)
    """Discard Pile Object"""
    drawn_card: int | None = None
    """Card drawn (if applicable)"""
    curr_player: int = 0
    """Index of current player"""
    turn_count: int = 0
    """Number of turns taken so far"""
    round_count: int = 0
    """Number of rounds so far"""
    is_round_ending: bool = 0
    """If this is the last turn for each player"""
    round_ending_player: int | None = None
    """Index of player who has ended the game"""
    valid_actions: list[SkyjoAction] = dataclasses.field(default_factory=list)
    """List of valid actions available from current state"""
    winning_player: int | None = None
    """Index of player who won the game"""

    @staticmethod
    def compute_round_scores(
        hands: list[Hand], round_ending_player_idx: int
    ) -> np.array:
        round_scores = np.zeros(len(hands))
        for player_idx in range(len(hands)):
            round_scores[player_idx] += hands[player_idx].total_points()
        if (
            round_scores[round_ending_player_idx]
            >= np.concat(
                [
                    round_scores[:round_ending_player_idx],
                    round_scores[round_ending_player_idx + 1 :],
                ],
            ).min()
        ):
            round_scores[round_ending_player_idx] *= 2
        return round_scores

    def setup_round(self) -> Self:
        next_deck = Deck(_cards=DECK, _rng=self.deck._rng).shuffle()
        next_hands, next_deck = next_deck.deal_hands(self.num_players)
        top_card, next_deck = next_deck.draw_top_card()
        next_hands = [hand.initial_flips() for hand in next_hands]
        starting_player = self.round_ending_player
        # if first round of game set starting player to player with highest showing value
        if self.round_ending_player is None:
            hand_values = [hand.visible_points() for hand in next_hands]
            starting_player = hand_values.index(max(hand_values))
        next_valid_actions = next_hands[
            starting_player
        ].valid_place_from_discard_actions + [SkyjoAction(SkyjoActionType.DRAW)]
        next_state = dataclasses.replace(
            self,
            player_scores=self.player_scores.copy(),
            hands=next_hands,
            deck=next_deck,
            discard_pile=DiscardPile().discard(top_card),
            drawn_card=None,
            curr_player=starting_player,
            turn_count=1,
            round_count=self.round_count + 1,
            is_round_ending=False,
            round_ending_player=None,
            valid_actions=next_valid_actions,
        )
        return next_state

    def take_action(self, action: SkyjoAction) -> Self:
        next_player_scores = self.player_scores.copy()
        next_hands = [hand.copy() for hand in self.hands]
        next_deck = self.deck.copy()
        next_drawn_card = None
        next_round_count = self.round_count
        next_turn_count = self.turn_count + 1
        next_player = (self.curr_player + 1) % self.num_players

        if action.action_type == SkyjoActionType.DRAW:
            next_discard_pile = self.discard_pile.copy()
            next_drawn_card, next_deck = self.deck.draw_top_card()
            next_turn_count = self.turn_count
            next_player = self.curr_player
            next_valid_actions = (
                self.hands[self.curr_player].valid_discard_and_flip_actions
                + self.hands[self.curr_player].valid_place_drawn_actions
            )

        elif action.action_type == SkyjoActionType.PLACE_DRAWN:
            replaced_card, next_hand = next_hands[self.curr_player].replace(
                action.row_idx, action.col_idx, self.drawn_card
            )
            next_hands[self.curr_player] = next_hand
            next_discard_pile = self.discard_pile.discard(replaced_card)
            next_valid_actions = next_hands[
                next_player
            ].valid_place_from_discard_actions + [SkyjoAction(SkyjoActionType.DRAW)]

        elif action.action_type == SkyjoActionType.DISCARD_AND_FLIP:
            next_discard_pile = self.discard_pile.discard(self.drawn_card)
            next_hands[self.curr_player] = next_hands[self.curr_player].flip(
                action.row_idx, action.col_idx
            )
            next_valid_actions = next_hands[
                next_player
            ].valid_place_from_discard_actions + [SkyjoAction(SkyjoActionType.DRAW)]

        elif action.action_type == SkyjoActionType.PLACE_FROM_DISCARD:
            replaced_card, next_hand = next_hands[self.curr_player].replace(
                action.row_idx, action.col_idx, self.discard_pile.top_card
            )
            next_hands[self.curr_player] = next_hand
            next_discard_pile = self.discard_pile.replace_top_card(replaced_card)
            next_valid_actions = next_hands[
                next_player
            ].valid_place_from_discard_actions + [SkyjoAction(SkyjoActionType.DRAW)]

        else:
            raise ValueError("This is not a supported action type")

        has_all_revealed = next_hands[self.curr_player].has_all_revealed()
        next_round_ending_player = self.round_ending_player
        if has_all_revealed and not self.is_round_ending:
            next_round_ending_player = self.curr_player

        next_state = dataclasses.replace(
            self,
            player_scores=next_player_scores,
            hands=next_hands,
            deck=next_deck,
            discard_pile=next_discard_pile,
            drawn_card=next_drawn_card,
            curr_player=next_player,
            turn_count=next_turn_count,
            round_count=next_round_count,
            is_round_ending=(self.is_round_ending or has_all_revealed),
            round_ending_player=next_round_ending_player,
            valid_actions=next_valid_actions,
        )

        if (
            next_state.is_round_ending
            and next_state.curr_player == next_state.round_ending_player
        ):
            round_scores = ImmutableSkyjoState.compute_round_scores(
                next_state.hands, next_state.round_ending_player
            )
            next_state = dataclasses.replace(
                next_state, player_scores=(next_state.player_scores + round_scores)
            ).setup_round()
            if max(next_state.player_scores) >= 100:
                next_state = dataclasses.replace(
                    next_state, winning_player=np.argmin(next_state.player_scores)
                )

        return next_state

    @cached_property
    def numpy_repr(self) -> np.array:
        pass

    def __hash__(self):
        return self.numpy_repr.tobytes()

    def _render(self, xray: bool = False) -> list[str]:
        first_line = (
            f"discard: {self.discard_pile.top_card}"
            f" drawn card: {self.drawn_card}"
            f"  round: {self.round_count}"
            f"  turn: {self.turn_count}"
            f" player_scores: {self.player_scores}"
            + ("  (ending)" if self.is_round_ending else "")
            + (f"  ({self.round_ending_player})" if self.is_round_ending else "")
        )

        hands_render = [""]
        for player_idx in range(self.num_players):
            hand_render = self.hands[player_idx]._render(xray=xray)
            base_length = len(
                hands_render[0]
            )  # needed to shift ^ player turn indicator by correct number of spaces
            if player_idx == self.curr_player:
                hand_render.append("^" * len(hand_render[-1]))
            for i, line in enumerate(hand_render):
                if i == len(hands_render):
                    hands_render.append(" " * base_length)
                if player_idx > 0:
                    hands_render[i] += "  "
                hands_render[i] += line

        return [
            first_line,
            *hands_render,
        ]

    def display(self, xray: bool = False) -> None:
        symbol = "#"
        for line in self._render(xray):
            print(symbol, line)
            symbol = ""


class RandomPlayer:
    def __init__(self, name):
        self.name = name

    def decide_action(self, state: ImmutableSkyjoState) -> SkyjoAction:
        if len(state.valid_actions) == 0:
            raise RuntimeError("no valid actions")
        return state.valid_actions[random.randint(0, len(state.valid_actions) - 1)]
