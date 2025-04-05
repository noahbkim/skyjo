"""
Implementation of the Skyjo game

Key features:
- Immutable game state
- Abstracts random events (e.g. drawing a card from the deck) by taking
  results of random events as arguments
- Numpy representation of game state and actions
"""

import dataclasses
import enum
import logging
import typing

import numpy as np
import numpy.typing as npt

import abstract as abstract

GAME_END_SCORE = 100
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
ACTION_SHAPE = (7, NUM_ROWS, NUM_COLUMNS)  # 6 action types, rows, columns
CARD_TYPES = 15 + 2  # +2 for face-down and cleared
HAND_SHAPE = (
    NUM_COLUMNS,
    NUM_ROWS,
    CARD_TYPES,
)  # Each column is independent so we index based on column


class SkyjoActionType(enum.Enum):
    # Enum values are used as indices in numpy representation
    DRAW = 0
    PLACE_DRAWN = 1
    PLACE_FROM_DISCARD = 2
    DISCARD_AND_FLIP = 3
    # These are "dummy" actions
    # that are used to help transition from a decision state to an after state
    # during simulation since we need an "action" to be associated with the game state for afterstates
    END_ROUND = 4
    START_ROUND = 5
    INITIAL_FLIP = 6


@dataclasses.dataclass(slots=True, frozen=True)
class SkyjoAction(abstract.AbstractGameAction):
    action_type: SkyjoActionType = dataclasses.field()
    row_idx: int | None = None
    col_idx: int | None = None

    def __hash__(self):
        return hash(self.numpy().tobytes())

    def __eq__(self, other):
        if isinstance(other, SkyjoAction):
            return self.__hash__() == other.__hash__()
        return False

    @classmethod
    def from_numpy(cls, numpy_repr: npt.NDArray[np.int8]) -> typing.Self:
        assert numpy_repr.shape == ACTION_SHAPE, (
            f"Invalid action shape, expected: {ACTION_SHAPE}, got: {numpy_repr.shape}"
        )
        if numpy_repr.dtype != np.int8:
            logging.warning("Casting action numpy array to int8")
        numpy_repr = numpy_repr.astype(np.int8)
        assert np.sum(numpy_repr) == 1, (
            f"Action must be one-hot encoded, got: {numpy_repr}"
        )

        if numpy_repr[SkyjoActionType.DRAW.value].sum() == 1:
            return cls(SkyjoActionType.DRAW)
        elif numpy_repr[SkyjoActionType.END_ROUND.value].sum() == 1:
            return cls(SkyjoActionType.END_ROUND)
        elif numpy_repr[SkyjoActionType.PLACE_DRAWN.value].sum() == 1:
            action_type = SkyjoActionType.PLACE_DRAWN
        elif numpy_repr[SkyjoActionType.PLACE_FROM_DISCARD.value].sum() == 1:
            action_type = SkyjoActionType.PLACE_FROM_DISCARD
        elif numpy_repr[SkyjoActionType.DISCARD_AND_FLIP.value].sum() == 1:
            action_type = SkyjoActionType.DISCARD_AND_FLIP
        else:
            raise ValueError(f"Invalid action: {numpy_repr}")
        row_idx = np.where(numpy_repr[action_type.value, :] == 1)[0].item()
        col_idx = np.where(numpy_repr[action_type.value, :] == 1)[1].item()
        return cls(action_type, row_idx, col_idx)

    def numpy(self) -> npt.NDArray[np.int8]:
        np_repr = np.zeros(ACTION_SHAPE, dtype=np.int8)
        match self.action_type:
            case SkyjoActionType.DRAW:
                np_repr[self.action_type.value, 0, 0] = 1
            case SkyjoActionType.END_ROUND:
                np_repr[self.action_type.value, 0, 0] = 1
            case SkyjoActionType.INITIAL_FLIP:
                np_repr[self.action_type.value, self.row_idx, self.col_idx] = 1
            case _:
                np_repr[self.action_type.value, self.row_idx, self.col_idx] = 1
        return np_repr


class Card(enum.Enum):
    # Value is index of card in one-hot encoding
    NEGATIVE_TWO = -2
    NEGATIVE_ONE = -1
    ZERO = 0
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    ELEVEN = 11
    TWELVE = 12
    FACE_DOWN = 13
    CLEARED = 14

    def __str__(self) -> str:
        match self:
            case Card.FACE_DOWN:
                return ""
            case Card.CLEARED:
                return "X"
            case _:
                return str(self.value)

    @classmethod
    def from_one_hot_encoding_index(cls, index: int) -> typing.Self:
        return cls(index - 2)

    @classmethod
    def from_one_hot_encoding(cls, one_hot: npt.NDArray[np.int8]) -> typing.Self:
        assert np.sum(one_hot) == 1, (
            f"One-hot encoding must be a single 1, got: {one_hot}"
        )
        assert one_hot.shape == (CARD_TYPES,), (
            f"One-hot encoding must be of shape (CARD_TYPES,), but got {one_hot.shape}"
        )
        idx = np.argmax(one_hot)
        return cls.from_one_hot_encoding_index(idx)

    @staticmethod
    def point_values() -> npt.NDArray[np.int8]:
        return np.array([card.point_value for card in Card], dtype=np.int8)

    @property
    def one_hot_encoding_index(self) -> int:
        return self.value + 2

    @property
    def point_value(self) -> int:
        match self:
            case Card.FACE_DOWN:
                return 0
            case Card.CLEARED:
                return 0
            case _:
                return self.value

    @property
    def is_clearable(self) -> bool:
        return self != Card.CLEARED and self != Card.FACE_DOWN

    def one_hot_encoding(self) -> npt.NDArray[np.int8]:
        """Returns a shape (CARD_TYPES,) numpy array one-hot encoded representation of the card"""
        one_hot = np.zeros(len(Card), dtype=np.int8)
        one_hot[self.one_hot_encoding_index] = 1
        return one_hot


@dataclasses.dataclass(slots=True, frozen=True)
class CardCounts:
    # Shape (CARD_TYPES,)
    _counts: npt.NDArray[np.int8]

    @classmethod
    def create_initial_deck_counts(cls) -> typing.Self:
        _initial_counts = np.ones(CARD_TYPES, dtype=np.int8) * 10
        _initial_counts[Card.ZERO.one_hot_encoding_index] = 15
        _initial_counts[Card.NEGATIVE_ONE.one_hot_encoding_index] = 10
        _initial_counts[Card.NEGATIVE_TWO.one_hot_encoding_index] = 5
        _initial_counts[Card.FACE_DOWN.one_hot_encoding_index] = 0
        _initial_counts[Card.CLEARED.one_hot_encoding_index] = 0
        return CardCounts(_counts=_initial_counts)

    def __post_init__(self):
        assert self._counts.shape == (CARD_TYPES,), (
            f"Counts must be of shape (CARD_TYPES,), but got {self._counts.shape}"
        )

    @property
    def num_cards(self) -> int:
        return np.sum(self._counts).item()

    @property
    def total_points(self) -> int:
        return np.dot(
            self._counts.astype(np.int16),  # counts of visible card types,
            Card.point_values(),  # point value of card types
        )

    @property
    def expected_value(self) -> float:
        return self.total_points / self.num_cards

    def copy(self) -> typing.Self:
        return CardCounts(self._counts.copy())

    def get_card_count(self, card: Card) -> int:
        return self._counts[card.one_hot_encoding_index]

    def remove_card(self, card: Card) -> typing.Self:
        new_counts = self._counts.copy()
        new_counts[card.one_hot_encoding_index] -= 1
        return CardCounts(_counts=new_counts)

    def remove_cards(self, cards: list[Card]) -> typing.Self:
        new_counts = self._counts.copy()
        for card in cards:
            new_counts[card.one_hot_encoding_index] -= 1
        return CardCounts(_counts=new_counts)

    def add_card(self, card: Card) -> typing.Self:
        new_counts = self._counts.copy()
        new_counts[card.one_hot_encoding_index] += 1
        return CardCounts(_counts=new_counts)

    def numpy(self) -> npt.NDArray[np.int8]:
        """Returns a numpy array representation of shape (CARD_TYPES, )"""
        return self._counts.copy()

    def generate_random_card(self) -> Card:
        card_one_hot_idx = np.random.choice(
            np.arange(CARD_TYPES), p=self._counts / self._counts.sum()
        )
        card = Card.from_one_hot_encoding_index(card_one_hot_idx)
        return card

    def generate_random_cards(self, num_cards: int) -> list[Card]:
        assert num_cards <= self.num_cards, (
            f"Cannot generate {num_cards} cards when only {self.num_cards} remain"
        )
        # Create list of all possible cards weighted by their counts
        all_cards = []
        for card in Card:
            count = self.get_card_count(card)
            all_cards.extend([card] * count)
        card_one_hot_idxs = np.random.choice(len(all_cards), num_cards, replace=False)
        return [all_cards[idx] for idx in card_one_hot_idxs]


@dataclasses.dataclass(slots=True, frozen=True)
class DiscardPile:
    discarded_card_counts: CardCounts = dataclasses.field(
        default_factory=lambda: CardCounts(_counts=np.zeros(CARD_TYPES, dtype=np.int8))
    )
    top_card: Card | None = None

    def copy(self) -> typing.Self:
        return DiscardPile(
            discarded_card_counts=self.discarded_card_counts.copy(),
            top_card=self.top_card,
        )

    def discard(self, card: Card) -> typing.Self:
        next_discarded_card_counts = self.discarded_card_counts.add_card(card)
        return DiscardPile(
            discarded_card_counts=next_discarded_card_counts, top_card=card
        )

    def replace_top_card(self, card: Card) -> typing.Self:
        next_discarded_card_counts = self.discarded_card_counts.remove_card(
            self.top_card
        ).add_card(card)
        return DiscardPile(
            discarded_card_counts=next_discarded_card_counts, top_card=card
        )

    def numpy(self) -> npt.NDArray[np.int8]:
        return np.concatenate(
            [
                self.discarded_card_counts.numpy(),
                self.top_card.one_hot_encoding()
                if self.top_card is not None
                else np.zeros(CARD_TYPES, dtype=np.int8),
            ]
        )


@dataclasses.dataclass(slots=True, frozen=True)
class Hand:
    # Array of one-hot encoded cards with shape (NUM_COLUMNS, NUM_ROWS, CARD_TYPES)
    _card_one_hots: npt.NDArray[np.int8]
    # One-hot encoded cards that have been cleared (NUM_COLUMNS, CARD_TYPES)
    _cleared_one_hots: npt.NDArray[np.int8]
    # Only not None if a column was just cleared before creation
    cleared_card: Card | None = None

    @classmethod
    def create_initial_hand(cls) -> typing.Self:
        return cls(
            _card_one_hots=np.array(
                [
                    Card.FACE_DOWN.one_hot_encoding()
                    for _ in range(NUM_ROWS * NUM_COLUMNS)
                ]
            ).reshape(NUM_COLUMNS, NUM_ROWS, CARD_TYPES),
            _cleared_one_hots=np.zeros((NUM_COLUMNS, CARD_TYPES), dtype=np.int8),
        )

    @staticmethod
    def clearable_columns(board_as_card_one_hots: npt.NDArray[np.int8]) -> list[int]:
        """Returns list of columns that are clearable"""
        assert board_as_card_one_hots.shape == HAND_SHAPE, (
            f"Invalid cards shape: {board_as_card_one_hots.shape}, but expected: {HAND_SHAPE}"
        )
        # find all columns with 3 of the same card type
        clearable_indices = np.argwhere(board_as_card_one_hots.sum(axis=1) == 3)
        return [
            col_idx
            for col_idx, card_one_hot_idx in clearable_indices
            if Card.from_one_hot_encoding_index(
                card_one_hot_idx
            ).is_clearable  # if not face-down or already cleared
        ]

    @property
    def cards(self) -> list[list[Card]]:
        """Returns a list of lists of cards of shape (NUM_ROWS, NUM_COLUMNS)"""
        return [
            [
                Card.from_one_hot_encoding_index(
                    np.argwhere(self._card_one_hots[col_idx, row_idx, :] == 1)
                )
                for col_idx in range(self._card_one_hots.shape[0])
            ]
            for row_idx in range(self._card_one_hots.shape[1])
        ]

    @property
    def num_face_down_cards(self) -> int:
        # To be cleared the is_flipped value would also be set to 1, so we can just
        # count how many flipped cards there are without worrying about cleared counts.
        return self.visible_card_types.get_card_count(Card.FACE_DOWN)

    @property
    def visible_card_types(self) -> CardCounts:
        return CardCounts(np.sum(self._card_one_hots, axis=(0, 1), dtype=np.int8))

    @property
    def visible_points(self) -> int:
        return self.visible_card_types.total_points

    @property
    def face_down_indices(self) -> list[tuple[int, int]]:
        return [
            (row_idx, col_idx)
            for col_idx in range(NUM_COLUMNS)
            for row_idx in range(NUM_ROWS)
            if Card.from_one_hot_encoding(self._card_one_hots[col_idx, row_idx])
            == Card.FACE_DOWN
        ]

    @property
    def cleared_indices(self) -> list[tuple[int, int]]:
        return [
            (row_idx, col_idx)
            for col_idx in range(NUM_COLUMNS)
            for row_idx in range(NUM_ROWS)
            if Card.from_one_hot_encoding(self._card_one_hots[col_idx, row_idx])
            == Card.CLEARED
        ]

    @property
    def non_cleared_indices(self) -> list[tuple[int, int]]:
        return [
            (row_idx, col_idx)
            for col_idx in range(NUM_COLUMNS)
            for row_idx in range(NUM_ROWS)
            if Card.from_one_hot_encoding(self._card_one_hots[col_idx, row_idx])
            != Card.CLEARED
        ]

    def _put_card_and_check_clear(
        self, card: Card, row_idx: int, col_idx: int
    ) -> typing.Self:
        """Helper that handles checking whether column can be cleared after a Card is "put" into position.
        Note: that a card can be "put" from both a replace, and also a flip. As conceptually even on a flip the flipped
        card is "put" into position of the face-down card."""
        card_one_hots = self._card_one_hots.copy()
        card_one_hots[col_idx, row_idx] = card.one_hot_encoding()
        cleared_one_hots = self._cleared_one_hots.copy()
        cleared_card = None
        if col_idx in Hand.clearable_columns(card_one_hots):
            card_one_hots[col_idx] = np.zeros((NUM_ROWS, CARD_TYPES), dtype=np.int8)
            card_one_hots[col_idx, :, Card.CLEARED.one_hot_encoding_index] = 1
            cleared_one_hots[col_idx] = card.one_hot_encoding()
            cleared_card = card
        return Hand(card_one_hots, cleared_one_hots, cleared_card=cleared_card)

    def _render(self) -> list[str]:
        card_width = max(
            max(max(len(str(card)) for card in row) for row in self.cards),
            2,
        )
        empty = " " * card_width
        divider = ("+" + "-" * card_width) * NUM_COLUMNS + "+"
        template = f"|{{:>{card_width}}}" * NUM_COLUMNS + "|"
        lines = [divider] * (NUM_ROWS * 2 + 1)
        lines[1::2] = (
            template.format(*(str(self.cards[r][c]) for c in range(NUM_COLUMNS)))
            for r in range(NUM_ROWS)
        )
        return lines

    def copy(self) -> typing.Self:
        return Hand(
            self._card_one_hots.copy(),
            self._cleared_one_hots.copy(),
            self.cleared_card,
        )

    def numpy(self) -> npt.NDArray[np.int8]:
        return self._card_one_hots

    def get_card(self, row_idx: int, col_idx: int) -> Card:
        assert row_idx < NUM_ROWS and row_idx >= 0, (
            f"row index: {row_idx} out of range, for board with {NUM_ROWS} rows"
        )
        assert col_idx < NUM_COLUMNS and col_idx >= 0, (
            f"col index: {col_idx} out of range, for board with {NUM_COLUMNS} columns"
        )
        return Card.from_one_hot_encoding(self._card_one_hots[col_idx, row_idx])

    def flip(self, card: Card, row_idx: int, col_idx: int) -> typing.Self:
        assert self.get_card(row_idx, col_idx) == Card.FACE_DOWN, (
            f"Cannot flip already flipped card at row_idx: {row_idx}, col_idx: {col_idx}"
        )
        return Hand._put_card_and_check_clear(self, card, row_idx, col_idx)

    def replace(
        self,
        row_idx: int,
        col_idx: int,
        card: Card,
    ) -> tuple[typing.Self, Card]:
        assert self.get_card(row_idx, col_idx) != Card.CLEARED, (
            f"Cannot replace cleared card at row_idx: {row_idx}, col_idx: {col_idx}"
        )
        return Hand._put_card_and_check_clear(
            self, card, row_idx, col_idx
        ), self.get_card(row_idx, col_idx)

    def reveal_all_face_down_cards(self, revealed_cards: list[Card]) -> typing.Self:
        """Returns the value of the hand after revealing all cards.
        This includes any previously undiscovered clearing columns"""
        assert (
            self._card_one_hots[:, :, Card.CLEARED.one_hot_encoding_index].sum()
            // NUM_ROWS
            == self._cleared_one_hots.sum()
        ), f"somehow cleared cards not a multiple of {NUM_ROWS}"
        revealed_hand = self
        for i, (row_idx, col_idx) in enumerate(self.face_down_indices):
            revealed_hand = revealed_hand.flip(revealed_cards[i], row_idx, col_idx)
        return revealed_hand

    def compute_valid_place_from_discard_actions(self) -> list[SkyjoAction]:
        return [
            SkyjoAction(SkyjoActionType.PLACE_FROM_DISCARD, row_idx, col_idx)
            for row_idx, col_idx in self.non_cleared_indices
        ]

    def compute_valid_discard_and_flip_actions(self) -> list[SkyjoAction]:
        return [
            SkyjoAction(SkyjoActionType.DISCARD_AND_FLIP, row_idx, col_idx)
            for row_idx, col_idx in self.face_down_indices
        ]

    def compute_valid_place_drawn_actions(self) -> list[SkyjoAction]:
        return [
            SkyjoAction(SkyjoActionType.PLACE_DRAWN, row_idx, col_idx)
            for row_idx, col_idx in self.non_cleared_indices
        ]


@dataclasses.dataclass(slots=True, frozen=True)
class ImmutableState:
    num_players: int
    """Number of players in the game"""
    player_scores: npt.NDArray[np.int16]  # shape (NUM_PLAYERS,)
    """Current player scores"""
    remaining_card_counts: CardCounts
    """Counts of remaining cards in the deck"""
    hands: list[Hand] = dataclasses.field(default_factory=list)
    """The list of participating players."""
    discard_pile: DiscardPile = dataclasses.field(default_factory=DiscardPile)
    """Discard Pile Object"""
    drawn_card: Card | None = None
    """Card drawn (if applicable)"""
    curr_player: int = 0
    """Index of current player"""
    turn_count: int = 0
    """Number of turns taken in this round"""
    round_turn_counts: list[int] = dataclasses.field(default_factory=list)
    """Number of turns taken in each round"""
    is_round_ending: bool = False
    """If this is the last turn for each player"""
    round_ending_player: int | None = None
    """Index of player who has ended the game"""
    valid_actions: list[SkyjoAction] = dataclasses.field(default_factory=list)
    """List of valid actions available from current state"""
    winning_player: int | None = None
    """Index of player who won the game"""

    def __hash__(self):
        return hash(self.numpy().tobytes())

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ImmutableState):
            return self.__hash__() == other.__hash__()
        return False

    def __post_init__(self):
        assert 2 <= self.num_players <= 8, "Number of players must be between 2 and 8"
        assert self.player_scores.shape == (self.num_players,), (
            "Player scores must be a 1D array of length num_players"
        )

    @classmethod
    def from_numpy(cls, numpy_array: npt.NDArray[np.int8]) -> typing.Self:
        raise NotImplementedError()

    @staticmethod
    def compute_round_scores(
        hands: list[Hand],
        round_ending_player_idx: int,
    ) -> tuple[npt.NDArray[np.int16], CardCounts]:
        assert sum([hand.num_face_down_cards for hand in hands]) == 0, (
            "All cards must be revealed before computing round scores"
        )
        round_scores = np.zeros(len(hands), dtype=np.int16)
        for player_idx in range(len(hands)):
            round_score = hands[player_idx].visible_points
            round_scores[player_idx] += round_score
        if (
            round_scores[round_ending_player_idx]
            >= np.delete(round_scores, round_ending_player_idx).min()
        ):
            round_scores[round_ending_player_idx] *= 2
        return round_scores

    @property
    def round_count(self) -> int:
        if self.game_has_ended:
            return len(self.round_turn_counts)
        else:
            return len(self.round_turn_counts) + 1

    @property
    def total_turn_count(self) -> int:
        return sum(self.round_turn_counts)

    @property
    def game_has_ended(self) -> bool:
        return self.winning_player is not None

    def _create_next_state(self, **kwargs) -> typing.Self:
        """
        Create a new ImmutableState with updated fields, automatically copying mutable attributes.
        Also checks and handles round ending logic automatically.

        This is a safer alternative to dataclasses.replace() as it ensures all mutable
        attributes are properly copied even when not explicitly provided, and it handles
        round ending logic automatically.

        Args:
            **kwargs: Attributes to explicitly set in the new instance

        Returns:
            A new ImmutableState instance with the requested changes and round ending logic applied
        """
        #
        next_turn_count = kwargs.get("turn_count", self.turn_count)
        next_winning_player = kwargs.get("winning_player", self.winning_player)

        # Copy mutable attributes not explicitly provided in kwargs
        next_player_scores = kwargs.get("player_scores", self.player_scores.copy())
        next_hands = kwargs.get("hands", [hand.copy() for hand in self.hands])
        next_round_turn_counts = kwargs.get(
            "round_turn_counts", self.round_turn_counts.copy()
        )
        next_discard_pile = kwargs.get("discard_pile", self.discard_pile.copy())
        next_remaining_card_counts = kwargs.get(
            "remaining_card_counts", self.remaining_card_counts.copy()
        )
        next_drawn_card = kwargs.get("drawn_card")
        next_curr_player = kwargs.get("curr_player", self.curr_player)
        next_is_round_ending = kwargs.get("is_round_ending", self.is_round_ending)
        next_round_ending_player = kwargs.get(
            "round_ending_player", self.round_ending_player
        )
        next_valid_actions = kwargs.get("valid_actions", [])

        # Handle round ending logic
        # Check if current player has all cards revealed
        has_all_revealed = len(next_hands[self.curr_player].face_down_indices) == 0

        # If player revealed all cards and round wasn't ending before, set ending player
        if has_all_revealed and not self.is_round_ending:
            next_is_round_ending = True
            next_round_ending_player = self.curr_player
        # Round is over if round was already ending
        # and next player is the round ending player
        # and current player is not the round ending player
        # last clause is needed since END_ROUND action will
        # still maintain curr_player as round_ending_player
        if (
            self.is_round_ending
            and next_curr_player == self.round_ending_player
            and next_curr_player != self.curr_player
            and max(next_player_scores) < GAME_END_SCORE
        ):
            next_valid_actions = [SkyjoAction(SkyjoActionType.END_ROUND)]
            # Keep turn count the same since END_ROUND action is not an actual turn
            next_turn_count = self.turn_count

        # Use dataclasses.replace to create the new instance
        return ImmutableState(
            num_players=self.num_players,
            player_scores=next_player_scores,
            hands=next_hands,
            turn_count=next_turn_count,
            round_turn_counts=next_round_turn_counts,
            discard_pile=next_discard_pile,
            remaining_card_counts=next_remaining_card_counts,
            drawn_card=next_drawn_card,
            curr_player=next_curr_player,
            is_round_ending=next_is_round_ending,
            round_ending_player=next_round_ending_player,
            valid_actions=next_valid_actions,
            winning_player=next_winning_player,
        )

    def _create_initial_hands(self, num_players: int) -> list[Hand]:
        return [Hand.create_initial_hand() for _ in range(num_players)]

    def start_round(self, top_card: Card) -> typing.Self:
        # For first round just have initial player go first with initial flips
        next_player = (
            self.round_ending_player if self.round_ending_player is not None else 0
        )
        next_hands = self._create_initial_hands(self.num_players)
        return self._create_next_state(
            num_players=self.num_players,
            hands=next_hands,
            remaining_card_counts=CardCounts.create_initial_deck_counts().remove_card(
                top_card
            ),
            discard_pile=DiscardPile().discard(top_card),
            drawn_card=None,
            curr_player=next_player,
            turn_count=1,
            is_round_ending=False,
            round_ending_player=None,
            valid_actions=[
                SkyjoAction(SkyjoActionType.INITIAL_FLIP, row_idx, col_idx)
                for row_idx, col_idx in next_hands[next_player].face_down_indices
            ],
        )

    # Game Actions
    def initial_flip(
        self, flipped_card: Card, row_idx: int, col_idx: int
    ) -> typing.Self:
        assert SkyjoActionType.INITIAL_FLIP in {
            action.action_type for action in self.valid_actions
        }, "Initial flip is not a valid action from this state"
        next_hands = [hand.copy() for hand in self.hands]
        next_hands[self.curr_player] = next_hands[self.curr_player].flip(
            flipped_card, row_idx, col_idx
        )
        next_remaining_card_counts = self.remaining_card_counts.remove_card(
            flipped_card
        )
        next_player = (self.curr_player + 1) % self.num_players
        next_valid_actions = [
            SkyjoAction(SkyjoActionType.INITIAL_FLIP, row_idx, col_idx)
            for row_idx, col_idx in self.hands[next_player].face_down_indices
        ]
        # start round
        if all(
            [
                hand.num_face_down_cards == NUM_ROWS * NUM_COLUMNS - 2
                for hand in next_hands
            ]
        ):
            # for first round player with most points goes first
            if len(self.round_turn_counts) == 0:
                next_player = np.argmax([hand.visible_points for hand in next_hands])
            next_valid_actions = next_hands[
                next_player
            ].compute_valid_place_from_discard_actions() + [
                SkyjoAction(SkyjoActionType.DRAW)
            ]

        return self._create_next_state(
            remaining_card_counts=next_remaining_card_counts,
            hands=next_hands,
            curr_player=next_player,
            valid_actions=next_valid_actions,
        )

    def draw_card(self, drawn_card: Card) -> typing.Self:
        assert SkyjoActionType.DRAW in {
            action.action_type for action in self.valid_actions
        }, "Draw is not a valid action from this state"
        assert self.drawn_card is None, "Must not have a drawn card to draw a new card"
        next_drawn_card = drawn_card
        next_remaining_card_counts = self.remaining_card_counts.remove_card(drawn_card)
        next_valid_actions = (
            self.hands[self.curr_player].compute_valid_place_drawn_actions()
            + self.hands[self.curr_player].compute_valid_discard_and_flip_actions()
        )

        return self._create_next_state(
            drawn_card=next_drawn_card,
            remaining_card_counts=next_remaining_card_counts,
            valid_actions=next_valid_actions,
        )

    def place_drawn(
        self, row_idx: int, col_idx: int, face_down_card: Card | None = None
    ) -> typing.Self:
        assert SkyjoActionType.PLACE_DRAWN in {
            action.action_type for action in self.valid_actions
        }, "Place drawn is not a valid action from this state"
        assert self.drawn_card is not None, "Must have a drawn card to place it"
        next_hands = [hand.copy() for hand in self.hands]
        next_hands[self.curr_player], replaced_card = next_hands[
            self.curr_player
        ].replace(row_idx, col_idx, self.drawn_card)
        if replaced_card == Card.FACE_DOWN:
            assert face_down_card is not None, (
                "Must provide a face down card when replacing a face down card"
            )
            replaced_card = face_down_card
            next_remaining_card_counts = self.remaining_card_counts.remove_card(
                replaced_card
            )
        else:
            next_remaining_card_counts = self.remaining_card_counts.copy()
        next_discard_pile = self.discard_pile.discard(replaced_card)

        # Update player + turn count
        next_player = (self.curr_player + 1) % self.num_players
        next_turn_count = self.turn_count + 1
        next_valid_actions = next_hands[
            next_player
        ].compute_valid_place_from_discard_actions() + [
            SkyjoAction(SkyjoActionType.DRAW)
        ]
        return self._create_next_state(
            hands=next_hands,
            discard_pile=next_discard_pile,
            remaining_card_counts=next_remaining_card_counts,
            drawn_card=None,
            curr_player=next_player,
            turn_count=next_turn_count,
            valid_actions=next_valid_actions,
        )

    def place_from_discard(
        self, row_idx: int, col_idx: int, face_down_card: Card | None = None
    ) -> typing.Self:
        assert SkyjoActionType.PLACE_FROM_DISCARD in {
            action.action_type for action in self.valid_actions
        }, "Place from discard is not a valid action from this state"
        assert self.drawn_card is None, (
            "Must not have a drawn card to place from discard"
        )
        next_hands = [hand.copy() for hand in self.hands]
        # Replace card and update discard pile
        next_hands[self.curr_player], replaced_card = next_hands[
            self.curr_player
        ].replace(row_idx, col_idx, self.discard_pile.top_card)
        if replaced_card == Card.FACE_DOWN:
            assert face_down_card is not None, (
                "Must provide a face down card when replacing a face down card"
            )
            replaced_card = face_down_card
            next_remaining_card_counts = self.remaining_card_counts.remove_card(
                replaced_card
            )
        else:
            next_remaining_card_counts = self.remaining_card_counts.copy()
        next_discard_pile = self.discard_pile.replace_top_card(replaced_card)

        # Update player + turn count
        next_player = (self.curr_player + 1) % self.num_players
        next_turn_count = self.turn_count + 1
        next_valid_actions = next_hands[
            next_player
        ].compute_valid_place_from_discard_actions() + [
            SkyjoAction(SkyjoActionType.DRAW)
        ]
        return self._create_next_state(
            hands=next_hands,
            discard_pile=next_discard_pile,
            drawn_card=None,
            curr_player=next_player,
            turn_count=next_turn_count,
            remaining_card_counts=next_remaining_card_counts,
            valid_actions=next_valid_actions,
        )

    def discard_and_flip(
        self, flipped_card: Card, row_idx: int, col_idx: int
    ) -> typing.Self:
        assert SkyjoActionType.DISCARD_AND_FLIP in {
            action.action_type for action in self.valid_actions
        }, "Discard and flip is not a valid action from this state"
        assert self.drawn_card is not None, "Must have a drawn card to discard and flip"
        assert (
            self.hands[self.curr_player].get_card(row_idx, col_idx) == Card.FACE_DOWN
        ), (
            f"Must provide a face down card index to discard and flip, got: {self.hands[self.curr_player].get_card(row_idx, col_idx)}"
        )
        next_remaining_card_counts = self.remaining_card_counts.remove_card(
            flipped_card
        )
        next_discard_pile = self.discard_pile.discard(self.drawn_card)
        next_hands = [hand.copy() for hand in self.hands]
        next_hands[self.curr_player] = next_hands[self.curr_player].flip(
            flipped_card, row_idx, col_idx
        )

        # Update player + turn count
        next_player = (self.curr_player + 1) % self.num_players
        next_turn_count = self.turn_count + 1
        next_drawn_card = None
        next_valid_actions = next_hands[
            next_player
        ].compute_valid_place_from_discard_actions() + [
            SkyjoAction(SkyjoActionType.DRAW)
        ]
        return self._create_next_state(
            drawn_card=next_drawn_card,
            remaining_card_counts=next_remaining_card_counts,
            hands=next_hands,
            discard_pile=next_discard_pile,
            curr_player=next_player,
            turn_count=next_turn_count,
            valid_actions=next_valid_actions,
        )

    def end_round(self, cards_revealed: list[Card]) -> typing.Self:
        assert SkyjoActionType.END_ROUND in {
            action.action_type for action in self.valid_actions
        }, "End round is not a valid action from this state"
        assert self.curr_player == self.round_ending_player, (
            "Must be the last player to reveal hands"
        )
        assert len(cards_revealed) == sum(
            [hand.num_face_down_cards for hand in self.hands]
        ), "Number of cards revealed must match number of face down cards in hands"
        next_hands = []
        cards_flipped = 0
        next_player_scores = self.player_scores.copy()
        for hand in self.hands:
            next_hands.append(
                hand.reveal_all_face_down_cards(
                    cards_revealed[
                        cards_flipped : cards_flipped + len(hand.face_down_indices)
                    ]
                )
            )
            cards_flipped += len(hand.face_down_indices)
        next_remaining_card_counts = self.remaining_card_counts.remove_cards(
            cards_revealed
        )
        next_player_scores = self.player_scores.copy()
        round_scores = self.compute_round_scores(next_hands, self.round_ending_player)
        next_player_scores += round_scores
        next_round_turn_counts = self.round_turn_counts.copy() + [self.turn_count]
        if max(next_player_scores) >= GAME_END_SCORE:
            next_valid_actions = []
            next_winning_player = np.argmin(next_player_scores)
        else:
            next_valid_actions = [SkyjoAction(SkyjoActionType.START_ROUND)]
            next_winning_player = None
        return self._create_next_state(
            player_scores=next_player_scores,
            hands=next_hands,
            remaining_card_counts=next_remaining_card_counts,
            drawn_card=None,
            curr_player=self.round_ending_player,
            valid_actions=next_valid_actions,
            winning_player=next_winning_player,
            round_turn_counts=next_round_turn_counts,
        )

    def involves_chance(self, action: SkyjoAction) -> bool:
        match action.action_type:
            case SkyjoActionType.DRAW:
                return True
            case SkyjoActionType.INITIAL_FLIP:
                return True
            case SkyjoActionType.DISCARD_AND_FLIP:
                return True
            case SkyjoActionType.PLACE_DRAWN:
                if (
                    self.hands[self.curr_player].get_card(
                        action.row_idx, action.col_idx
                    )
                    == Card.FACE_DOWN
                ):
                    return True
                else:
                    return False
            case SkyjoActionType.PLACE_FROM_DISCARD:
                if (
                    self.hands[self.curr_player].get_card(
                        action.row_idx, action.col_idx
                    )
                    == Card.FACE_DOWN
                ):
                    return True
                else:
                    return False
            case SkyjoActionType.END_ROUND:
                return True
            case SkyjoActionType.START_ROUND:
                return True
            case _:
                raise ValueError(f"Invalid action: {action}")

    def apply_action(self, action: SkyjoAction) -> typing.Self:
        match action.action_type:
            case SkyjoActionType.DRAW:
                drawn_card = self.remaining_card_counts.generate_random_card()
                return self.draw_card(drawn_card)
            case SkyjoActionType.PLACE_DRAWN:
                face_down_card = None
                if (
                    self.hands[self.curr_player].get_card(
                        action.row_idx, action.col_idx
                    )
                    == Card.FACE_DOWN
                ):
                    face_down_card = self.remaining_card_counts.generate_random_card()
                return self.place_drawn(action.row_idx, action.col_idx, face_down_card)
            case SkyjoActionType.PLACE_FROM_DISCARD:
                face_down_card = None
                if (
                    self.hands[self.curr_player].get_card(
                        action.row_idx, action.col_idx
                    )
                    == Card.FACE_DOWN
                ):
                    face_down_card = self.remaining_card_counts.generate_random_card()
                return self.place_from_discard(
                    action.row_idx, action.col_idx, face_down_card
                )
            case SkyjoActionType.DISCARD_AND_FLIP:
                flipped_card = self.remaining_card_counts.generate_random_card()
                return self.discard_and_flip(
                    flipped_card, action.row_idx, action.col_idx
                )
            case SkyjoActionType.END_ROUND:
                cards_revealed = self.remaining_card_counts.generate_random_cards(
                    sum([hand.num_face_down_cards for hand in self.hands])
                )
                return self.end_round(cards_revealed)
            case SkyjoActionType.START_ROUND:
                top_card = self.remaining_card_counts.generate_random_card()
                return self.start_round(top_card)
            case SkyjoActionType.INITIAL_FLIP:
                flipped_card = self.remaining_card_counts.generate_random_card()
                return self.initial_flip(flipped_card, action.row_idx, action.col_idx)
            case _:
                raise ValueError(f"Invalid action: {action}")

    def spatial_numpy(self) -> npt.NDArray[np.int8]:
        """Returns a numpy array representation of the player hands.
        The representation is from the perspective of the current player.

        OUTPUT SHAPE: (NUM_PLAYERS, CARD_TYPES, NUM_COLUMNS, NUM_ROWS)
        """
        return np.stack(
            [hand.numpy() for hand in np.roll(self.hands, -self.curr_player)],
            dtype=np.int8,
            axis=0,
        )

    def non_spatial_numpy(self) -> npt.NDArray[np.int16]:
        """Returns a numpy array representation of the non-spatial features of the state.

        OUTPUT SHAPE: sum(
            NUM_PLAYERS,
            NUM_CARD_TYPES * 2, (discard counts + top card)
            NUM_CARD_TYPES, (remaining counts)
            NUM_CARD_TYPES, (drawn card)
            1, (round ending)
        ,)
        """
        curr_score_numpy = np.roll(self.player_scores, -self.curr_player)
        discard_pile_numpy = self.discard_pile.numpy()
        # TODO: Correctly discard cleared cards
        # remaining_card_counts_numpy = self.remaining_card_counts.numpy()
        drawn_card_numpy = np.zeros(CARD_TYPES, dtype=np.int8)
        if self.drawn_card is not None:
            drawn_card_numpy = self.drawn_card.one_hot_encoding()
        # round_ending_numpy = np.zeros(1, dtype=np.int8)
        # if self.is_round_ending:
        #     round_ending_numpy += 1
        return np.concatenate(
            [
                curr_score_numpy,
                # remaining_card_counts_numpy,
                discard_pile_numpy,
                drawn_card_numpy,
                # round_ending_numpy,
            ],
            dtype=np.int16,
        )

    def numpy(self) -> tuple[npt.NDArray[np.int16]]:
        """Returns a numpy array representation of the state from the perspective of the current player"""

        return np.concatenate(
            [
                self.non_spatial_numpy(),
                self.spatial_numpy().reshape(-1, 1).squeeze(),
            ],
            dtype=np.int16,
        )

    def create_valid_actions_mask(self) -> npt.NDArray[np.int8]:
        raise NotImplementedError("Not implemented")

    # Methods to display the current state
    def _render(self) -> list[str]:
        first_line = (
            f"discard: {self.discard_pile.top_card}"
            f" drawn card: {self.drawn_card}"
            f"  round: {self.round_count}"
            f"  turn: {self.turn_count}"
            f" player_scores: {self.player_scores}"
            + (
                f"  (player {self.round_ending_player} ended)"
                if self.is_round_ending
                else ""
            )
            + (
                f"  (WINNER: {self.winning_player})"
                if self.winning_player is not None
                else ""
            )
        )

        hands_render = [""]
        for player_idx in range(self.num_players):
            hand_render = self.hands[player_idx]._render()
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

    def display_str(self) -> str:
        return "# " + "\n".join(self._render())

    def display(self) -> None:
        print(self.display_str())


if __name__ == "__main__":
    # Simulate a single game
    initial_counts = CardCounts.create_initial_deck_counts()
    state = ImmutableState(
        num_players=2,
        player_scores=np.array([0, 0]),
        remaining_card_counts=initial_counts,
    ).start_round(initial_counts.generate_random_card())
    state.display()
    while not state.game_is_over():
        action = state.valid_actions[np.random.randint(0, len(state.valid_actions))]
        state = state.apply_action(action)
        state.display()
    print(state.round_turn_counts)
