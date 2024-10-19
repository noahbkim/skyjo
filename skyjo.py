from __future__ import annotations

import abc
import datetime
import functools
import multiprocessing
import random
import time
from dataclasses import dataclass, field
from typing import ClassVar, Collection, Final, Iterator

DECK_SIZE = 150
DECK = (
    (-2,) * 5 + (-1,) * 10 + (0,) * 15 +
    (1,) * 10 + (2,) * 10 + (3,) * 10 + (4,) * 10 +
    (5,) * 10 + (6,) * 10 + (7,) * 10 + (8,) * 10 +
    (9,) * 10 + (10,) * 10 + (11,) * 10 + (12,) * 10
)

HAND_ROWS = 3
HAND_COLUMNS = 4
HAND_SIZE = HAND_ROWS * HAND_COLUMNS


class RuleError(Exception):
    """Raised when a rule is broken."""


@dataclass(slots=True)
class Cards:
    """The discard and draw piles in the center of the table."""

    _rng: random.Random = field(default_factory=random.Random)
    _deck: Collection[int] = field(default=DECK)
    _buffer: list[int] = field(init=False)
    _discard_index: int = 0
    _draw_index: int = 1

    def __post_init__(self) -> None:
        self._buffer = list(self._deck)

    @property
    def top_discard(self) -> int:
        return self._buffer[self._discard_index]
    
    @property
    def is_empty(self) -> int:
        return self._draw_index == len(self._buffer)
    
    def _peek_draw(self) -> int:
        assert self._draw_index < len(self._buffer), "Deck must be restocked!"
        return self._buffer[self._draw_index]

    def _draw_card(self) -> int:
        assert self._draw_index < len(self._buffer), "Deck must be restocked!"
        card = self._buffer[self._draw_index]
        self._draw_index += 1
        if self._draw_index == len(self._buffer):
            self._restock_draw()
        return card
    
    def _discard_card(self, card: int) -> None:
        assert self._discard_index < len(self._buffer), "Deck has too many cards!"
        self._discard_index += 1
        self._buffer[self._discard_index] = card

    def _replace_discard_card(self, card: int) -> None:
        assert self._discard_index < len(self._buffer), "Deck has too many cards!"
        self._buffer[self._discard_card] = card

    def _restock_draw(self) -> None:
        assert self._discard_index < len(self._buffer), "Invalid discard index!"
        assert self._draw_index == len(self._buffer), "Restocked when not empty!"
        buried = self._buffer[:self._discard_index]
        self._rng.shuffle(buried)
        self._buffer[0] = self._buffer[self._discard_index]
        self._buffer[-len(buried):] = buried
        self._discard_index = 0
        self._draw_index = len(self._buffer) - len(buried)

    def _reset_and_shuffle(self) -> None:
        self._buffer = list(self._deck)
        self._discard_index = 0
        self._draw_index = 1
        self._rng.shuffle(self._buffer)

    def validate(self) -> None:
        assert 0 <= self._discard_index < len(self._buffer)
        assert 0 < self._draw_index <= len(self._buffer)
        assert sorted(self._buffer) == sorted(DECK)


@dataclass(slots=True)
class Finger:
    """The slot for a card in a hand."""

    _card: int

    is_visible: bool = 0


    @property
    def card(self) -> int | None:
        return self._card if self.is_visible else None
    
    def _flip_card(self) -> None:
        assert not self.is_visible, "Cannot flip already-visible card!"
        self.is_visible = True
    
    def _replace_card(self, card: int) -> int:
        replaced_card = self._card
        self._card = card
        self.is_visible = True
        return replaced_card


@dataclass(slots=True)
class Hand:
    """A player's set of cards."""

    _fingers: list[Finger] = field(default_factory=list)

    rows: int = HAND_ROWS
    original_columns: int = HAND_COLUMNS

    def __post_init__(self) -> None:
        assert self.rows > 0
        assert self.original_columns > 0

    def __getitem__(self, index_or_coordinates: int | tuple[int, int]) -> Finger:
        if isinstance(index_or_coordinates, int):
            return self._fingers[index_or_coordinates]
        row, column = index_or_coordinates
        return self._fingers[row * self.columns + column]
    
    def __iter__(self) -> Iterator[Finger]:
        return iter(self._fingers)

    def __len__(self) -> int:
        return len(self._fingers)
    
    @property
    def columns(self) -> int:
        return self.card_count // self.rows

    @property
    def card_count(self) -> int:
        return len(self._fingers)
    
    @property
    def original_card_count(self) -> int:
        return self.rows * self.original_columns
    
    @property
    def visible_value(self) -> int:
        value = 0
        for finger in self._fingers:
            if finger.is_visible:
                value += finger.card
        return value
    
    @property
    def visible_count(self) -> int:
        count = 0
        for finger in self._fingers:
            if finger.is_visible:
                count += 1
        return count

    @property
    def cleared_count(self) -> int:
        return self.original_card_count - len(self._fingers)
    
    @property
    def are_all_cards_visible(self) -> int:
        return self.visible_count == self.card_count
    
    def _get_valid_index(self, row: int, column: int | None) -> None:
        index = row * self.columns + column if column is not None else row
        if index > len(self._fingers):
            raise RuleError(f"Card ({row}, {column}) is outside hand dimensions")
        return index
        
    def _deal_from(self, cards: Cards) -> None:
        self._fingers.clear()
        for _ in range(self.original_card_count):
            self._fingers.append(Finger(cards._draw_card()))
    
    def _try_clear(self, column: int) -> bool:
        card = self._fingers[column]._card
        for finger in self._fingers[column::self.rows]:
            if not finger.is_visible or finger._card != card:
                return False
        del self._fingers[column::self.rows]
        return True
    
    def _replace_card(self, index: int, card: int) -> int:
        replaced_card = self._fingers[index]._replace_card(card)
        self._try_clear(index % self.rows)
        return replaced_card

    def _flip_card(self, index: int) -> None:
        self._fingers[index]._flip_card()
        self._try_clear(index % self.rows)

    def _flip_all_cards(self) -> None:
        for finger in self._fingers:
            if not finger.is_visible:
                finger._flip_card()
        for column in reversed(range(self.columns)):  # Reverse to avoid skipping
            self._try_clear(column)

    def _clear(self) -> None:
        self._fingers.clear()

@dataclass(slots=True)
class Flip:
    """An action controller for flipping the first two cards."""

    _hand: Hand
    _first_index: int | None = None
    _second_index: int | None = None

    @property
    def flipped_count(self) -> int:
        if self._first_index is None:
            return 0
        elif self._second_index is None:
            return 1
        return 2

    def flip_card(self, row: int, column: int | None = None, /) -> int:
        index = self._hand._get_valid_index(row, column)
        if self._first_index is None:
            self._first_index = index
            return self._hand[index]._card
        elif self._second_index is None:
            self._second_index = index
            return self._hand[index]._card
        raise RuleError("Tried to flip more than two cards!")
    
    def _apply_to_hand(self) -> None:
        if self._first_index is None or self._second_index is None:
            raise RuleError("Fewer than two cards were flipped!")
        self._hand[self._first_index]._flip_card()
        self._hand[self._second_index]._flip_card()


@dataclass
class Turn:
    """An action controller for taking a turn."""

    DISCARD_AND_FLIP: ClassVar[Final[int]] = 1
    PLACE_DRAWN_CARD: ClassVar[Final[int]] = 2
    PLACE_FROM_DISCARD: ClassVar[Final[int]] = 3

    _hand: Hand
    _cards: Cards
    _action: int | None = None
    _selected_card: int | None = None
    _replaced_or_flipped_index: int | None = None

    def draw_card(self) -> int:
        if self._action is not None:
            raise RuleError("Cannot take multiple actions in a single turn")
        if self._selected_card is not None:
            raise RuleError("Cannot draw more than one card")
        self._selected_card = self._cards._peek_draw()
        return self._selected_card

    def discard_and_flip(self, row: int, column: int | None = None, /) -> int:
        if self._action is not None:
            raise RuleError("Cannot take multiple actions in a single turn")
        if self._selected_card is None:
            raise RuleError("Cannot discard and flip without first drawing a card")
        self._action = Turn.DISCARD_AND_FLIP
        self._replaced_or_flipped_index = self._hand._get_valid_index(row, column)
        return self._hand[self._replaced_or_flipped_index]._card

    def place_drawn_card(self, row: int, column: int | None = None, /) -> int:
        if self._action is not None:
            raise RuleError("Cannot take multiple actions in a single turn")
        if self._selected_card is None:
            raise RuleError("Cannot place drawn card without first drawing a card")
        self._action = Turn.PLACE_DRAWN_CARD
        self._replaced_or_flipped_index = self._hand._get_valid_index(row, column)
        return self._hand[self._replaced_or_flipped_index]._card

    def place_from_discard(self, row: int, column: int | None = None, /) -> int:
        if self._action is not None:
            raise RuleError("Cannot take multiple actions in a single turn")
        if self._selected_card is not None:
            raise RuleError("Cannot place from discard after drawing a card")
        self._selected_card = self._cards.top_discard
        self._action = Turn.PLACE_FROM_DISCARD
        self._replaced_or_flipped_index = self._hand._get_valid_index(row, column)
        return self._hand[self._replaced_or_flipped_index]._card

    def _apply_to_hand_and_cards(self) -> None:
        if self._action == Turn.DISCARD_AND_FLIP:
            self._cards._discard_card(self._cards._draw_card())
            self._hand._flip_card(self._replaced_or_flipped_index)
        elif self._action == Turn.PLACE_DRAWN_CARD:
            drawn_card = self._cards._draw_card()
            replaced_card = self._hand._replace_card(self._replaced_or_flipped_index, drawn_card)
            self._cards._discard_card(replaced_card)
        elif self._action == Turn.PLACE_FROM_DISCARD:
            replaced_card = self._hand._replace_card(self._replaced_or_flipped_index, self._cards.top_discard)
            self._cards._replace_discard_card(replaced_card)


@dataclass(slots=True)
class Player(abc.ABC):
    """A Skyjo player."""

    hand: Hand = field(default_factory=Hand)
    score: int = 0

    def __str__(self) -> None:
        return type(self).__qualname__
        
    @abc.abstractmethod
    def flip(self, state: State, action: Flip) -> None:
        """Start the game by flipping two cards in your hand."""

    @abc.abstractmethod
    def turn(self, state: State, action: Turn) -> None:
        """Take a turn as part of a round."""


@dataclass(slots=True)
class State:
    """The entire state of a Skyjo game."""

    players: list[Player] = field()
    round_index: int = field(init=False, default=0)
    turn_index: int = field(init=False, default=0)
    round_ender_index: int | None = field(init=False, default=None)
    
    _rng: random.Random = field(default_factory=random.Random)
    _cards: Cards = field(default=None)

    def __post_init__(self) -> None:
        if self._cards is None:
            self._cards = Cards(_rng=self._rng)

    @property
    def top_discard(self) -> Cards:
        return self._cards.top_discard
    
    @property
    def is_round_ending(self) -> bool:
        return self.round_ender_index is not None

    @property
    def largest_visible_hand_player_index(self) -> None:
        return max(range(len(self.players)), key=lambda i: self.players[i].hand.visible_value)
    
    @property
    def lowest_score_player_index(self) -> int:
        return min(range(len(self.players)), key=lambda i: self.players[i].score)
    
    def play(self):
        """Play a single game of Skyjo, returning final scores."""

        if len(self.players) < 3:
            raise RuleError(f"Must have at least 3 players, got {len(self.players)}")
        elif len(self.players) > 8:
            raise RuleError(f"Must have 8 or fewer players, got {len(self.players)}")

        self.round_index = 0
        self.turn_index = 0
        self.round_ender_index = None
        self._cards._reset_and_shuffle()
        
        for index, player in enumerate(self.players):
            player.score = 0
            player.hand._clear()
            player.hand._deal_from(self._cards)

        flips = [Flip(player.hand) for player in self.players]
        for player, flip in zip(self.players, flips):
            player.flip(self, flip)
        for flip in flips:
            flip._apply_to_hand()
        del flips

        starting_index = self.largest_visible_hand_player_index
        rotation = self.players[starting_index:] + self.players[:starting_index]

        while True:
            while not self.is_round_ending:
                for index, player in enumerate(rotation):
                    turn = Turn(player.hand, self._cards)
                    player.turn(self, turn)
                    turn._apply_to_hand_and_cards()
                    self.turn_index += 1
                    if player.hand.are_all_cards_visible:
                        round_ender_index = self.round_ender_index = index
                        break
            
            for _ in range(len(rotation) - 1):
                player = rotation[self.turn_index % len(rotation)]
                turn = Turn(player.hand, self._cards)
                player.turn(self, turn)
                turn._apply_to_hand_and_cards()
                self.turn_index += 1

            for player in rotation:
                player.hand._flip_all_cards()
            round_scores = [player.hand.visible_value for player in rotation]

            round_ender_score = round_scores[round_ender_index]
            if min(round_scores) < round_ender_score or round_scores.count(round_ender_score) > 1:
                round_scores[round_ender_index] *= 2
            for player, round_score in zip(rotation, round_scores):
                player.score += round_score
            
            self.round_index += 1  # So we get the right count after breaking
            self.turn_index = 0
            
            if max(rotation, key=lambda player: player.score).score >= 100:
                break

            rotation = rotation[round_ender_index:] + rotation[:round_ender_index]
            self._cards._reset_and_shuffle()
            for index, player in enumerate(rotation):
                player.hand._clear()
                player.hand._deal_from(self._cards)


@dataclass(slots=True)
class Outcomes:
    player_count: int
    game_count: int = 0
    average_round_count: float = 0.0
    average_scores: list[float] = field(default=None)
    win_counts: list[int] = field(default=None)

    def __post_init__(self) -> None:
        if self.average_scores is None:
            self.average_scores = [0.0] * self.player_count
        if self.win_counts is None:
            self.win_counts = [0] * self.player_count

    def __add__(self, other: object) -> Outcomes:
        if not isinstance(other, Outcomes):
            raise NotImplemented
        if self.player_count != other.player_count:
            raise ValueError(f"Player counts {self.player_count} and {other.player_count} do not match")
        game_count = self.game_count + other.game_count
        weighted = lambda x, y: (x * self.game_count + y * other.game_count) / game_count
        return Outcomes(
            player_count=self.player_count,
            game_count=game_count,
            average_round_count=weighted(self.average_round_count, other.average_round_count),
            average_scores=[weighted(x, y) for x, y in zip(self.average_scores, other.average_scores)],
            win_counts=[x + y for x, y in zip(self.win_counts, other.win_counts)],
        )
    
    def _finalize(self) -> None:
        self.average_round_count /= self.game_count
        for i in range(self.player_count):
            self.average_scores[i] /= self.game_count


def play(
    players: list[Player],
    games: int = 1,
    *,
    rng: random.Random | None = None,
    processes: int = multiprocessing.cpu_count(),
    display: bool = True,
) -> Outcomes:
    """Play `games` rounds of Skyjo, aggregating statisics."""

    if rng is not None and processes > 1:
        raise ValueError("Cannot multiprocess with a fixed rng")
    elif rng is None:
        rng = random.Random()

    if display:
        start_time = time.monotonic()

    if processes > 1:
        partial = functools.partial(play, players, processes=1, display=False)
        chunks = tuple(filter(None, (games // processes + (i < games % processes) for i in range(processes))))
        processes = len(chunks)
        with multiprocessing.Pool(processes=processes) as pool:
            outcomes = sum(pool.map(partial, chunks), start=Outcomes(player_count=len(players)))

    elif processes != 1:
        raise ValueError(f"Invalid process count {processes}, must be 1 or larger")
    
    else:
        state = State(players, _rng=rng)
        outcomes = Outcomes(player_count=len(players))
        for _ in range(games):
            state.play()
            outcomes.game_count += 1
            outcomes.average_round_count += state.round_index
            for index, player in enumerate(state.players):
                outcomes.average_scores[index] += player.score
            outcomes.win_counts[state.lowest_score_player_index] += 1
        outcomes._finalize()

    if display:
        delta = datetime.timedelta(seconds=time.monotonic() - start_time)
        print(f"Played {games} games on {processes} cores in {delta}")
        print(f"  Average rounds: {outcomes.average_round_count:.2f}")
        for index, player in enumerate(players):
            win_count = outcomes.win_counts[index]
            win_percent = win_count / games * 100
            average_score = outcomes.average_scores[index]
            print(f"  {player}[{index}]: {win_count} wins ({win_percent:.2f}%), average score {average_score:.2f}")

    return outcomes