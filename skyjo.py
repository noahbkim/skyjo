from __future__ import annotations

import abc
import datetime
import functools
import inspect
import multiprocessing
import pickle
import random
import textwrap
import time
import traceback
from dataclasses import dataclass, field
from typing import ClassVar, Collection, Final, Iterator, TypeVar

T = TypeVar("T")

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

    repro: object | None = None


def inspectable(t: T) -> T:
    """Mark an instance method as safe to call for introspection."""

    setattr(t, "__inspectable__", True)
    return t


def isinspectable(o: object) -> bool:
    """Determines whether an object has been marked inspectable."""

    return getattr(o, "__inspectable__", False)


@dataclass(slots=True, frozen=True)
class Average:
    """Holds a dividend and divisor, provides arithmetic overrides."""

    dividend: float = 0
    divisor: float = 0

    def __add__(self, other: int | float | Average) -> Average:
        if isinstance(other, Average):
            return Average(self.dividend + other.dividend, self.divisor + other.divisor)
        return Average(self.dividend + other, self.divisor + 1)
    
    def __radd__(self, other: int | float | Average) -> Average:
        return self + other

    def __float__(self) -> float:
        if self.divisor == 0:
            return float("nan")
        return self.dividend / self.divisor
    
    def __format__(self, spec: str) -> str:
        return float(self).__format__(spec)


@dataclass(slots=True)
class Cards:
    """The discard and draw piles in the center of the table.
    
    Because there are a fixed number of cards for the duration of a
    game, we can represent both the discard and draw piles using a list
    of constant size, `_buffer`, and two indices `_discard_index` and
    `_draw_index`. A (dubiously) shuffled `Cards` look like:

          _discard_index
          v
        [-2, -1, 0, 1, 2, 3, 4, ...] < _buffer
              ^
              _draw_index
    
    This represents a discard pile with just -2 facing up and a draw
    pile containing everything else (with a -1 on top).
    
    When a card is `drawn`, the card at `_draw_index` is retrieved then
    the index is incremented. We don't bother overwriting draw cards
    with e.g. `None` because they should never be accessed. If `_draw`
    was called four times, our indices would now be:

          _discard_index
          v
        [-2, -1, 0, 1, 2, 3, 4, ...] < _buffer
                          ^
                          _draw_index

    When a card is discarded, we increment `_discard_index` and set the
    corresponding element in `_buffer` to its value. If we discarded our
    four cards in reverse order, we would have:

                       _discard_index
                       v
        [-2, 2, 1, 0, -1, 3, 4, ...] < _buffer
                          ^
                          _draw_index
    
    Once we've exhausted our draw pile, we can shuffle the buried
    discards and slice them back into the draw pile.
    
    To save ourselves the trouble of collecting all cards still in hand
    at the end of each round, we hold a reference to the original list
    of cards in our deck in `_deck` and reset to that instead.
    """

    _rng: random.Random = field(default_factory=random.Random, repr=False)
    """A shared random number generator for deterministic shuffling."""

    _deck: Collection[int] = field(default=DECK, repr=False)
    """The original collection of cards to reset the deck to."""

    _buffer: list[int] = field(init=False, repr=False)
    """A container for the discard and draw piles."""

    _discard_index: int = 0
    """The index of the topmost discard."""

    _draw_index: int = 1
    """The index of the topmost card in the draw pile."""

    # Prevent `dataclass` from trying to generate a `__hash__`.
    def __hash__(self) -> None:
        return id(self)

    # Reset the buffer on start.
    def __post_init__(self) -> None:
        self._buffer = list(self._deck)

    @property
    def last_discard(self) -> int:
        """The card currently on top of the discard pile."""

        return self._buffer[self._discard_index]

    @property
    @functools.cache
    def average_deck_card_value(self) -> float:
        """Get the average card value of the complete deck."""

        return sum(self._deck) / len(self._deck)

    @property
    def _next_draw_card(self) -> int:
        """The (hidden) card on top of the draw pile."""
        
        assert self._draw_index < len(self._buffer), "Deck must be restocked!"
        return self._buffer[self._draw_index]
    
    def _draw_card(self) -> int:
        assert self._draw_index < len(self._buffer), "Deck must be restocked!"
        card = self._buffer[self._draw_index]
        self._draw_index += 1
        if self._draw_index == len(self._buffer):
            self._restock_draw()
        return card
    
    def _discard_card(self, card: int) -> int:
        assert self._discard_index < len(self._buffer), "Deck has too many cards!"
        self._discard_index += 1
        self._buffer[self._discard_index] = card
        return card

    def _replace_discard_card(self, card: int) -> int:
        assert self._discard_index < len(self._buffer), "Deck has too many cards!"
        previous_discard_card = self._buffer[self._discard_index]
        self._buffer[self._discard_index] = card
        return previous_discard_card

    def _restock_draw(self) -> None:
        assert self._discard_index < len(self._buffer), "Invalid discard index!"
        assert self._draw_index == len(self._buffer), "Restocked when not empty!"
        buried = self._buffer[:self._discard_index]
        self._rng.shuffle(buried)
        self._buffer[0] = self._buffer[self._discard_index]
        self._buffer[-len(buried):] = buried
        self._discard_index = 0
        self._draw_index = len(self._buffer) - len(buried)

    def _reset(self) -> None:
        self._buffer = list(self._deck)
        self._discard_index = 0
        self._draw_index = 1

    def _shuffle(self) -> None:
        self._rng.shuffle(self._buffer)

    def validate(self) -> None:
        assert 0 <= self._discard_index < len(self._buffer)
        assert 0 < self._draw_index <= len(self._buffer)
        assert sorted(self._buffer) == sorted(DECK)


@dataclass(slots=True)
class Finger:
    """The slot for a card in a hand.
    
    The class' role is primarily to hide the value of the card placed
    here from each `Player` until it's flipped or replaced.
    """

    _card: int
    """The card in this position of the hand."""

    is_flipped: bool = 0
    """Whether the card is face up."""

    @property
    def card(self) -> int | None:
        """The value of the card if it's face up, `None` otherwise."""

        return self._card if self.is_flipped else None
    
    def _flip_card(self) -> None:
        assert not self.is_flipped, "Cannot flip already-flipped card!"
        self.is_flipped = True
    
    def _replace_card(self, card: int) -> int:
        replaced_card = self._card
        self._card = card
        self.is_flipped = True
        return replaced_card


@dataclass(slots=True)
class Hand:
    """A player's set of cards arranged in a grid.
    
    The `Hand` represents its `Finger`'s as a flat `list` for both
    performance (less indirection) and convenience (easy to slice). The
    number of rows is constant, so the number of columns can always be
    determined by `len(_fingers) / rows`. Fingers can also be indexed by
    row and column, in which case they are arranged as follows:

                 0   1   2   3 < columns
                -4  -3  -2  -1

        0 -1  [  0,  1,  2,  3,
        1 -2     4,  5,  6,  7,
        2 -3     8,  9, 10, 11, ]
        ^                    ^
        rows                 indices
    """

    _fingers: list[Finger] = field(default_factory=list)

    row_count: int = HAND_ROWS
    original_column_count: int = HAND_COLUMNS
    flipped_card_count: int = 0  # Optimization

    def __post_init__(self) -> None:
        assert self.row_count > 0
        assert self.original_column_count > 0

    def __getitem__(self, index: int | tuple[int, int]) -> Finger:
        if isinstance(index, int):
            return self._fingers[index]
        row, column = index
        return self._fingers[row * self.column_count + column]
    
    def __iter__(self) -> Iterator[Finger]:
        return iter(self._fingers)

    def __len__(self) -> int:
        return len(self._fingers)
    
    @property
    def card_count(self) -> int:
        """The remaining number of cards in this hand."""

        return len(self._fingers)
    
    @property
    def column_count(self) -> int:
        """The remaining number of columns in this hand."""

        return self.card_count // self.row_count
    
    @property
    def original_card_count(self) -> int:
        """The original number of cards dealt to this hand."""

        return self.row_count * self.original_column_count
    
    @property
    def cleared_card_count(self) -> int:
        """The number of cards that have been cleared from this hand."""

        return self.original_card_count - len(self._fingers)
    
    @property
    def cleared_column_count(self) -> int:
        """The number of columns that have been cleared from this hand."""

        return self.original_column_count - self.column_count
    
    @property
    def unflipped_card_count(self) -> int:
        """The number of cards yet to be flipped."""

        return self.card_count - self.flipped_card_count

    @property
    def are_all_cards_flipped(self) -> bool:
        """Whether all cards in this hand have been flipped."""

        return self.flipped_card_count == self.card_count
    
    def iter_columns(self) -> Iterator[list[Finger]]:
        """Yield each column in this hand."""

        for i in range(self.column_count):
            yield self._fingers[i::self.column_count]

    def iter_rows(self) -> Iterator[list[Finger]]:
        """Yield each row in this hand."""

        for i in range(0, self.card_count, self.column_count):
            yield self._fingers[i:i+self.column_count]
    
    def sum_flipped_cards(self) -> int:
        """Get the total value of flipped cards in this hand."""

        return sum(finger.card for finger in self._fingers if finger.is_flipped)
    
    def find_first_unflipped_card_index(self) -> int | None:
        """Get the index of the first unflipped card if one is present."""

        for i, finger in enumerate(self._fingers):
            if not finger.is_flipped:
                return i
        return None
    
    def find_highest_card_index(self) -> int | None:
        """Get the index of the highest card if any are flipped."""

        flipped_card_indices = [i for i, finger in enumerate(self._fingers) if finger.is_flipped]
        if not flipped_card_indices:
            return None
        return max(flipped_card_indices, key=lambda i: self._fingers[i].card)
    
    def find_lowest_card_index(self) -> int | None:
        """Get the index of the lowest card if any are flipped."""

        flipped_card_indices = [i for i, finger in enumerate(self._fingers) if finger.is_flipped]
        if not flipped_card_indices:
            return None
        return min(flipped_card_indices, key=lambda i: self._fingers[i].card)

    def find_first_unflipped_or_highest_card_index(self) -> int:
        """Get the first unflipped card or the highest card; must exist."""

        index = self.find_first_unflipped_card_index()
        if index is not None:
            return index
        index = self.find_highest_card_index()
        assert index is not None
        return index
    
    def count_cards(self, card: int) -> int:
        """Count how many of the specified card are visible."""

        count = 0
        for finger in self._fingers:
            if finger.is_flipped and finger._card == card:
                count += 1
        return count
    
    def find_first_clearing_index(self, card: int) -> int | None:
        """Find a position where the given card will clear a column.
        
        This method will return a position even if the provided card is
        negative and would be detrimental to clear.
        """

        for column in range(self.column_count):
            if self[0, column].card == column[1, column].card == card:
                return 2 * self.column_count + column
            if self[0, column].card == column[2, column].card == card:
                return 1 * self.column_count + column
            if self[1, column].card == column[2, column].card == card:
                return 0 * self.column_count + column
        return None

    def is_column_cleared_with_placement(
        self,
        card: int,
        row_or_index: int,
        column: int | None = None,
    ) -> bool:
        """Check if placing the given card will clear a column."""

        row, column = self.coordinates(row_or_index, column)
        for i in range(self.row_count):
            if i != row and self[i, column].card != card:
                return False
        return True

    def coordinates(self, row_or_index: int, column: int | None = None) -> tuple[int, int]:
        """Convert an index to coordintes and validate."""

        if column is not None:
            self._validate_coordinates(row_or_index, column)
        else:
            self._validate_index(row_or_index)
            row_or_index, column = divmod(row_or_index, self.column_count) 
            if row_or_index < 0:  # Account for negative divmod
                column -= self.column_count

        return row_or_index, column

    def index(self, row_or_index: int, column: int | None = None) -> int:
        """Convert coordinates to an index and validate."""

        if column is None:
            self._validate_index(row_or_index)
        else:
            self._validate_coordinates(row_or_index, column)
            row_or_index = row_or_index * self.column_count + column

        return row_or_index

    def _validate_coordinates(self, row: int, column: int) -> None:
        row_count = self.row_count
        column_count = self.column_count
        if row < -row_count or row >= row_count or column < -column_count or column >= column_count:
            raise IndexError(f"Hand coordinates ({row}, {column}) outside hand size [{row_count}, {column_count}]")
    
    def _validate_index(self, index: int) -> None:
        card_count = self.card_count
        if index < -card_count or index >= card_count:
            raise IndexError(f"Hand index {index} outside hand size [{self.row_count}, {self.column_count}] ")

    def _deal_from(self, cards: Cards) -> None:
        self.flipped_card_count = 0
        self._fingers.clear()
        for _ in range(self.original_card_count):
            self._fingers.append(Finger(cards._draw_card()))
    
    def _try_clear(self, column: int) -> bool:  # Cannot be negative
        card = self._fingers[column]._card
        for i in range(column, self.card_count, self.column_count):
            finger = self._fingers[i]
            if not finger.is_flipped or finger._card != card:
                return False
        for i in reversed(range(column, self.card_count, self.column_count)):
            del self._fingers[i]        
        assert len(self._fingers) % self.row_count == 0
        self.flipped_card_count -= self.row_count
        return True
    
    def _replace_card(self, index: int, card: int) -> int:
        finger = self._fingers[index]
        self.flipped_card_count += not finger.is_flipped
        replaced_card = finger._replace_card(card)
        self._try_clear(index % self.column_count)
        return replaced_card

    def _flip_card(self, index: int) -> None:
        self._fingers[index]._flip_card()
        self.flipped_card_count += 1
        self._try_clear(index % self.column_count)

    def _flip_all_cards(self) -> None:
        for finger in self._fingers:
            if not finger.is_flipped:
                finger._flip_card()
        self.flipped_card_count = self.card_count
        for column in reversed(range(self.column_count)):  # Reverse to avoid skipping
            self._try_clear(column)
        return self.sum_flipped_cards()

    def _render(self, *, xray: bool = False) -> list[str]:
        card_width = max(max(len(str(finger._card)) for finger in self._fingers), 2)
        empty = " " * card_width
        divider = ("+" + "-" * card_width) * self.column_count + "+"
        template = f"|{{:>{card_width}}}" * self.column_count + "|"
        lines = [divider] * (self.row_count * 2 + 1)
        lines[1::2] = (
            template.format(*(
                str(finger._card) if finger.is_flipped or xray else empty
                for finger in self._fingers[i*self.column_count:(i + 1)*self.column_count])
            ) for i in range(self.row_count)
        )
        return lines


@dataclass(slots=True)
class Flip:
    """An action controller for flipping the first two cards."""

    _hand: Hand
    _first_index: int | None = None
    _second_index: int | None = None

    @property
    def flipped_count(self) -> int:
        """The number of cards that have been flipped so far."""

        if self._first_index is None:
            return 0
        elif self._second_index is None:
            return 1
        return 2

    def flip_card(self, row_or_index: int, column: int | None = None, /) -> int:
        """Flip a single card, seeing its value.
        
        This method accepts either the row and column of the card to
        flip or its index in the flat list of fingers.
        """

        index = self._hand.index(row_or_index, column)
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
        self._hand._flip_card(self._first_index)
        self._hand._flip_card(self._second_index)


@dataclass
class Turn:
    """An action controller for taking a turn."""

    DISCARD_AND_FLIP: ClassVar[Final[str]] = "discard and flip"
    PLACE_DRAWN_CARD: ClassVar[Final[str]] = "place drawn card"
    PLACE_FROM_DISCARD: ClassVar[Final[str]] = "place from discard"

    _hand: Hand
    _cards: Cards
    _action: str | None = None
    _selected_card: int | None = None
    _replaced_or_flipped_index: int | None = None

    def draw_card(self) -> int:
        """Draw a card from the pile, seeing its value.
        
        This method enables `discard_and_flip` and `place_drawn_card`
        but disables `place_from_discard`.
        """

        if self._action is not None:
            raise RuleError("Cannot take multiple actions in a single turn")
        if self._selected_card is not None:
            raise RuleError("Cannot draw more than one card")
        self._selected_card = self._cards._next_draw_card
        return self._selected_card

    def discard_and_flip(self, row_or_index: int, column: int | None = None, /) -> int:
        """Discard the drawn card and flip a card in the player's hand.
        
        This method accepts either the row and column of the card to
        flip or its index in the flat list of fingers. `draw_card` must
        be called first.
        """

        if self._action is not None:
            raise RuleError("Cannot take multiple actions in a single turn")
        if self._selected_card is None:
            raise RuleError("Cannot discard and flip without first drawing a card")
        if self._hand.are_all_cards_flipped:
            raise RuleError("Cannot discard and flip when all cards are flipped")
        self._action = Turn.DISCARD_AND_FLIP
        self._replaced_or_flipped_index = self._hand.index(row_or_index, column)
        if self._hand[self._replaced_or_flipped_index].is_flipped:
            raise RuleError(f"Cannot flip already-flipped card at {self._replaced_or_flipped_index}")
        return self._hand[self._replaced_or_flipped_index]._card

    def place_drawn_card(self, row_or_index: int, column: int | None = None, /) -> int:
        """Replace a card in the player's hand with the drawn card.
        
        This method accepts either the row and column of the card to
        replace or its index in the flat list of fingers. `draw_card`
        must be called first.
        """
        
        if self._action is not None:
            raise RuleError("Cannot take multiple actions in a single turn")
        if self._selected_card is None:
            raise RuleError("Cannot place drawn card without first drawing a card")
        self._action = Turn.PLACE_DRAWN_CARD
        self._replaced_or_flipped_index = self._hand.index(row_or_index, column)
        return self._hand[self._replaced_or_flipped_index]._card

    def place_from_discard(self, row_or_index: int, column: int | None = None, /) -> int:
        """Replace a card in the player's hand with the topmost discard.
        
        This method accepts either the row and column of the card to
        replace or its index in the flat list of fingers. `draw_card`
        may not be called first.
        """

        if self._action is not None:
            raise RuleError("Cannot take multiple actions in a single turn")
        if self._selected_card is not None:
            raise RuleError("Cannot place from discard after drawing a card")
        self._selected_card = self._cards.last_discard
        self._action = Turn.PLACE_FROM_DISCARD
        self._replaced_or_flipped_index = self._hand.index(row_or_index, column)
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
            replaced_card = self._hand._replace_card(self._replaced_or_flipped_index, self._cards.last_discard)
            self._cards._replace_discard_card(replaced_card)


@dataclass(slots=True)
class Player(abc.ABC):
    """A Skyjo player.
    
    To implement your own Skyjo bot, inherit from this class and
    override the `flip` and `turn` methods. See main.py for examples.
    To ensure your bot doesn't cheat or break the simulation:

      - Do not modify any provided attributes, e.g. `hand` or `score`
        on `Player` or `cards` on `State`. Modifying new attributes
        defined by your subclass is totally fine.
      - Do not access private variables or methods, all of which are
        prefixed with an underscore e.g. `Finger._card`.

    See README.md for how to run the simulation.
    """

    hand: Hand = field(default_factory=Hand, repr=False)
    """The players current hand of cards."""

    score: int = 0
    """The player's cumulative game score."""

    _flip_elapsed: Average = Average()
    """The amount of time spent flipping cards."""

    _turn_elapsed: Average = Average()
    """Cumulative number of seconds spent taking turns."""

    def __str__(self) -> None:
        return type(self).__qualname__
        
    @abc.abstractmethod
    def flip(self, state: State, action: Flip) -> None:
        """Start the game by flipping two cards in your hand.
        
        To flip a card, call `action.flip_card` with its index or row
        and column. Doing so will return the value of the flipped card,
        but note `self.hand` won't be updated until afterwards to ensure
        your flips don't influence later players'.
        """

    @abc.abstractmethod
    def turn(self, state: State, action: Turn) -> None:
        """Take a turn as part of a round.
        
        Either `action.draw_card` or `action.place_from_discard()`. The
        former will allow you to then `action.place_drawn_card()` or
        `action.discard_and_flip()`. The latter three methods take the
        index or row and column of the card in your hand you wish to
        replace or flip.
        """


@dataclass(slots=True)
class Debugger:
    """Used to interactively debug games."""

    _continuing: bool = False
    
    def reset(self) -> None:
        self._continuing = False

    def print(self, message: str, symbol: str = "", end="\n") -> None:
        if symbol:
            prefix = len(symbol) + 1
            print(symbol, textwrap.indent(message, " " * prefix)[prefix:], end=end)
        else:
            print(textwrap.indent(message, "  "), end=end)

    def info(self, message: str) -> None:
        self.print(message, symbol="-")
    
    def alert(self, message: str) -> None:
        self.print(message, symbol="*")

    def display(self, state: State, *, xray: bool = False) -> None:
        symbol = "#"
        for line in state._render(xray=xray):
            print(symbol, line)
            symbol = " "
    
    def traceback(self, exception: Exception) -> None:
        for trace in traceback.format_exception(exception):
            self.print(trace, end="")

    def inspect(self, o: object) -> None:
        items: list[tuple[str, object]] = []
        for name in getattr(type(o), "__slots__", ()):
            if not name.startswith("_"):
                items.append((name, getattr(o, name)))
        for name, descriptor in vars(type(o)).items():
            if not name.startswith("_"):
                if isinstance(descriptor, property):
                    items.append((name, getattr(o, name)))
                elif isinspectable(descriptor):
                    if inspect.isfunction(descriptor):
                        items.append((name, getattr(o, name)()))
        width = max(len(name) for name, _ in items)
        for name, value in items:
            self.print(f"{name:>{width}} = {value!r}")

    def prompt(
        self,
        state: State,
        namespace: dict[str, object],
        context: BaseException | None = None,
    ) -> None:
        if self._continuing and context is None:
            return
        self.display(state)
        try:
            while True:
                command, _, text = input("> ").strip().partition(" ")
                if not command or command in {"s", "step"}:
                    break
                elif command in {"g", "game"}:
                    for line in state._render():
                        self.print(line)
                elif command in {"i", "inspect"}:
                    self.inspect(state)
                elif command in {"c", "continue"}:
                    self._continuing = True
                    break
                elif command in {"p", "print", "e", "eval"}:
                    try:
                        print(repr(eval(text, globals(), namespace)))
                    except BaseException as error:
                        if error.__context__ is context:
                            error.__context__ = None
                        self.traceback(error)
                elif command in {"x", "xray"}:
                    self.display(state, xray=True)
                elif command in {"h", "help"}:
                    self.print("s/step      progress the simulation")
                    self.print("<enter>     same as step")
                    self.print("c/continue  progress until the game finishes")
                    self.print("g/game      reprint the game state")
                    self.print("i/inspect   print state properties")
                    self.print("e/eval      evaluate the following expression")
                    self.print("p/print     same as eval")
                    self.print("x/xray      show unflipped cards in player hands")
                    self.print("h/help      show this dialog")
                    self.print("q/quit      exit the simulation")
                elif command in {"q", "quit"}:
                    exit(0)
                else:
                    self.alert(f"unknown command {text}, see help for commands")
        except (KeyboardInterrupt, EOFError):
            exit(0)


@dataclass(slots=True)
class State:
    """The entire state of a Skyjo game.
    
    This class is reusable but not threadsafe. Call `play()` to reset
    and simulate a single game of Skyjo. Games can be made deterministic
    by passing a seeded `_rng: random.Random` to the constructor.
    """

    players: list[Player] = field()
    """The list of participating players."""

    cards: Cards = field(default=None)
    """The draw and discard piles."""

    round_index: int = field(init=False, default=0)
    """The 0-based index of the current round."""

    turn_index: int = field(init=False, default=0)
    """The 0-based index of the current turn."""

    round_starter_index: int = field(init=False, default=0)
    """The index of the player that goes first each turn this round."""

    round_ender_index: int | None = field(init=False, default=None)
    """The index of the player that flipped all their cards first."""
    
    _rng: random.Random = field(default_factory=random.Random)
    _repro: object | None = field(default=None)

    def __post_init__(self) -> None:
        if self.cards is None:
            self.cards = Cards(_rng=self._rng)

    @property
    def is_round_ending(self) -> bool:
        """Whether a player has flipped all their cards this round."""

        return self.round_ender_index is not None
    
    @property
    def player_index(self) -> int:
        """The index of the current player with action."""

        return (self.round_starter_index + self.turn_index) % self.player_count
        
    @property
    def player_count(self) -> int:
        """The number of players in this game."""

        return len(self.players)
    
    @property
    def player_indices(self) -> range:
        """Shorthand for the range of player indices."""

        return range(self.player_count)

    @property
    def player(self) -> Player:
        """The current player with action."""
        
        return self.players[self.player_index]
    
    @property
    def next_player_index(self) -> int:
        """The index of the next player to take a turn."""

        return (self.player_index + 1) % self.player_count

    @property
    def next_player(self) -> Player:
        """The next player to take a turn."""

        return self.players[self.next_player_index]
    
    @property
    def previous_player_index(self) -> int:
        """The index of the last player to take a turn."""

        return (self.player_index + self.player_count - 1) % self.player_count

    @property
    def previous_player(self) -> Player:
        """The last player to take a turn."""

        return self.players[self.previous_player_index]
    
    @property
    def turn_order(self) -> int:
        """The index in the turn order of the current player."""

        return self.count_turn_order()
    
    @property
    def turns_taken_count(self) -> int:
        """The number of turns the current player has taken."""

        return self.count_turns_taken()

    @inspectable
    def find_highest_flipped_card_sum_player_index(self) -> None:
        """Find player with the high sum of flipped card values."""

        return max(self.player_indices, key=lambda i: self.players[i].hand.sum_flipped_cards())

    @inspectable
    def find_highest_score_player_index(self) -> int:
        """Get the index of the player with the highest overall score."""

        return max(self.player_indices, key=lambda i: self.players[i].score)
    
    @inspectable
    def find_highest_score_player(self) -> Player:
        """Get the player with the highest overall score."""

        return max(self.players, key=lambda player: player.score)

    @inspectable
    def find_lowest_score_player_index(self) -> int:
        """Get the index of the player with the lowest overall score."""

        return min(self.player_indices, key=lambda i: self.players[i].score)
     
    @inspectable
    def find_lowest_score_player(self) -> int:
        """Get the player with the lowest overall score."""

        return min(self.players, key=lambda player: player.score)
    
    @inspectable
    def count_turn_order(self, player_index: int | None = None) -> int:
        """Get the index in the turn order of the given player.
        
        The first player to take their turn will have order 0. The last
        will have order `self.player_count - 1`. `player_index` defaults
        to the current player.
        """

        if player_index is None:
            player_index = self.player_index
        reference = self.round_ender_index if self.is_round_ending else self.round_starter_index
        return (player_index - reference) % self.player_count
    
    def count_turn_differential(self, player_index: int, from_player_index: int | None = None) -> int:
        """Determine whether a player is a turn ahead or behind.
        
        Returns 1 if `player_index` takes their nth turn before
        `from_player_index` does. Returns -1 in the opposite case.
        Returns 0 if both indices are the same. `from_player_index`
        defaults to the current player.

        All other players are behind the first player in the turn order
        and all other players are ahead of the last player in the turn
        order at all times.
        """

        if from_player_index is None:
            from_player_index = self.player_index
        player_turn_order = self.count_turn_order(player_index) 
        from_player_turn_order = self.count_turn_order(from_player_index)
        return (
            1 if player_turn_order < from_player_turn_order  # They go first
            else -1 if player_turn_order > from_player_turn_order  # We go first
            else 0  # Same player specified twice
        )
    
    @inspectable
    def count_turns_taken(self, player_index: int | None = None) -> int:
        """Determine how many turns a given player has taken.
        
        Does not count the current turn if the player has action. For
        example, on the very first turn, all players will have taken
        zero turns. `player_index` defaults to the current player.
        """

        if player_index is None:
            player_index = self.player_index
        has_taken_current_turn = self.count_turn_order(player_index) < self.turn_index % self.player_count
        return self.turn_index // self.player_count + has_taken_current_turn

    @inspectable
    def count_turns_to_end(self, player_index: int | None = None) -> int:
        """Determine the minimum number of turns a player can finish in."""

        if player_index is None:
            return self.player_index
        return self.players[player_index].hand.unflipped_card_count
    
    @inspectable
    def count_minimum_turns_remaining(self, player_index: int | None = None) -> int:
        """Determine the minimum number of turns the current player has.
        
        Accounts for the relative position in the turn order of other
        players. Does not include the current turn if the specified
        player has action.
        """

        return max(
            self.count_turns_to_end(i)
            for i in self.player_indices
            if i != player_index
        ) + 1  # Endgame
    
    def play(self, debugger: Debugger | None = None) -> None:
        """Play a single game of Skyjo, returning final scores."""

        try:
            # If we throw here, we can potentially recover our state for debugging.
            self._repro = self._rng.getstate()

            if self.player_count < 3:
                raise RuleError(f"Must have at least 3 players, got {self.player_count}")
            elif self.player_count > 8:
                raise RuleError(f"Must have 8 or fewer players, got {self.player_count}")

            if debugger:
                debugger.reset()
                debugger.alert("new game ".ljust(78, "*"))

            self.round_index = 0
            self.turn_index = 0
            self.round_starter_index = 0
            self.round_ender_index = None
            self.cards._reset()
            self.cards._shuffle()        
            for player in self.players:
                player.score = 0
                player.hand._deal_from(self.cards)

            if debugger:
                debugger.info("fully reset game")
                debugger.prompt(self, locals())

            flips = [Flip(player.hand) for player in self.players]
            for player, flip in zip(self.players, flips):
                start_time = time.monotonic()
                player.flip(self, flip)
                player._flip_elapsed = time.monotonic() - start_time
            for player, flip in zip(self.players, flips):  # Do separately
                flip._apply_to_hand()
            del flips
            self.round_starter_index = self.find_highest_flipped_card_sum_player_index()

            if debugger:
                debugger.info("flipped cards")
                debugger.info(f"player {self.round_starter_index} starts with the highest hand")
                debugger.prompt(self, locals())

            while True:
                while not self.is_round_ending:
                    for _ in range(self.player_count):
                        player = self.player
                        player_index = self.player_index

                        turn = self._turn(player)
                        if debugger:
                            debugger.info(f"player {player_index} chose to {turn._action}")
                            debugger.prompt(self, locals())
    
                        self.turn_index += 1  # Careful here, this changes internal state
                        if player.hand.are_all_cards_flipped:
                            round_ender_index = self.round_ender_index = player_index
                            break

                    self.round_index += 1

                if debugger:
                    debugger.info("entering endgame")

                for _ in range(self.player_count - 1):
                    player = self.player
                    player_index = self.player_index

                    turn = self._turn(player)
                    if debugger:
                        debugger.info(f"player {player_index} chose to {turn._action}")
                        debugger.prompt(self, locals())

                    self.turn_index += 1

                round_scores = [player.hand._flip_all_cards() for player in self.players]
                round_ender_score = round_scores[round_ender_index]
                if min(round_scores) < round_ender_score or round_scores.count(round_ender_score) > 1:
                    round_scores[round_ender_index] *= 2
                for player, round_score in zip(self.players, round_scores):
                    player.score += round_score
                
                self.round_index += 1  # So we get the right count after breaking
                
                if max(self.players, key=lambda player: player.score).score >= 100:
                    break

                if debugger:
                    debugger.alert(f"scores are {', '.join(str(player.score) for player in self.players)}")

                self.round_starter_index = round_ender_index
                self.round_ender_index = None
                self.cards._reset()
                self.cards._shuffle()
                for player in self.players:
                    player.hand._deal_from(self.cards)

            if debugger:
                debugger.alert(f"player {self.find_lowest_score_player_index()} wins")
                debugger.prompt(self, locals())

        except RuleError as error:
            error.repro = self._repro
            if debugger:
                debugger.alert(f"a rule was broken: {error}")
                debugger.traceback(error)
                debugger.prompt(self, locals(), context=error)
            raise

    def _turn(self, player: Player) -> Turn:
        turn = Turn(self.player.hand, self.cards)
        start_time = time.monotonic()
        player.turn(self, turn)
        player._turn_elapsed += time.monotonic() - start_time
        turn._apply_to_hand_and_cards()
        return turn

    def _render(self, *, xray: bool = False) -> list[str]:
        first_line = (
            f"discard: {self.cards.last_discard}"
            f"  draw: ({self.cards._next_draw_card})"
            f"  round: {self.round_index}"
            f"  turn: {self.turn_index}/{self.turn_index%self.player_count}"
            + (f"  (ending)" if self.is_round_ending else "")
        )

        hands_render = [""]
        for index, player in enumerate(self.players):
            hand_render = player.hand._render(xray=xray)
            base_length = len(hands_render[0])
            if index == self.player_index:
                hand_render.append("^" * len(hand_render[-1]))
            for i, line in enumerate(hand_render):
                if i == len(hands_render):
                    hands_render.append(" " * base_length)
                if index > 0:
                    hands_render[i] += "  "
                hands_render[i] += line

        return [
            first_line,
            *hands_render,
        ]


@dataclass(slots=True)
class Outcomes:
    player_count: int
    game_count: int = 0
    average_round_count: Average = Average()
    average_scores: list[Average] = field(default=None)
    average_flip_elapseds: list[Average] = field(default=None)
    average_turn_elapseds: list[Average] = field(default=None)
    win_counts: list[int] = field(default=None)

    def __post_init__(self) -> None:
        if self.average_scores is None:
            self.average_scores = [Average()] * self.player_count
        if self.average_flip_elapseds is None:
            self.average_flip_elapseds = [Average()] * self.player_count
        if self.average_turn_elapseds is None:
            self.average_turn_elapseds = [Average()] * self.player_count
        if self.win_counts is None:
            self.win_counts = [0] * self.player_count

    def __add__(self, other: object) -> Outcomes:
        if not isinstance(other, Outcomes):
            raise NotImplemented
        if self.player_count != other.player_count:
            raise ValueError(f"Player counts {self.player_count} and {other.player_count} do not match")
        game_count = self.game_count + other.game_count
        return Outcomes(
            player_count=self.player_count,
            game_count=game_count,
            average_round_count=self.average_round_count + other.average_round_count,
            average_scores=[x + y for x, y in zip(self.average_scores, other.average_scores)],
            average_flip_elapseds=[x + y for x, y in zip(self.average_flip_elapseds, other.average_flip_elapseds)],
            average_turn_elapseds=[x + y for x, y in zip(self.average_turn_elapseds, other.average_turn_elapseds)],
            win_counts=[x + y for x, y in zip(self.win_counts, other.win_counts)],
        )


def simulate(
    players: list[Player],
    games: int = 1,
    *,
    seed: int | float | str | bytes | bytearray | None = None,
    repro: object | None = None,
    rng: random.Random | None = None,
    interactive: bool = False,
    processes: int = multiprocessing.cpu_count(),
    subprocess: bool = False,
) -> Outcomes:
    """Play `games` rounds of Skyjo, aggregating game statistics.    

    Use `seed` or `rng` to make rounds deterministic. Use `interactive`
    to debug or see how rounds play out. Use `processes` to run games
    in parallel.
    """
    
    if (seed is not None) + (repro is not None) + (rng is not None) > 1:
        raise ValueError("Parameters seed, repro, and rng are mutually exclusive")
    elif seed is not None:
        rng = random.Random(seed)
    elif repro is not None:
        rng = random.Random()
        rng.setstate(pickle.loads(bytes.fromhex(repro)))

    if not subprocess:
        start_time = time.monotonic()

    if processes > 1:
        if rng is not None:
            raise ValueError("Cannot multiprocess with fixed rng")
        if repro is not None:
            raise ValueError("Cannot multiprocess a repro")
        
        partial = functools.partial(simulate, players, processes=1, subprocess=True)
        chunks = tuple(filter(None, (games // processes + (i < games % processes) for i in range(processes))))
        processes = len(chunks)
        try:
            with multiprocessing.Pool(processes=processes) as pool:
                outcomes = sum(pool.map(partial, chunks), start=Outcomes(player_count=len(players)))
        except RuleError as error:
            if error.repro is not None:
                simulate(players, interactive=True, repro=error.repro)
            exit(1)

    elif processes != 1:
        raise ValueError(f"Invalid process count {processes}, must be 1 or larger")
    
    else:
        if rng is None:
            rng = random.Random()
        
        state = State(players, _rng=rng)
        outcomes = Outcomes(player_count=len(players))
        debugger = Debugger() if interactive else None

        for _ in range(games):
            try:
                state.play(debugger)
            except RuleError as error:
                if repro is None and error.repro is not None:
                    print(f"* repro: {pickle.dumps(error.repro).hex()}")
                if interactive:
                    exit(1)

            outcomes.game_count += 1
            outcomes.average_round_count += state.round_index
            for index, player in enumerate(state.players):
                outcomes.average_scores[index] += player.score
                outcomes.average_flip_elapseds[index] += player._flip_elapsed
                outcomes.average_turn_elapseds[index] += player._turn_elapsed
            outcomes.win_counts[state.find_lowest_score_player_index()] += 1

    if not subprocess:
        delta = datetime.timedelta(seconds=time.monotonic() - start_time)
        print(f"= Played {games:,} games on {processes} cores in {delta}")
        print(f"= Average rounds: {outcomes.average_round_count:.2f}")
        for index, player in enumerate(players):
            win_count = outcomes.win_counts[index]
            win_percent = win_count / games * 100
            average_score = outcomes.average_scores[index]
            average_turn_elapsed = outcomes.average_turn_elapseds[index]
            print(
                f"= {player}[{index}]: "
                f"{win_count} wins ({win_percent:.2f}%), "
                f"mean score {average_score:.2f}, "
                f"mean turn elapsed {average_turn_elapsed:.2f}"
            )

    return outcomes