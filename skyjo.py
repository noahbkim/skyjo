from __future__ import annotations

import abc
import datetime
import functools
import multiprocessing
import pickle
import random
import textwrap
import time
import traceback
from dataclasses import dataclass, field
from typing import ClassVar, Collection, Final, Iterator, overload

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
    pile containing everything else (with the next -2 on top).
    
    When a card is `drawn`, the card at `_draw_index` is retrieved then
    the index is incremented. We don't bother overwriting draw cards
    with e.g. `None` because they should never be accessed. If `_draw`
    was called four times, our indices would now be:

          _discard_index
          v
        [-2, -1, 0, 1, 2, 3, 4, ...] < _buffer
                          ^
                          _draw_index

    When a card is discarded, `_discard_index` is incremented then the
    new corresponding element is set to its value. If we discarded our
    four cards in reverse order, we would have:

                       _discard_index
                       v
        [1, 0, -1, -2, 2, 3, 4, ...] < _buffer
                          ^
                          _draw_index
    
    Once we've exhausted our draw pile, we can shuffle the buried
    discards and slice them back into the draw pile.
    
    To save ourselves the trouble of collecting all cards still in hand
    at the end of each round, we hold a reference to the original list
    of cards in our deck in `_deck` and reset to that instead.
    """

    _rng: random.Random = field(default_factory=random.Random)
    """A shared random number generator for deterministic shuffling."""

    _deck: Collection[int] = field(default=DECK)
    """The original collection of cards to reset the deck to."""

    _buffer: list[int] = field(init=False)
    """A container for the discard and draw piles."""

    _discard_index: int = 0
    """The index of the topmost discard."""

    _draw_index: int = 1
    """The index of the topmost card in the draw pile."""

    def __hash__(self) -> None:
        return id(self)

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
    def _next_draw(self) -> int:
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
    
    def _discard_card(self, card: int) -> None:
        assert self._discard_index < len(self._buffer), "Deck has too many cards!"
        self._discard_index += 1
        self._buffer[self._discard_index] = card

    def _replace_discard_card(self, card: int) -> None:
        assert self._discard_index < len(self._buffer), "Deck has too many cards!"
        self._buffer[self._discard_index] = card

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
        0 [  0,  1,  2,  3,
        1    4,  5,  6,  7,
        2    8,  9, 10, 11, ]
        ^          ^
        rows       indices
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
    def are_all_cards_flipped(self) -> bool:
        """Whether all cards in this hand have been flipped."""

        return self.flipped_card_count == self.card_count
    
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
    
    def find_first_unflipped_or_highest_card_index(self) -> int:
        """Get the first unflipped card or the highest card; must exist."""

        index = self.find_first_unflipped_card_index()
        if index is not None:
            return index
        index = self.find_highest_card_index()
        assert index is not None
        return index

    def _get_valid_index(self, row: int, column: int | None) -> None:
        index = row * self.column_count + column if column is not None else row
        if index < -len(self._fingers) or index >= len(self._fingers):
            raise RuleError(f"Card ({row}, {column}) is outside hand dimensions")
        return index

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

    def flip_card(self, row: int, column: int | None = None, /) -> int:
        """Flip a single card, seeing its value.
        
        This method accepts either the row and column of the card to
        flip or its index in the flat list of fingers.
        """

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
        self._selected_card = self._cards._next_draw
        return self._selected_card

    def discard_and_flip(self, row: int, column: int | None = None, /) -> int:
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
        self._replaced_or_flipped_index = self._hand._get_valid_index(row, column)
        if self._hand[self._replaced_or_flipped_index].is_flipped:
            raise RuleError(f"Cannot flip already-flipped card at {self._replaced_or_flipped_index}")
        return self._hand[self._replaced_or_flipped_index]._card

    def place_drawn_card(self, row: int, column: int | None = None, /) -> int:
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
        self._replaced_or_flipped_index = self._hand._get_valid_index(row, column)
        return self._hand[self._replaced_or_flipped_index]._card

    def place_from_discard(self, row: int, column: int | None = None, /) -> int:
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

    hand: Hand = field(default_factory=Hand)
    """The players current hand of cards."""

    score: int = 0
    """The player's cumulative game score."""

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

        return (self.round_starter_index + self.turn_index) % len(self.players)

    @property
    def player(self) -> Player:
        """The current player with action."""
        
        return self.players[self.player_index]
    
    @property
    def next_player_index(self) -> int:
        """The index of the next player to take a turn."""

        return (self.player_index + 1) % len(self.players)

    @property
    def next_player(self) -> Player:
        """The next player to take a turn."""

        return self.players[self.next_player_index]
    
    @property
    def previous_player_index(self) -> int:
        """The index of the last player to take a turn."""

        return (self.player_index + len(self.players) - 1) % len(self.players)

    @property
    def previous_player(self) -> Player:
        """The last player to take a turn."""

        return self.players[self.previous_player_index]

    def find_highest_flipped_card_sum_player_index(self) -> None:
        """Find player with the high sum of flipped card values."""

        return max(range(len(self.players)), key=lambda i: self.players[i].hand.sum_flipped_cards())

    def find_highest_score_player_index(self) -> int:
        """Get the index of the player with the highest overall score."""

        return max(range(len(self.players)), key=lambda i: self.players[i].score)
    
    def find_highest_score_player(self) -> Player:
        """Get the player with the highest overall score."""

        return max(self.players, key=lambda player: player.score)

    def find_lowest_score_player_index(self) -> int:
        """Get the index of the player with the lowest overall score."""

        return min(range(len(self.players)), key=lambda i: self.players[i].score)
     
    def find_lowest_score_player(self) -> int:
        """Get the player with the lowest overall score."""

        return min(self.players, key=lambda player: player.score)
    
    def play(self, debugger: Debugger | None = None) -> None:
        """Play a single game of Skyjo, returning final scores."""

        try:
            # If we throw here, we can potentially recover our state for debugging.
            self._repro = self._rng.getstate()

            if len(self.players) < 3:
                raise RuleError(f"Must have at least 3 players, got {len(self.players)}")
            elif len(self.players) > 8:
                raise RuleError(f"Must have 8 or fewer players, got {len(self.players)}")

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
                player.flip(self, flip)
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
                    for _ in range(len(self.players)):
                        turn = self._turn(self.player)
                        player_index = self.player_index
                        self.turn_index += 1

                        if debugger:
                            debugger.info(f"player {player_index} chose to {turn._action}")
                            debugger.prompt(self, locals())

                        if self.player.hand.are_all_cards_flipped:
                            round_ender_index = self.round_ender_index = self.player_index
                            break
                
                if debugger:
                    debugger.info("entering endgame")

                for _ in range(len(self.players) - 1):
                    turn = self._turn(self.player)
                    player_index = self.player_index
                    self.turn_index += 1
                    
                    if debugger:
                        debugger.info(f"player {player_index} chose to {turn._action}")
                        debugger.prompt(self, locals())

                round_scores = [player.hand._flip_all_cards() for player in self.players]
                round_ender_score = round_scores[round_ender_index]
                if min(round_scores) < round_ender_score or round_scores.count(round_ender_score) > 1:
                    round_scores[round_ender_index] *= 2
                for player, round_score in zip(self.players, round_scores):
                    player.score += round_score
                
                self.round_index += 1  # So we get the right count after breaking
                
                if max(self.players, key=lambda player: player.score).score >= 100:
                    break

                self.round_starter_index = round_ender_index
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
        player.turn(self, turn)
        turn._apply_to_hand_and_cards()
        return turn

    def _render(self, *, xray: bool = False) -> list[str]:
        first_line = (
            f"discard: {self.cards.last_discard}"
            f"  draw: {self.cards._next_draw}"
            f"  round: {self.round_index}"
            f"  turn: {self.turn_index // len(self.players)} + {self.turn_index%len(self.players)}"
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
            outcomes.win_counts[state.find_lowest_score_player_index()] += 1

        outcomes._finalize()

    if not subprocess:
        delta = datetime.timedelta(seconds=time.monotonic() - start_time)
        print(f"= Played {games:,} games on {processes} cores in {delta}")
        print(f"= Average rounds: {outcomes.average_round_count:.2f}")
        for index, player in enumerate(players):
            win_count = outcomes.win_counts[index]
            win_percent = win_count / games * 100
            average_score = outcomes.average_scores[index]
            print(f"= {player}[{index}]: {win_count} wins ({win_percent:.2f}%), average score {average_score:.2f}")

    return outcomes