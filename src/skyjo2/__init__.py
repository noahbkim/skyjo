"""A framework for simulating games of Skyjo.

From https://magilano.com/en:

> Skyjo is a fast and exciting card game where players compete to
> achieve the lowest score by flipping, trading, and gathering cards
> over several rounds.
"""

from .game import CARD_COUNTS as CARD_COUNTS
from .game import CARD_VALUES as CARD_VALUES
from .game import DECK as DECK
from .game import HAND_COLUMNS as HAND_COLUMNS
from .game import HAND_ROWS as HAND_ROWS
from .game import Finger as Finger
from .game import Game as Game
from .game import GameState as GameState
from .game import Player as Player
