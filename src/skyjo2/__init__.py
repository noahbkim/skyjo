"""A framework for simulating games of Skyjo.

From https://magilano.com/en:

> Skyjo is a fast and exciting card game where players compete to
> achieve the lowest score by flipping, trading, and gathering cards
> over several rounds.
"""

from ._game import CARD_COUNTS as CARD_COUNTS
from ._game import CARD_VALUES as CARD_VALUES
from ._game import DECK as DECK
from ._game import HAND_COLUMNS as HAND_COLUMNS
from ._game import HAND_ROWS as HAND_ROWS
from ._game import Finger as Finger
from ._game import Game as Game
from ._game import GameState as GameState
from ._game import Player as Player
from ._game import Rule as Rule
from ._play import Action as Action
from ._play import ActionKind as ActionKind
from ._play import Actor as Actor
from ._play import Transition as Transition
from ._play import iter_actions as iter_actions
from ._play import iter_outcomes as iter_outcomes
from ._play import iter_transitions as iter_transitions
from ._play import play_game as play_game
from ._play import play_round as play_round
from ._play import with_action as with_action
from ._play import with_transition as with_transition
from ._print import format_game as format_game
from ._print import print_game as print_game
