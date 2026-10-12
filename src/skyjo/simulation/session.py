"""One-decision-at-a-time games, usable by any interactive frontend."""

from copy import deepcopy
from dataclasses import replace

import numpy as np

from skyjo.engine import game as sj

from .decisions import PlayerDecision, get_decision, validate_decision
from .jobs import GameRandomStreams
from .recording import Recording, make_frame


class GameSession:
    def __init__(self, players, *, seed: int, metadata: list[dict]):
        if not 2 <= len(players) <= sj.PLAYER_COUNT or len(metadata) != len(players):
            raise ValueError("Expected 2–8 players with matching metadata")
        self.players = tuple(players)
        self.streams = GameRandomStreams.from_seed(seed)
        state = sj.start_round(
            sj.new(players=len(players), rng=self.streams.environment),
            rng=self.streams.environment,
        )
        self.recording = Recording(metadata, seed, [make_frame(state, 1)])

    @property
    def state(self):
        return self.recording.frames[-1].state

    @property
    def human_turn(self):
        return (
            not sj.get_round_over(self.state)
            and self.players[sj.get_player(self.state)] is None
        )

    def propose(self, *, diagnostics=True) -> PlayerDecision:
        """May run in a worker; call only once for the current decision."""
        if self.human_turn or sj.get_round_over(self.state):
            raise ValueError("No automated decision is available")
        return get_decision(
            self.players[sj.get_player(self.state)], self.state, diagnostics=diagnostics
        )

    def advance(
        self, decision: PlayerDecision | None = None, *, action: int | None = None
    ):
        """Commit an automated proposal, human action, or next-round transition."""
        frame = self.recording.frames[-1]
        if self.recording.complete:
            raise ValueError("Game is already complete")
        if sj.get_round_over(self.state):
            if decision is not None or action is not None:
                raise ValueError("Round boundaries do not accept actions")
            state = sj.start_next_round(self.state, rng=self.streams.environment)
            self.recording.frames.append(make_frame(state, frame.round_number + 1))
            return
        if self.human_turn:
            if (
                decision is not None
                or type(action) is not int
                or not 0 <= action < sj.MASK_SIZE
                or not sj.actions(self.state)[action]
            ):
                raise ValueError("Choose a legal human action")
            probabilities = np.zeros(sj.MASK_SIZE, dtype=np.float32)
            probabilities[action] = 1
            decision = PlayerDecision(probabilities)
        else:
            if decision is None or action is not None:
                raise ValueError("An automated turn requires a player decision")
            validate_decision(self.state, decision)
            action = int(
                self.streams.actions.choice(sj.MASK_SIZE, p=decision.probabilities)
            )
        state = sj.apply_action(self.state, action, rng=self.streams.environment)
        self.recording.frames[-1] = replace(
            frame, action=action, decision=deepcopy(decision)
        )
        self.recording.frames.append(make_frame(state, frame.round_number))
