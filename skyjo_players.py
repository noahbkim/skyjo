import logging

import numpy as np

import abstract as ab
import skyjo_immutable as sj


class RandomPlayer(ab.AbstractPlayer):
    """A player that selects actions randomly"""

    def select_action(self, state: sj.ImmutableState) -> sj.SkyjoAction:
        return np.random.choice(state.valid_actions)


class GreedyExpectedValuePlayer(ab.AbstractPlayer):
    """A player that treats face-down cards as having the current expected value based on remaining cards.
    Then, decides actions based on whatever has lowest expected point value.
    Note: This player doesn't account for when to end the round."""

    def select_action(self, state: sj.ImmutableState) -> sj.SkyjoAction:
        remaining_expected_value = state.remaining_card_counts.expected_value
        best_action_idx = np.argmin(
            [
                self.expected_action_value(state, action, remaining_expected_value)
                for action in state.valid_actions
            ]
        )
        return state.valid_actions[best_action_idx]

    def expected_action_value(
        self,
        state: sj.ImmutableState,
        action: sj.SkyjoAction,
        unknown_expected_value: float,
    ) -> float:
        match action.action_type:
            case sj.SkyjoActionType.DRAW:
                return self._expected_draw_value(state, action, unknown_expected_value)
            case sj.SkyjoActionType.PLACE_FROM_DISCARD:
                return self._expected_place_from_discard_value(
                    state, action, unknown_expected_value
                )
            case sj.SkyjoActionType.PLACE_DRAWN:
                return self._expected_place_from_draw_value(
                    state, action, unknown_expected_value
                )
            case sj.SkyjoActionType.DISCARD_AND_FLIP:
                return self._expected_discard_flip_value(
                    state, action, unknown_expected_value
                )
            case sj.SkyjoActionType.END_ROUND:
                return 0
            case sj.SkyjoActionType.START_ROUND:
                return 0
            case sj.SkyjoActionType.INITIAL_FLIP:
                return 0
            case _:
                raise ValueError(f"Unknown action type: {action.action_type}")

    def _expected_place_from_discard_value(
        self,
        state: sj.ImmutableState,
        action: sj.SkyjoAction,
        unknown_expected_value: float | None = None,
    ) -> float:
        curr_card = state.hands[state.curr_player].get_card(
            action.row_idx, action.col_idx
        )
        curr_col_cards = (
            [
                state.hands[state.curr_player].get_card(r, action.col_idx)
                for r in range(sj.NUM_ROWS)
                if r != action.row_idx
            ],
        )
        # Can clear column (expected value of action is then -point_value for each cleared card)
        if all([state.discard_pile.top_card == card for card in curr_col_cards]):
            return -sj.NUM_ROWS * state.discard_pile.top_card.point_value
        if curr_card == sj.Card.FACE_DOWN:
            return state.discard_pile.top_card.point_value - unknown_expected_value
        return state.discard_pile.top_card.point_value - curr_card.point_value

    def _expected_draw_value(
        self,
        state: sj.ImmutableState,
        action: sj.SkyjoAction,
        unknown_expected_value: float,
    ) -> float:
        highest_curr_card_value = max(
            state.hands[state.curr_player].get_card(r, c).point_value
            for r, c in state.hands[state.curr_player].non_cleared_indices
        )
        # small incentive to draw a card over placing for when placing from discard wouldn't change value at all
        return min(unknown_expected_value - highest_curr_card_value, -0.01)

    def _expected_place_from_draw_value(
        self,
        state: sj.ImmutableState,
        action: sj.SkyjoAction,
        unknown_expected_value: float,
    ) -> float:
        curr_card = state.hands[state.curr_player].get_card(
            action.row_idx, action.col_idx
        )
        curr_col_cards = (
            [
                state.hands[state.curr_player].get_card(r, action.col_idx)
                for r in range(sj.NUM_ROWS)
                if r != action.row_idx
            ],
        )
        # Can clear column (expected value of action is then -point_value for each cleared card)
        if all([state.drawn_card == card for card in curr_col_cards]):
            return -sj.NUM_ROWS * state.drawn_card.point_value
        if curr_card == sj.Card.FACE_DOWN:
            return state.drawn_card.point_value - unknown_expected_value
        return state.drawn_card.point_value - curr_card.point_value

    def _expected_discard_flip_value(
        self,
        state: sj.ImmutableState,
        action: sj.SkyjoAction,
        unknown_expected_value: float,
    ) -> float:
        return 0


if __name__ == "__main__":
    np.random.seed(0)
    logging.basicConfig(level=logging.INFO)
    display_game = True
    # Simulate a single game
    players = [GreedyExpectedValuePlayer(), GreedyExpectedValuePlayer()]
    initial_counts = sj.CardCounts.create_initial_deck_counts()
    state = sj.ImmutableState(
        num_players=len(players),
        player_scores=np.zeros(len(players)),
        remaining_card_counts=initial_counts,
    ).start_round(initial_counts.generate_random_card())
    if display_game:
        state.display()
    while not state.game_is_over():
        action = players[state.curr_player].select_action(state)
        state = state.apply_action(action)
        if display_game:
            state.display()
            logging.info(
                f"Remaining expected value: {state.remaining_card_counts.expected_value}"
            )
    logging.info(f"Round turn counts: {state.round_turn_counts}")
