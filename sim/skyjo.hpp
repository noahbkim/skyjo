#pragma once

#include <cassert>
#include <cstdint>

#include <algorithm>
#include <array>
#include <exception>
#include <random>
#include <span>
#include <tuple>

#include <iostream>

#include "magic.hpp"

namespace skyjo {

using rng_t = std::mt19937;

using decksize_t = uint_fast8_t;
using handsize_t = uint_fast8_t;
using score_t = int_fast32_t;

using roundcount_t = uint_fast16_t;
using turncount_t = uint_fast16_t;
using position_t = uint_fast16_t;

constexpr decksize_t DECK_SIZE = 150;
constexpr handsize_t HAND_ROWS = 3;
constexpr handsize_t HAND_COLUMNS = 4;
constexpr handsize_t HAND_SIZE = HAND_ROWS * HAND_COLUMNS;

using Card = int8_t;

struct Deck : public std::array<Card, DECK_SIZE> {
    Deck() : std::array<Card, DECK_SIZE>{{
        -2, -2, -2, -2, -2,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
        10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
        11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
        12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
    }} {}

    void shuffle(rng_t rng) {
        std::shuffle(this->begin(), this->end(), rng);
    }
    
    void validate() const {
        Card counter[14]{};
        for (Card card : *this) {
            counter[card + 2] += 1;
        }
        assert(counter[-2 + 2] == 5);
        assert(counter[-1 + 2] == 10);
        assert(counter[0 + 2] == 15);
        for (size_t i = 1; i < 12; ++i) {
            assert(counter[i + 2] == 10);
        }
    }
};

struct Piles : protected Deck {
protected:
    decksize_t _discard{0};
    decksize_t _draw{1};
    
public:
    Card draw() { return (*this)[this->_draw++]; }

    Card discard() const { return (*this)[this->_discard]; }
    void discard(Card card) { (*this)[++this->_discard] = card; }
    
    void restock(rng_t rng) {
        decksize_t buried = this->_discard;
        if (buried > 0) {
            std::swap(this->front(), (*this)[this->_discard]);
            this->_discard = 0;
            this->_draw -= buried;
            std::copy_n(this->begin() + 1, buried, this->begin() + this->_draw);
        }
    }
    
    void shuffle(rng_t rng) {
        std::shuffle(this->begin(), this->end(), rng);
    }

    bool is_empty() const {
        return this->_draw == DECK_SIZE;
    }

    void validate() const {
        Deck::validate();
        assert(this->_discard < DECK_SIZE);
        assert(this->_draw <= DECK_SIZE);
    }
};

struct Coordinates {
    handsize_t row : 2 {};     // 0-3
    handsize_t column : 2 {};  // 0-2
    
    Coordinates() = default;
    Coordinates(handsize_t index)
        : row(index / HAND_COLUMNS)
        , column(index % HAND_COLUMNS) {}
    Coordinates(handsize_t row, handsize_t column)
        : row(row)
        , column(column) {}
    
    bool operator==(const Coordinates& other) const {
        return this->row == other.row && this->column == other.column;
    }
};

struct Finger {
    Card card;
    enum State {VISIBLE, HIDDEN, CLEARED} state : 2 {HIDDEN};
    handsize_t row : 2 {};
    handsize_t column : 2 {};
    
    bool is_visible() const { return this->state == VISIBLE; }
    bool is_hidden() const { return this->state == HIDDEN; }
    bool is_cleared() const { return this->state == CLEARED; }
    
    Finger get() const {
        return {
            this->state == VISIBLE ? this->card : Card(-16),
            this->state,
            this->row,
            this->column
        };
    }
    
    void flip() {
        switch (this->state) {
            case HIDDEN:
                this->state = VISIBLE;
                break;
            case VISIBLE: throw std::domain_error("tried to flip visible finger!");
            case CLEARED: throw std::domain_error("tried to flip cleared finger!");
        }
    }
    
    Card replace(Card card) {
        switch (this->state) {
            case HIDDEN:
                this->state = VISIBLE;
            case VISIBLE:
                return std::exchange(this->card, card);
            case CLEARED: throw std::domain_error("tried to place cleared finger!");
        }
    }
};

struct Hand : protected std::array<Finger, HAND_SIZE> {
    Finger& at(Coordinates coordinates) {
        return (*this)[coordinates.row * HAND_COLUMNS + coordinates.column];
    }

    Finger at(Coordinates coordinates) const {
        if (coordinates.row >= HAND_ROWS) throw std::domain_error("Invalid hand row");
        if (coordinates.column >= HAND_COLUMNS) throw std::domain_error("Invalid hand column");
        return (*this)[coordinates.row * HAND_COLUMNS + coordinates.column].get();
    }
    
    score_t visible_card_value() const {
        score_t score = 0;
        for (Finger finger : *this) {
            if (finger.state != Finger::CLEARED) {
                score += finger.card;
            }
        }
        return score;
    }

    handsize_t visible_card_count() const {
        handsize_t count = 0;
        for (Finger finger : *this) {
            if (finger.state == Finger::VISIBLE) {
                count += 1;
            }
        }
        return count;
    }

    handsize_t card_count() const {
        handsize_t count = 0;
        for (Finger finger : *this) {
            if (finger.state != Finger::CLEARED) {
                count += 1;
            }
        }
        return count;
    }

    bool are_all_cards_visible() const {
        for (Finger finger : *this) {
            if (finger.state == Finger::HIDDEN) {
                return false;
            }
        }
        return true;
    }

    bool is_column_clearable(handsize_t index) {
        Finger first = this->at({0, index});
        if (first.state != Finger::VISIBLE) { return false; }
        Finger rest = this->at({1, index});
        if (rest.state != Finger::VISIBLE || rest.card != first.card) { return false; }
        rest = this->at({2, index});
        if (rest.state != Finger::VISIBLE || rest.card != first.card) { return false; }
        return true;
    }

    void try_clear_column(handsize_t index) {
        if (this->is_column_clearable(index)) {
            this->at({0, index}).state = Finger::CLEARED;
            this->at({1, index}).state = Finger::CLEARED;
            this->at({2, index}).state = Finger::CLEARED;
        }
    }

    Card replace(Coordinates coordinates, Card card) {
        Card discard = this->at(coordinates).replace(card);
        this->try_clear_column(coordinates.column);
        return discard;
    }

    void deal(Piles& piles) {
        for (handsize_t i = 0; i < HAND_SIZE; ++i) {
            (*this)[i].card = piles.draw();
            (*this)[i].state = Finger::HIDDEN;
        }
    }

    void flip(Coordinates coordinates) {
        this->at(coordinates).flip();
        this->try_clear_column(coordinates.column);
    }

    void flip_all() {
        for (Finger& finger : *this) {
            if (finger.state == Finger::HIDDEN) {
                finger.flip();
            }
        }
    }
};

struct Flip {
private:
    Hand& hand;
    uint8_t count{0};
    Coordinates _first{};
    Coordinates _second{};
    
    Flip(Hand& hand) : hand(hand) {}
    
    void commit() {
        if (this->count != 2) {
            throw std::domain_error("Must flip at two cards!");
        }
        this->hand.at(this->_first).flip();
        this->hand.at(this->_second).flip();
    }
    
    template<typename... Agents>
    friend struct Simulation;
    
public:
    Coordinates first() const { return this->_first; }
    Coordinates second() const { return this->_second; }

    Card flip_card(Coordinates coordinates) {
        switch (this->count) {
            case 0:
                this->_first = coordinates;
                this->count = 1;
                return this->hand.at(coordinates).card;
            case 1:
                if (coordinates == this->_first) throw std::domain_error("Cannot flip already-flipped card!");
                this->_second = coordinates;
                this->count = 2;
                return this->hand.at(coordinates).card;
            default:
                throw std::domain_error("Cannot flip more than two cards!");
        }
    }
};

struct Turn {
private:
    Hand& hand;
    Piles& piles;
    enum Step : uint8_t { START, DRAW, DONE } step{START};
    Card _card{};
    Coordinates _coordinates{};
    enum Action : uint8_t { UNSET, DISCARD_AND_FLIP, PLACE_DRAW, PLACE_DISCARD } _action{UNSET};
    
    Turn(Hand& hand, Piles& piles) : hand(hand), piles(piles) {}
    
    void commit() {
        switch (this->_action) {
            case DISCARD_AND_FLIP:
                this->piles.discard(this->_card);
                this->hand.flip(this->_coordinates);
                return;
            case PLACE_DRAW:
            case PLACE_DISCARD:
                this->piles.discard(this->hand.replace(this->_coordinates, this->_card));
                return;
            case UNSET:
                throw std::domain_error("Must either draw or pick up the discard!");
        }
    }
    
    template<typename... Agents>
    friend struct Simulation;
    
public:
    const Coordinates coordinates() const { return this->_coordinates; }
    const Action action() const { return this->_action; }
    const Card card() const { return this->_card; }

    Card draw_card() {
        switch (this->step) {
            case START:
                this->step = DRAW;
                this->_card = this->piles.draw();
                return this->_card;
            case DRAW: throw std::domain_error("Cannot draw more than once!");
            case DONE: throw std::domain_error("Cannot draw after turn is complete!");
        }
    }
    
    void discard_and_flip(Coordinates coordinates) {
        switch (this->step) {
            case START: throw std::domain_error("Must draw before discarding and flipping!");
            case DRAW:
                this->step = DONE;
                this->_action = DISCARD_AND_FLIP;
                this->_coordinates = coordinates;
                return;
            case DONE: throw std::domain_error("Cannot discard and flip after turn is complete!");
        }
    }
    
    void place_draw(Coordinates coordinates) {
        switch (this->step) {
            case START: throw std::domain_error("Must draw a card before placing!");
            case DRAW:
                this->step = DONE;
                this->_action = PLACE_DRAW;
                this->_coordinates = coordinates;
                return;
            case DONE: throw std::domain_error("Cannot place card after turn is complete!");
        }
    }
    
    void place_discard(Coordinates coordinates) {
        switch (this->step) {
            case START:
                this->_card = this->piles.draw();
                this->step = DONE;
                this->_action = PLACE_DISCARD;
                this->_coordinates = coordinates;
                return;
            case DRAW: throw std::domain_error("Cannot place from discard after drawing!");
            case DONE: throw std::domain_error("Cannot place card after turn is complete!");
        }
    }
};

struct Player {
    Hand hand;
    score_t score;
    position_t index;
};

struct State {
    uint32_t round_index;
    uint32_t turn_index;
    position_t turn_offset;
    position_t turn_starter_index;
    position_t round_ender_index;
    bool is_round_ending;
    Piles piles;
    std::span<Player> players;

    explicit State(std::span<Player> players) : players(players) {}
    
    Player& player() { return this->players[this->player_index()]; }
    const Player& player() const { return this->players[this->player_index()]; }
    position_t player_index() const { return (this->turn_starter_index + turn_offset) % this->players.size(); }
    const Hand& hand() const { return this->player().hand; }
    score_t score() const { return this->player().score; }

    const Player& next_player() const { return this->players[this->next_player_index()]; }
    position_t next_player_index() const { return (this->player_index() + 1) % this->players.size(); }
    const Player& last_player() const { return this->players[this->last_player_index()]; }
    position_t last_player_index() const { return (this->player_index() + this->players.size() - 1) % this->players.size(); }

    const Player& largest_visible_hand_player() const { return this->players[this->largest_visible_hand_value_player_index()]; }
    position_t largest_visible_hand_value_player_index() const {
        return imax(this->players.begin(), this->players.end(), [](const Player& player) { return player.hand.visible_card_value(); });
    }

    score_t lowest_score() const { return this->lowest_score_player().score; }
    const Player& lowest_score_player() const { return this->players[this->lowest_score_player_index()]; }
    position_t lowest_score_player_index() const { 
        return imax(this->players.begin(), this->players.end(), [](const Player& player) { return player.score; });
    }
};


struct Agent {
    virtual ~Agent() {}
    
    /// The player who ended the last game round gets to start the new game round.
    
    /// Rule: each player reveals two of their playing cards at will. We will
    /// provide no visibility of other players' flipped cards here to simulate
    /// simultaneous flipping.
    virtual void flip(const State& state, Flip& action) = 0;
    virtual void turn(const State& state, Turn& action) = 0;

    // virtual void on_game_start(const State& state) {}
    // virtual void on_round_start(const State& state) {}
    // virtual void on_other_turn(const State& state, const Turn& action) {}
    // virtual void on_round_end(const State& state, std::span<score_t> scores) {}
    // virtual void on_game_end(const State& state, std::span<score_t> scores) {}
};

template<typename... Agents>
struct Simulation {
    static constexpr position_t size = sizeof...(Agents);
    static_assert(3 <= size && size <= 8);
        
    rng_t rng;
    vtuple<Agent, Agents...> agents;
    
    Simulation(Agents&&... agents)
        : rng{std::random_device{}()}
        , agents{std::forward<Agents>(agents)...} {}
    Simulation(rng_t rng, Agents&&... agents)
        : rng{rng}
        , agents{std::forward<Agents>(agents)...} {}

    std::array<score_t, size> play() {
        std::array<Player, size> players{};
        for (position_t i = 0; i < size; ++i) { players[i].index = i; }

        State state(players);
        state.piles.shuffle(this->rng);
        for (position_t i = 0; i < size; ++i) {
            players[i].hand.deal(state.piles);
        }

        // All players flip their first two cards in isolation.
        {
            std::array<Flip, size> flips = starmap([](Player& player) { return Flip(player.hand); }, players);
            for (position_t i = 0; i < size; ++i) { this->agents[i]->flip(state, flips[i]); }
            for (position_t i = 0; i < size; ++i) { flips[i].commit(); }
        }

        // Determine the first in the turn order.
        state.turn_starter_index = state.largest_visible_hand_value_player_index();

    turns:
        for (position_t offset = 0; offset < size; ++offset) {
            if (state.piles.is_empty()) {
                state.piles.restock(this->rng);
            }

            state.turn_offset = offset;
            position_t player_index = state.player_index();
            Hand& hand = players[player_index].hand;
            Turn turn(hand, state.piles);
            this->agents[player_index]->turn(state, turn);
            turn.commit();

            if (hand.are_all_cards_visible()) {
                state.round_ender_index = player_index;
                state.is_round_ending = true;
                goto endround;
            }
        }

        state.turn_index += 1;
        goto turns;

    endround:
        for (position_t offset = 0; offset < size - 1; ++offset) {
            state.turn_offset += 1;
            position_t player_index = state.player_index();
            Hand& hand = players[player_index].hand;
            Turn turn(hand, state.piles);
            this->agents[player_index]->turn(state, turn);
            turn.commit();
        }

        state.is_round_ending = false;

        for (position_t i = 0; i < size; ++i) {
            players[i].hand.flip_all();
            players[i].score += players[i].hand.visible_card_value();
            if (players[i].score >= 100) {
                goto done;
            }
        }

        state.round_index += 1;
        state.turn_starter_index = imin(players.begin(), players.end(), [](Player& player) { return player.hand.visible_card_value(); });
        state.piles.shuffle(this->rng);
        for (position_t i = 0; i < size; ++i) {
            players[i].hand.deal(state.piles);
        }

        goto turns;

    done:
        // Retrieve and return scores.
        std::array<score_t, size> scores;
        for (position_t i = 0; i < size; ++i) {
            scores[i] = players[i].score;
        }
        return scores;
    }
};

};
