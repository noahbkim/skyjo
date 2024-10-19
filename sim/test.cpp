#include <cassert>
#include <chrono>
#include <random>

#include "skyjo.hpp"

using namespace skyjo;

struct BasicAgent : public Agent {
    virtual void flip(const State& state, Flip& action) override {
        action.flip_card({0, 0});
        action.flip_card({0, 1});
    }

    virtual void turn(const State& state, Turn& action) override {
        action.draw_card();
        for (handsize_t i = 0; i < HAND_SIZE; ++i) {
            if (state.hand().at(i).is_hidden()) {
                action.place_draw(Coordinates(i));
                return;
            }
        }
    }
};

void test_deck_construction() {
    Deck deck;
    assert(deck.front() == -2);
    assert(deck.back() == 12);
    deck.validate();
}

void test_deck_shuffle() {
    rng_t rng(0);
    Deck deck;
    deck.shuffle(rng);
    deck.validate();
}

void test_piles_draw_all() {
    rng_t rng(0);
    Piles piles;
    piles.shuffle(rng);
    for (size_t i = 0; i < DECK_SIZE; ++i) {
        piles.discard(piles.draw());
    }
    piles.validate();
}

void test_game_3() {
    rng_t rng(0);
    Simulation simulation(rng, BasicAgent(), BasicAgent(), BasicAgent());
    simulation.play();
    assert(simulation.scores[0] == 140);
    assert(simulation.scores[1] == 59);
    assert(simulation.scores[2] == 61);
}

void test_game_4() {
    rng_t rng(0);
    Simulation simulation(rng, BasicAgent(), BasicAgent(), BasicAgent(), BasicAgent());
    simulation.play();
    assert(simulation.scores[0] == 104);
    assert(simulation.scores[1] == 85);
    assert(simulation.scores[2] == 89);
    assert(simulation.scores[3] == 81);
}

int main() {
    test_deck_construction();
    test_deck_shuffle();
    test_game_3();
    test_game_4();
}
