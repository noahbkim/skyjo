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
    std::array<score_t, 3> scores = simulation.play();
    assert(scores[0] == 120);
    assert(scores[1] == 59);
    assert(scores[2] == 61);
}

void test_game_4() {
    rng_t rng(0);
    Simulation simulation(rng, BasicAgent(), BasicAgent(), BasicAgent(), BasicAgent());
    std::array<score_t, 4> scores = simulation.play();
    assert(scores[0] == 98);
    assert(scores[1] == 127);
    assert(scores[2] == 89);
    assert(scores[3] == 81);
}

void test_game_stress(size_t count) {
    rng_t rng(0);
    Simulation simulation(rng, BasicAgent(), BasicAgent(), BasicAgent(), BasicAgent());
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < count; ++i) {
        simulation.play();
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Ran " << count << " games in " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << "ms" << std::endl;
}

int main() {
    test_deck_construction();
    test_deck_shuffle();
    test_game_3();
    test_game_4();
    test_game_stress(10'000);
}
