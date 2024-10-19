#include <iostream>

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
                action.place_draw(i);
                return;
            }
        }
    }
};

int main(int argc, char** argv) {
    size_t count = 10'000;
    rng_t rng(std::random_device{}());

    Simulation simulation(rng, BasicAgent(), BasicAgent(), BasicAgent(), BasicAgent());
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < count; ++i) {
        simulation.play();
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Ran " << count << " games in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    std::cout << "Scores: ";
    for (position_t i = 0; i < 4; ++ i) {
        std::cout << simulation.scores[i] / static_cast<double>(count) << ", ";
    }
    std::cout << std::endl;

    std::cout << "Wins: ";
    for (position_t i = 0; i < 4; ++ i) {
        std::cout << simulation.wins[i] << ", ";
    }
    std::cout << std::endl;
}
