#ifndef RANDOM_UTILS_H
#define RANDOM_UTILS_H

#include <random>
#include <ctime>
#include <cmath>
#include <climits>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include "config.h"
#include <array>

// All generator state. One instance per thread (see threadRandomState), so blocks can be
// embedded in parallel. Hot loops fetch the state once and call the inline members directly.
struct RandomState {
    std::mt19937 generator{std::random_device{}()};  // uniform numbers, indices
    std::mt19937 gen{std::random_device{}()};        // normal numbers
    bool initialized = false;
    bool normal_saved_available = false;
    double normal_saved = 0.0;

    void initOnce() {
        if (!initialized) {
            generator.seed(static_cast<unsigned int>(std::time(nullptr)));
            initialized = true;
        }
    }

    // 53-bit uniform number in [0, 1) from two 32-bit draws. Same value, draw for draw, as
    // std::uniform_real_distribution<double>(0, 1) over std::mt19937 in libstdc++, but without
    // the long double arithmetic and the run-time log() of std::generate_canonical.
    static double canonical(std::mt19937& engine) {
        const double low = static_cast<double>(engine());
        const double high = static_cast<double>(engine());
        const double value = (low + high * 4294967296.0) / 18446744073709551616.0;
        return (value >= 1.0) ? std::nextafter(1.0, 0.0) : value;
    }

    // Uniform random number in [0, 1)
    double uniform() { return canonical(generator); }

    // Uniform random number in [-1, 1)
    double uniformSigned() { return 2.0 * uniform() - 1.0; }

    // Standard normal number, Marsaglia polar method (the algorithm of std::normal_distribution
    // in libstdc++, reproduced draw for draw): every second call returns the saved twin value.
    double standardNormal() {
        if (normal_saved_available) {
            normal_saved_available = false;
            return normal_saved;
        }
        double x, y, r2;
        do {
            x = 2.0 * canonical(gen) - 1.0;
            y = 2.0 * canonical(gen) - 1.0;
            r2 = x * x + y * y;
        } while (r2 > 1.0 || r2 == 0.0);
        const double mult = std::sqrt(-2 * std::log(r2) / r2);
        normal_saved = x * mult;
        normal_saved_available = true;
        return y * mult;
    }

    // Normally distributed number clamped to [0, 1]
    double normalClamped() {
        initOnce();
        return std::max(0.0, std::min(1.0, standardNormal()));
    }

    // Random population index
    size_t index() { return generator() % POP_SIZE; }

    // Four distinct population indices, different from cur_ind and best_ind
    void indexes(std::array<size_t, 4>& out, size_t cur_ind, size_t best_ind) {
        int cnt = 0;
        std::array<bool, POP_SIZE> used_indices = { false };
        used_indices[cur_ind] = true;
        used_indices[best_ind] = true;

        while (cnt < 4) {
            size_t temp = index();
            if (!used_indices[temp]) {
                out[cnt] = temp;
                cnt++;
                used_indices[temp] = true;
            }
        }
    }
};

// Generator state of the calling thread
RandomState& threadRandomState();

// Initializes the random number generator
void init_random();

// Seeds every generator of the calling thread with a fixed value (reproducible runs)
void seed_random(unsigned int seed);

// Seeds the calling thread for one unit of work (an 8x8 block): the stream depends only on
// (base_seed, index), not on which thread runs it or in what order.
void seed_random_stream(uint64_t base_seed, uint64_t index);

// Generates a random number in the range [0.0, 1.0]
double rand_num();

// Generates a normally distributed number
double randn();

// Generates a value of rho based on the parameter alpha
double new_rho(double alpha);

// Generates four unique indices
void gen_indexes(std::array<size_t, 4>& indexes, size_t cur_ind, size_t best_ind);

// Generates a random index without checking
size_t gen_random_index();

// Generates a random number in the range from -1 to 1 inclusive
double rand_neg_one_to_one();

// Generates a random value of 0 or 1 of type unsigned char
unsigned char rand_binary();

// Generates a random integer in the range [1, 100] inclusive
int rand_int_1_to_100();

#endif // RANDOM_UTILS_H
