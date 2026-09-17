#include "random_utils.h"
#include <algorithm>

// All generator state. One instance per thread, so blocks can be embedded in parallel.
struct RandomState {
    std::mt19937 generator{std::random_device{}()};  // uniform numbers, indices
    std::mt19937 gen{std::random_device{}()};        // normal numbers
    bool initialized = false;
    bool normal_saved_available = false;
    double normal_saved = 0.0;
};
static thread_local RandomState rng;

// Initializes the random number generator
void init_random() {
    if (!rng.initialized) {
        rng.generator.seed(static_cast<unsigned int>(std::time(nullptr)));
        rng.initialized = true;
    }
}

// Seeds every generator of the calling thread with a fixed value (reproducible runs)
void seed_random(unsigned int seed) {
    rng.generator.seed(seed);
    rng.gen.seed(seed ^ 0x9E3779B9u);
    rng.normal_saved_available = false;
    rng.initialized = true;
}

// 53-bit uniform number in [0, 1) from two 32-bit draws. Same value, draw for draw, as
// std::uniform_real_distribution<double>(0, 1) over std::mt19937 in libstdc++, but without
// the long double arithmetic and the run-time log() of std::generate_canonical.
static inline double canonical(std::mt19937& engine) {
    const double low = static_cast<double>(engine());
    const double high = static_cast<double>(engine());
    const double value = (low + high * 4294967296.0) / 18446744073709551616.0;
    return (value >= 1.0) ? std::nextafter(1.0, 0.0) : value;
}

// Generates a random number in the range [0.0, 1.0]
double rand_num() {
    return canonical(rng.generator);
}

// Standard normal number, Marsaglia polar method (the algorithm of std::normal_distribution
// in libstdc++, reproduced draw for draw): every second call returns the saved twin value.
static double standard_normal() {
    if (rng.normal_saved_available) {
        rng.normal_saved_available = false;
        return rng.normal_saved;
    }
    double x, y, r2;
    do {
        x = 2.0 * canonical(rng.gen) - 1.0;
        y = 2.0 * canonical(rng.gen) - 1.0;
        r2 = x * x + y * y;
    } while (r2 > 1.0 || r2 == 0.0);
    const double mult = std::sqrt(-2 * std::log(r2) / r2);
    rng.normal_saved = x * mult;
    rng.normal_saved_available = true;
    return y * mult;
}

// Generates a normally distributed number clamped to [0, 1]
double randn() {
    init_random();
    double val = std::max(0.0, std::min(1.0, standard_normal()));
    return val;
}

// Generates a value of rho based on the parameter alpha
double new_rho(double alpha) {
    return (2.0 * rand_num() * alpha) - alpha;
}

// Generates four unique indices
void gen_indexes(std::array<size_t, 4>& indexes, size_t cur_ind, size_t best_ind) {
    int cnt = 0;
    std::array<bool, POP_SIZE> used_indices = { false };
    used_indices[cur_ind] = true;
    used_indices[best_ind] = true;

    while (cnt < 4) {
        size_t temp = gen_random_index();
        if (!used_indices[temp]) {
            indexes[cnt] = temp;
            cnt++;
            used_indices[temp] = true;
        }
    }
}

// Generates a random index without checking
size_t gen_random_index() {
    return rng.generator() % POP_SIZE;  // Generates a random index
}

// Generates a random number in the range from -1 to 1 inclusive
double rand_neg_one_to_one() {
    return 2.0 * rand_num() - 1.0;  // Generates a number from -1 to 1
}

// Generates a random value of 0 or 1 of type unsigned char
unsigned char rand_binary() {
    // Uses rand_num() to generate a random number in the range [0.0, 1.0]
    double random_value = rand_num();

    // If random_value < 0.5, returns 0, otherwise returns 1
    return (random_value < 0.5) ? 0 : 1;
}

// Generates a random integer in the range [1, 100] inclusive
int rand_int_1_to_100() {
    std::uniform_int_distribution<int> int_dist(1, 100);  // Distribution for integers from 1 to 100
    return int_dist(rng.generator);  // Generates a random integer in the specified range
}