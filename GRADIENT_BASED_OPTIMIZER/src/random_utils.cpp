#include "random_utils.h"

RandomState& threadRandomState() {
    static thread_local RandomState state;
    return state;
}

// Initializes the random number generator
void init_random() {
    threadRandomState().initOnce();
}

// Seeds every generator of the calling thread with a fixed value (reproducible runs)
void seed_random(unsigned int seed) {
    RandomState& rng = threadRandomState();
    rng.generator.seed(seed);
    rng.gen.seed(seed ^ 0x9E3779B9u);
    rng.normal_saved_available = false;
    rng.initialized = true;
}

static inline uint64_t splitmix64(uint64_t& x) {
    uint64_t z = (x += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

void seed_random_stream(uint64_t base_seed, uint64_t index) {
    RandomState& rng = threadRandomState();
    uint64_t x = base_seed ^ (index * 0xD1342543DE82EF95ull);
    const uint64_t a = splitmix64(x), b = splitmix64(x);
    std::seed_seq seq_uniform{static_cast<uint32_t>(a), static_cast<uint32_t>(a >> 32)};
    std::seed_seq seq_normal{static_cast<uint32_t>(b), static_cast<uint32_t>(b >> 32)};
    rng.generator.seed(seq_uniform);
    rng.gen.seed(seq_normal);
    rng.normal_saved_available = false;
    rng.initialized = true;
}

// Generates a random number in the range [0.0, 1.0]
double rand_num() {
    return threadRandomState().uniform();
}

// Generates a normally distributed number clamped to [0, 1]
double randn() {
    return threadRandomState().normalClamped();
}

// Generates a value of rho based on the parameter alpha
double new_rho(double alpha) {
    return (2.0 * rand_num() * alpha) - alpha;
}

// Generates four unique indices
void gen_indexes(std::array<size_t, 4>& indexes, size_t cur_ind, size_t best_ind) {
    threadRandomState().indexes(indexes, cur_ind, best_ind);
}

// Generates a random index without checking
size_t gen_random_index() {
    return threadRandomState().index();
}

// Generates a random number in the range from -1 to 1 inclusive
double rand_neg_one_to_one() {
    return threadRandomState().uniformSigned();
}

// Generates a random value of 0 or 1 of type unsigned char
unsigned char rand_binary() {
    // If random_value < 0.5, returns 0, otherwise returns 1
    return (rand_num() < 0.5) ? 0 : 1;
}

// Generates a random integer in the range [1, 100] inclusive
int rand_int_1_to_100() {
    std::uniform_int_distribution<int> int_dist(1, 100);
    return int_dist(threadRandomState().generator);
}
