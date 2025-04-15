#include "random_utils.h"

std::random_device rd1;
static std::mt19937 generator(rd1());
static std::uniform_real_distribution<double> distribution(0.0, 1.0);
static bool initialized = false;
std::normal_distribution<double> dist(0.0, 1.0);
std::random_device rd;  // Source of entropy
std::mt19937 gen(rd());

// Initializes the random number generator
void init_random() {
    if (!initialized) {
        generator.seed(static_cast<unsigned int>(std::time(nullptr)));
        initialized = true;
    }
}

// Generates a random number in the range [0.0, 1.0]
double rand_num() {
    return distribution(generator);
}

// Generates a normally distributed number (Box-Muller method)
double randn() {
    init_random();
    double val = std::clamp(dist(gen), 0.0, 1.0);
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
    return generator() % POP_SIZE;  // Generates a random index
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
int rand_int_1_to_10() {
    std::uniform_int_distribution<int> int_dist(1, 10);  // Distribution for integers from 1 to 100
    return int_dist(generator);  // Generates a random integer in the specified range
}