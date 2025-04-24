#ifndef RANDOM_UTILS_H
#define RANDOM_UTILS_H

#include <random>
#include <ctime>
#include <cmath>
#include <climits>
#include <cstring>
#include "config.h"
#include <array>

// Initializes the random number generator
void init_random();

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
int rand_int_1_to_11();

#endif // RANDOM_UTILS_H