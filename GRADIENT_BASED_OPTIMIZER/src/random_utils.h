#ifndef RANDOM_UTILS_H
#define RANDOM_UTILS_H

#include <random>
#include <ctime>
#include <cmath>
#include <climits>
#include <cstring>
#include "config.h"
#include <array>

// Инициализация генератора случайных чисел
void init_random();

// Генерация случайного числа в диапазоне [0.0, 1.0]
double rand_num();

// Генерация нормально распределенного числа
double randn();

// Генерация значения rho на основе параметра alpha
double new_rho(double alpha);

// Генерация четырёх уникальных индексов
void gen_indexes(std::array<size_t, 4>& indexes, size_t cur_ind, size_t best_ind);

// Генерация одного случайного индекса без проверки
size_t gen_random_index();

double rand_neg_one_to_one();

unsigned char rand_binary();

#endif // RANDOM_UTILS_H