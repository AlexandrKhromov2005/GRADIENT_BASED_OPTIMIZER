#ifndef RANDOM_UTILS_H
#define RANDOM_UTILS_H

#include <random>
#include <ctime>
#include "config.h"

double randomDoubleForPopulation();
double randomDoubleZeroToOne();
std::vector<size_t> generateFourUniqueIndices(size_t excludeIndex1, size_t excludeIndex2);
double generateNormalZeroToOne();
int generateFastPopulationIndex();
double randomBinaryFast();
double randomOneFast();

#endif //RANDOM_UTILS
