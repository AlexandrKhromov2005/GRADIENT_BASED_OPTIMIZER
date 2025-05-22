#include "../include/random_utils.h"

//randomDoubleForPopulation is need for generating random vector coefficients 
//from -TH to TH
double randomDoubleForPopulation() {
    static std::mt19937 generator(std::time(nullptr));
    std::uniform_real_distribution<double> distribution(-TH, TH);    
    return distribution(generator);
}