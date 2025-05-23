#include "../include/random_utils.h"
#include <algorithm>
#include <chrono>
//randomDoubleForPopulation is need for generating random vector coefficients 
//from [-TH; TH]
double randomDoubleForPopulation() {
    static std::mt19937 generator(std::time(nullptr));
    std::uniform_real_distribution<double> distribution(-TH, TH);    
    return distribution(generator);
}

//randomDoubleZeroToOne is need for generating random number from [0.0; 1.0]
double randomDoubleZeroToOne() {
    static std::mt19937 generator(std::time(nullptr)); 
    static std::uniform_real_distribution<double> distribution(0.0, 1.0);
    return distribution(generator);
}


//generateFourUniqueIndices is need for generating 4 random unique indexes
std::vector<size_t> generateFourUniqueIndices(
    size_t excludeIndex1,  
    size_t excludeIndex2       
) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dist(0, POPULATION_SIZE - 1);

    std::vector<size_t> indices;

    while (indices.size() < 4) {
        size_t candidate = dist(gen);

        if (candidate != excludeIndex1 && 
            candidate != excludeIndex2 && 
            std::find(indices.begin(), indices.end(), candidate) == indices.end()) {
            indices.push_back(candidate);
        }
    }

    return indices;
}

//generateNormalZeroToOne is needs for generating normally distributed numbers in [0.0; 1.0]
double generateNormalZeroToOne() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    
    static std::normal_distribution<double> dist(0.5, 0.2);

    double value;
    do {
        value = dist(gen);
    } while (value < 0.0 || value > 1.0);  

    return value;
}

//generateFastPopulationIndex == rand(1:N)
int generateFastPopulationIndex() {
    static thread_local std::minstd_rand generator(
        std::chrono::system_clock::now().time_since_epoch().count()
    );
    return 1 + generator() % POPULATION_SIZE;
}
