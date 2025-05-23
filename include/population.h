#ifndef POPULATION_H
#define POPULATION_H

#include <opencv2/opencv.hpp>
#include <array>
#include "config.h"
#include "types.h"
#include "smart_block.h"

struct Individual {
    populationVector    vector;
    double              objectiveFunctionValue;
    
    void                applyVector(Block& block, Mask coords);
    void                calcObjectiveFunction(Block& block, uchar bit, Mask coords, Mask rS0, Mask rS1);
};

struct Population {
    Block                                   block;
    std::array<Individual, POPULATION_SIZE> individuals;
    Individual                              worstSolution;
    size_t                                  indexOfBestSolution;  
    Mask                                    regionOfS0;
    Mask                                    regionOfS1;
    Mask                                    wholeСoefficient;
    uchar                                   bit;

    Population(Block b, Mask rS0, Mask rS1, Mask wC, uchar bit);
    void update(Individual trial, size_t indexOfUpdatingIndividual);
    void GBO();
};

#endif //POPULATION_H
