#include "../include/population.h"
#include "../include/patterns.h"
#include "../include/random_utils.h"


//applyVector is need for applying vector to block
void Individual::applyVector(Block& block, Mask coords) {
    cv::parallel_for_(
        cv::Range(0, static_cast<int>(coords.size())),
        [&](const cv::Range& range) {
            for (int idx = range.start; idx < range.end; ++idx) {
                const auto& [i, j] = coords[idx];
                block.frequencyDomain(i, j) += vector(i, j);
            }
        }
    );
}

//calcObjectiveFunction return valure of objective function for current solution
void Individual::calcObjectiveFunction(Block& block, uchar bit, Mask coords, Mask rS0, Mask rS1){
    applyVector(block, coords);
    cv::idct(block.frequencyDomain, block.spatialDomain);
    double psnr = block.psnr();
    double s0 = block.calcAbsSum(rS0);
    double s1 = block.calcAbsSum(rS1);
    objectiveFunctionValue =  (bit == 0? (s1/s0) : (s0/s1)) * psnr * 0.01;
    block.frequencyDomain = block.frequencyDomain_.clone(); //reset to initial value
}


//Population is need for inititialisation population
Population::Population(Block b, Mask rS0, Mask rS1, Mask wC, uchar bit)
: block(b), regionOfS0(rS0),regionOfS1(rS1), wholeСoefficient(wC), bit(bit) {
    Individual ws = Individual{.objectiveFunctionValue = 100.0};
    size_t iobs = 0;
    for (int i = 0; i < POPULATION_SIZE; ++i){
        cv::parallel_for_(cv::Range(0, wC.size()), [&](const cv::Range& range){
            for (int j = range.start; j < range.end; ++i){
                const auto& [x, y] = wC[j];
                individuals[i].vector(x,y) = randomDoubleForPopulation();
            }
        });
        individuals[i].calcObjectiveFunction(b, bit, wC, rS0, rS1);

        if (individuals[i].objectiveFunctionValue < individuals[iobs].objectiveFunctionValue) {
            iobs = i;
        }

        if (individuals[i].objectiveFunctionValue > ws.objectiveFunctionValue) {
            ws = individuals[i];
        }
    }

    worstSolution = ws;
    indexOfBestSolution = iobs;
}

//update is need for updating vectors in population
void Population::update(Individual trial, size_t indexOfUpdatingIndividual) {
    if (trial.objectiveFunctionValue < individuals[indexOfUpdatingIndividual].objectiveFunctionValue) {
        individuals[indexOfUpdatingIndividual] = trial;
        if (individuals[indexOfUpdatingIndividual].objectiveFunctionValue < individuals[indexOfBestSolution].objectiveFunctionValue) {
            indexOfBestSolution = indexOfUpdatingIndividual;
        }
    } else if (individuals[indexOfUpdatingIndividual].objectiveFunctionValue > worstSolution.objectiveFunctionValue) {
        worstSolution = individuals[indexOfUpdatingIndividual];
    }
}