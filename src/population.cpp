#include "../include/population.h"
#include "../include/patterns.h"
#include "../include/random_utils.h"


//applyVector is need for applying vector to block
void Individual::applyVector(Block& block, Mask coords) {
    cv::Mat mask = cv::Mat::zeros(8, 8, CV_8U);
    
    cv::parallel_for_(cv::Range(0, coords.size()), [&](const cv::Range& range) {
        for (int i = range.start; i < range.end; ++i) {
            const auto& [y, x] = coords[i];
            if (y >= 0 && y < mask.rows && x >= 0 && x < mask.cols) {
                mask.at<uchar>(y, x) = 1;
            }
        }
    });

    cv::add(block.frequencyDomain, vector, block.frequencyDomain, mask);
}

//calcObjectiveFunction return valure of objective function for current solution
void Individual::calcObjectiveFunction(Block& block, uchar bit, Mask coords, Mask rS0, Mask rS1){
    applyVector(block, coords);
    cv::idct(block.frequencyDomain, block.spatialDomain);
    double psnr = block.psnr();
    double s0 = block.calcAbsSum(rS0);
    double s1 = block.calcAbsSum(rS1);
    objectiveFunctionValue =  (bit == 0? (s1/s0) : (s0/s1)) * psnr * 0.01;
}


//Population is need for inititialisation population
Population::Population(Block b, Mask rS0, Mask rS1, Mask wC, uchar bit)
: block(b), regionOfS0(rS0),regionOfS1(rS1), wholeСoefficient(wC), bit(bit) {
    Individual ws = Individual{.objectiveFunctionValue = 10.0};
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