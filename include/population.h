#ifndef POPULATION_H
#define POPULATION_H

#include <opencv2/opencv.hpp>
#include <array>
#include <vector>
#include "config.h"
#include "types.h"
#include "smart_block.h"
#include "random_utils.h"

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
    GBOVector GSR(GSR_Kit gsrKit);
};

struct GSR_Kit {
    GBOVector           delx;
    GBOVector           delta;
    GBOVector           step;
    GBOVector           dm;
    GBOVector           xs;
    GBOVector           gsr;
    GBOVector           xr1;      
    GBOVector           xr2;      
    GBOVector           xr3;      
    GBOVector           xr4;      
    GBOVector           x_best;   
    GBOVector           x_worst;  
    GBOVector           x_current;
    GBOVector           yp;       
    GBOVector           yq;       

    double              eps;
    double              rho1;
    double              rho2;

    std::vector<size_t> indexes;
    size_t              iocv;
};

#endif //POPULATION_H
