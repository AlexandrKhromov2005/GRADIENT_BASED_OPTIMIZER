#include "../include/population.h"

//GSR is needed to calculate gsr value
populationVector Population::GSR(GSR_Kit gsrKit) {
    // calculate delta
    populationVector sum = individuals[gsrKit.indexes[0]].vector;
    sum = sum + std::make_pair(individuals[gsrKit.indexes[1]].vector, wholeСoefficient);
    sum = sum + std::make_pair(individuals[gsrKit.indexes[2]].vector, wholeСoefficient);
    sum = sum + std::make_pair(individuals[gsrKit.indexes[3]].vector, wholeСoefficient);
    
    populationVector avg = sum * std::make_pair(0.25, wholeСoefficient);
    populationVector diff = avg - std::make_pair(individuals[gsrKit.iocv].vector, wholeСoefficient);
    gsrKit.delta = 2.0 * randomDoubleZeroToOne() * absPopulationVectorFast(diff);

    // calculate step
    populationVector bestDiff = individuals[indexOfBestSolution].vector - 
                              std::make_pair(individuals[gsrKit.indexes[0]].vector, wholeСoefficient);
    gsrKit.step = bestDiff * std::make_pair(0.5, wholeСoefficient) + std::make_pair(gsrKit.delta, wholeСoefficient);

    // calculate delx
    gsrKit.delx = generateFastPopulationIndex() * absPopulationVectorFast(gsrKit.step);

    // calculate GSR
    populationVector numerator = individuals[gsrKit.iocv].vector * 
                               std::make_pair(generateNormalZeroToOne() * gsrKit.rho1 * 2.0 * gsrKit.delx, wholeСoefficient);
    
    populationVector denominator = worstSolution.vector - 
                                 std::make_pair(individuals[indexOfBestSolution].vector, wholeСoefficient) + 
                                 std::make_pair(gsrKit.eps, wholeСoefficient);
    
    gsrKit.gsr = safeDivideMasked(numerator, denominator, wholeСoefficient);

    return gsrKit.gsr;
}

//GBO is needed in order to find the best solution
void Population::GBO(){
    double alpha, betta, rho1, rho2;
    populationVector x1, x2, x3, gsr;
    GSR_Kit gsrKit;
    for (size_t m = 0; m < GBO_ITERATIONS; ++m){
        betta = BETTA_MIN + (BETTA_MAX - BETTA_MIN)* pow(1.0 - pow(static_cast<double>(m + 1) / static_cast<double>(GBO_ITERATIONS), 3.0), 2.0);
        alpha = fabs(betta * sin(ANGLE + sin(ANGLE * betta)));
        rho1, rho2 = alpha;
        
        for (size_t indexOfCurrentVector = 0; indexOfCurrentVector < POPULATION_SIZE; ++indexOfCurrentVector){
            rho1 *= (2.0 * randomDoubleZeroToOne() - 1.0);
            rho2 *= (2.0 * randomDoubleZeroToOne() - 1.0);
            gsrKit.rho1 = rho1;
            gsrKit.rho2 = rho2;
            gsrKit.iocv = indexOfCurrentVector;
            gsrKit.xs = individuals[indexOfCurrentVector].vector;
            gsrKit.indexes = generateFourUniqueIndices(indexOfCurrentVector, indexOfBestSolution);
            gsrKit.eps = randomDoubleZeroToOne() * 0.01;
        }
    }
}