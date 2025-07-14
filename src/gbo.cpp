#include "../include/population.h"

//GSR is needed to calculate gsr value
GBOVector Population::GSR(GSR_Kit gsrKit) {
    copyMaskedToVectorFast(individuals[gsrKit.indexes[2]].vector, wholeСoefficient, gsrKit.xr3);
    copyMaskedToVectorFast(individuals[gsrKit.indexes[3]].vector, wholeСoefficient, gsrKit.xr4);
    copyMaskedToVectorFast(worstSolution.vector, wholeСoefficient, gsrKit.x_worst);
    copyMaskedToVectorFast(individuals[gsrKit.iocv].vector, wholeСoefficient, gsrKit.x_current);

    gsrKit.delta = 2.0 * randomDoubleZeroToOne() * abs(0.25 * (gsrKit.xr1 + gsrKit.xr2 + gsrKit.xr3 + gsrKit.xr4) - gsrKit.x_current);
    gsrKit.step = 0.5 * (gsrKit.x_best - gsrKit.xr1 + gsrKit.delta);
    gsrKit.delx = static_cast<double>(generateFastPopulationIndex()) * abs(gsrKit.step);
    gsrKit.gsr = (generateNormalZeroToOne() * gsrKit.rho1 * 2.0 * gsrKit.x_current) / (gsrKit.x_worst - gsrKit.x_best + gsrKit.eps);
    gsrKit.xs = gsrKit.xs - gsrKit.gsr + gsrKit.dm;
    gsrKit.yp = randomDoubleZeroToOne() * (0.5 * (gsrKit.xs + gsrKit.x_current) + randomDoubleZeroToOne() * gsrKit.delx);
    gsrKit.yq = randomDoubleZeroToOne() * (0.5 * (gsrKit.xs + gsrKit.x_current) - randomDoubleZeroToOne() * gsrKit.delx);
    gsrKit.gsr = (generateNormalZeroToOne() * gsrKit.rho2 * 2.0 * gsrKit.delx * gsrKit.x_current) / (gsrKit.yp - gsrKit.yq + gsrKit.eps);

    return gsrKit.gsr;
}
//GBO is needed in order to find the best solution
void Population::GBO(){
    double alpha, betta, rho1, rho2, ra, rb, L1, L2, u1, u2, u3, nu2, f1, f2;
    GBOVector x1, x2, x3, gsr, xs, dm, x_best, xr1, xr2, x_current, x_next, x_mk, x_p, x_rand, Y, term1, term2;
    populationVector trialVector;
    Individual trialIndividual;
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
            copyMaskedToVectorFast(individuals[indexOfCurrentVector].vector, wholeСoefficient, xs);
            gsrKit.xs = xs;
            gsrKit.indexes = generateFourUniqueIndices(indexOfCurrentVector, indexOfBestSolution);
            gsrKit.eps = randomDoubleZeroToOne() * 0.01;
            copyMaskedToVectorFast(individuals[indexOfBestSolution].vector, wholeСoefficient, x_best);
            gsrKit.x_best = x_best;
            copyMaskedToVectorFast(individuals[gsrKit.indexes[0]].vector, wholeСoefficient, xr1);
            gsrKit.xr1 = xr1;
            copyMaskedToVectorFast(individuals[gsrKit.indexes[1]].vector, wholeСoefficient, xr2);
            gsrKit.xr2 = xr2;
            copyMaskedToVectorFast(individuals[gsrKit.iocv].vector, wholeСoefficient, x_current);
            gsrKit.x_current = x_current;
            gsrKit.dm = randomDoubleZeroToOne() * gsrKit.rho1 * (gsrKit.x_best - gsrKit.xr1);
            gsr = GSR(gsrKit);
            dm = randomDoubleZeroToOne() * rho1 * (x_best - xr1);
            x1 = x_current - gsr + dm;
            
            gsrKit.dm = randomDoubleZeroToOne() * rho1 * (xr1 - xr2);
            copyMaskedToVectorFast(individuals[indexOfBestSolution].vector, wholeСoefficient, xs);
            gsrKit.xs = xs;
            gsr = GSR(gsrKit);
            dm = randomDoubleZeroToOne() * rho1 * (xr1 - xr2);
            x2 = x_best - gsr + dm;

            ra = randomDoubleZeroToOne();
            rb = randomDoubleZeroToOne();
            
            x3 = x_current - rho1 * (x2 - x1);
            x_next = ra * (rb * x1 + (1 - rb) * x2) + (1 - ra) * x3;
            clampVector(x_next);

            //LEO
            if (randomDoubleZeroToOne() < PR) {
                L1 = randomBinaryFast();
                L2 = randomBinaryFast();
                u1 = L1 * 2.0 * randomDoubleZeroToOne() + (1.0 - L1);
                u2 = L1 * randomDoubleZeroToOne() + (1.0 - L1);
                u3 = L1 * randomDoubleZeroToOne() + (1.0 - L1);
                nu2 = randomDoubleZeroToOne();
                fillRandom(x_rand);
                copyMaskedToVectorFast(individuals[generateFastPopulationIndex()].vector, wholeСoefficient, x_p);
                x_mk = L2 * x_p + (1.0 - L2) * x_rand;
                Y = (randomDoubleZeroToOne() < 0.5) ? x_next : x_best;
                f1 = randomOneFast();
                f2 = randomOneFast();
                term1 = f1 * (u1 * x_best - u2 * x_mk);
                term2 = f2 * rho1 * (u3 * (x2 - x1) + u2 * (xr1 - xr2));
                x_next = Y + term1 + term2;
                clampVector(x_next);
            }

            applyMaskedToPopulationVectorFast(x_next, wholeСoefficient, trialVector);
            trialIndividual.vector = trialVector;
            trialIndividual.calcObjectiveFunction(block, bit, wholeСoefficient, regionOfS0, regionOfS1);
            update(trialIndividual, indexOfCurrentVector);
        }
    }
    trialIndividual.applyVector(block, wholeСoefficient);
    cv::idct(block.frequencyDomain, block.spatialDomain);
}