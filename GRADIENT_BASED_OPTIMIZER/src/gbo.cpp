#include "gbo.h"
#include <iostream>


std::array<double, VEC_SIZE> gsr_func(double rho2, std::array<double, VEC_SIZE> best_x, std::array<double, VEC_SIZE> worst_x, std::array<double, VEC_SIZE> cur_x, std::array<double, VEC_SIZE> xr1, std::array<double, VEC_SIZE> dm, std::array<double, VEC_SIZE> xm, size_t flag) {
	std::array<double, VEC_SIZE> gsr = { 0.0 };
	double a = rand_num();
	double b = static_cast<double>(gen_random_index());
	double c = randn();
	std::array<double, VEC_SIZE> delx = { 0.0 };
	double eps = rand_num() * 0.01;
	for (size_t i = 0; i < VEC_SIZE; ++i) {
		double delta = 2.0 * a * std::fabs(xm[i] - cur_x[i]);
		double step = 0.5 * (best_x[i] - xr1[i] + delta);
		delx[i] = b * std::fabs(step);
		gsr[i] = (c * rho2 * 2.0 * delx[i] * cur_x[i]) / (best_x[i] - worst_x[i] + eps );
	}

	std::array<double, VEC_SIZE> xs = (flag == 1) ? cur_x : best_x;
	for (size_t i = 0; i < VEC_SIZE; ++i) {
		xs[i] = xs[i] -  gsr[i] + dm[i];
	}

	double p1 = rand_num();
	double p2 = rand_num();
	double q1 = rand_num();
	double q2 = rand_num();
	double d = randn();

	for (size_t i = 0; i < VEC_SIZE; ++i) {
		double yp = p1 * (0.5 * (xs[i] + cur_x[i]) + p2 * delx[i]);
		double yq = q1 * (0.5 * (xs[i] + cur_x[i]) - q2 * delx[i]);
		gsr[i] = (d * rho2 * 2.0 * delx[i] * cur_x[i]) / (yp - yq + eps);
	}

	return gsr;
}

void GBO::main_loop() {
	Population population = Population();
	population.initOf(block, bit);
	//for (size_t i = 0; i < VEC_SIZE; ++i) {
	//	std::cout << population.vecs[i].second << " ";
	//}
	//std::cout << std::endl;
	//std::cout << "pop init" << std::endl;
	for (size_t m = 0; m < ITERATIONS; ++m) {
		double betta = 0.2 + (1.2 - 0.2) * pow(1.0 - pow(static_cast<double>(m + 1) / static_cast<double>(ITERATIONS), 3.0), 2.0);
		double angle = 1.5 * M_PI;
		double alpha = fabs(betta * sin(angle + sin(angle * betta)));

		for (size_t cur_vec = 0; cur_vec < POP_SIZE; ++cur_vec) {
			double rho1 = alpha * (2 * rand_num() - 1.0);
			double rho2 = alpha * (2 * rand_num() - 1.0);
			double dm_rand = rand_num();
			std::array<size_t, 4> indexes = {0};
			gen_indexes(indexes, cur_vec, population.best_ind);
			std::array<double, VEC_SIZE> x1 = {0.0}, x2 = {0.0}, x3 = {0.0}, xm = {0.0}, dm = {0.0}, gsr = {0.0};
			x1.fill(0);
			x2.fill(0);
			x3.fill(0);
			xm.fill(0);
			dm.fill(0);
			gsr.fill(0);

			for (size_t i = 0; i < VEC_SIZE; ++i) {
				xm[i] = (population.vecs[indexes[0]].first[i] + population.vecs[indexes[1]].first[i] + population.vecs[indexes[2]].first[i] + population.vecs[indexes[3]].first[i]) * 0.25;
				dm[i] = dm_rand * rho1 * (population.vecs[population.best_ind].first[i] - population.vecs[indexes[0]].first[i]);
			}
			
			gsr = gsr_func(rho2, population.vecs[population.best_ind].first, population.worst_vec.first, population.vecs[cur_vec].first, population.vecs[indexes[0]].first, dm, xm, 1);
			dm_rand = rand_num();
			dm.fill(0);
			for (size_t i = 0; i < VEC_SIZE; ++i) {
				dm[i] = dm_rand * rho1 * (population.vecs[population.best_ind].first[i] - population.vecs[indexes[0]].first[i]);
				x1[i] = population.vecs[cur_vec].first[i] - gsr[i] + dm[i];
			}

			dm_rand = rand_num();
			dm.fill(0);
			for (size_t i = 0; i < VEC_SIZE; ++i) {
				dm[i] = dm_rand * rho1 * (population.vecs[indexes[0]].first[i] - population.vecs[indexes[1]].first[i]);
			}
			gsr.fill(0);
			gsr = gsr_func(rho2, population.vecs[population.best_ind].first, population.worst_vec.first, population.vecs[cur_vec].first, population.vecs[indexes[0]].first, dm, xm, 2);

			dm_rand = rand_num();
			dm.fill(0);
			for (size_t i = 0; i < VEC_SIZE; ++i) {
				dm[i] = dm_rand * rho1 * (population.vecs[indexes[0]].first[i] - population.vecs[indexes[1]].first[i]);
				x2[i] = population.vecs[population.best_ind].first[i] - gsr[i] + dm[i];
			}

			rho1 = alpha * (2 * rand_num() - 1.0);
			std::array<double, VEC_SIZE> x_next = { 0.0 };
			x_next.fill(0);
			double ra = rand_num();
			double rb = rand_num();

			for (size_t i = 0; i < VEC_SIZE; ++i) {
				x3[i] = population.vecs[cur_vec].first[i] - rho1 * (x2[i] - x1[i]);
				x_next[i] = ra * (rb * x1[i] + (1 - rb) * x2[i]) + (1 - ra) * x3[i];
				x_next[i] = std::clamp(x_next[i], -TH, TH);
			}


			if (rand_num() < PR) {
				double L1 = (rand_num() < 0.5) ? 0.0 : 1.0;
				double u1 = L1 * 2.0 * rand_num() + (1.0 - L1);
				double u2 = L1 * rand_num() + (1.0 - L1);
				double u3 = L1 * rand_num() + (1.0 - L1);

				double nu2 = rand_num();
				std::array<double, VEC_SIZE> x_mk = {};
				x_mk.fill(0);
				std::array<double, VEC_SIZE> x_p = population.vecs[gen_random_index()].first;
				std::array<double, VEC_SIZE> x_rand;
				x_rand.fill(0);
				for (size_t i = 0; i < VEC_SIZE; ++i) {
					x_rand[i] = TH * (2.0 * rand_num() - 1.0);
				}
				double L2 = (rand_num() < 0.5) ? 0.0 : 1.0;
				
				for (size_t i = 0; i < VEC_SIZE; ++i) {
					x_mk[i] = L2 * x_p[i] + (1.0 - L2) * x_rand[i];
				}
					
				std::array<double, VEC_SIZE> Y = (rand_num() < 0.5) ? x_next : population.vecs[population.best_ind].first;
				double f1 = rand_neg_one_to_one();
				double f2 = rand_neg_one_to_one();

				for (size_t i = 0; i < VEC_SIZE; ++i) {
					x_next[i] = Y[i] + f1 * (u1 * population.vecs[population.best_ind].first[i] - u2 * x_mk[i]) + f2 * rho1 * (u3 * (x2[i] - x1[i]) + u2 * (population.vecs[indexes[0]].first[i] - population.vecs[indexes[1]].first[i])) * 0.5;
					x_next[i] = std::clamp(x_next[i], -TH, TH);
				}
			}

			//std::cout << "leo end" << std::endl;


			double x_next_of = population.calculateOf(block, x_next, bit);
			VecOf trial = {};
			trial.first = x_next;
			trial.second = x_next_of;
			population.update(trial, cur_vec);
			//std::cout << population.vecs[population.best_ind].second << std::endl;


		}
	}

	cv::Mat blockDouble;
	block.convertTo(blockDouble, CV_64F);
	cv::Mat DCTblock;
	cv::dct(blockDouble, DCTblock);
	cv::Mat newDCTblock = population.apply_vec(DCTblock, population.vecs[population.best_ind].first);
	cv::Mat newblockDouble;
	cv::idct(newDCTblock, newblockDouble);
	cv::Mat newblock;
	newblockDouble.convertTo(newblock, CV_8U);
	block = newblock.clone();
}