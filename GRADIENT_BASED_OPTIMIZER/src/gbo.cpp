#include "gbo.h"
#include "embedding_schemes.h"
#include <iostream>
#include <algorithm>


// Gradient search rule. `gsr`, `delx` and `xs` are caller-provided work buffers (no allocations).
static void gsr_func(std::vector<double>& gsr, std::vector<double>& delx, std::vector<double>& xs, double rho2, const std::vector<double>& best_x, const std::vector<double>& worst_x, const std::vector<double>& cur_x, const std::vector<double>& xr1, const std::vector<double>& dm, const std::vector<double>& xm, size_t flag) {
	double a = rand_num();
	double b = static_cast<double>(gen_random_index());
	double c = randn();
	double eps = rand_num() * 0.01;
	for (size_t i = 0; i < CURRENT_VEC_SIZE; ++i) {
		double delta = 2.0 * a * std::fabs(xm[i] - cur_x[i]);
		double step = 0.5 * (best_x[i] - xr1[i] + delta);
		delx[i] = b * std::fabs(step);
		gsr[i] = (c * rho2 * 2.0 * delx[i] * cur_x[i]) / (best_x[i] - worst_x[i] + eps );
	}

	const std::vector<double>& xs_base = (flag == 1) ? cur_x : best_x;
	for (size_t i = 0; i < CURRENT_VEC_SIZE; ++i) {
		xs[i] = xs_base[i] -  gsr[i] + dm[i];
	}

	double p1 = rand_num();
	double p2 = rand_num();
	double q1 = rand_num();
	double q2 = rand_num();
	double d = randn();

	for (size_t i = 0; i < CURRENT_VEC_SIZE; ++i) {
		double yp = p1 * (0.5 * (xs[i] + cur_x[i]) + p2 * delx[i]);
		double yq = q1 * (0.5 * (xs[i] + cur_x[i]) - q2 * delx[i]);
		gsr[i] = (d * rho2 * 2.0 * delx[i] * cur_x[i]) / (yp - yq + eps);
	}
}

size_t GBO::getVectorSize() {
	const auto& scheme = EmbeddingSchemeManager::getInstance().getCurrentScheme();
	if (scheme) {
		return scheme->getTotalVectorSize();
	}
	return CURRENT_VEC_SIZE; // fallback
}

void GBO::main_loop() {
	Population population = Population(attack_type);
	int quality = rand_int_1_to_100();
	population.initOf(block, bit, quality);
	
	// Work buffers, allocated once per block instead of once per trial vector
	const size_t n = CURRENT_VEC_SIZE;
	std::vector<double> x1(n), x2(n), xm(n), dm(n), gsr(n), delx(n), xs(n), x_rand(n), x_mk(n);
	VecOf trial;
	trial.first.resize(n);
	std::vector<double>& x_next = trial.first;

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

			const std::vector<double>& x_best = population.vecs[population.best_ind].first;
			const std::vector<double>& x_cur = population.vecs[cur_vec].first;
			const std::vector<double>& x_r1 = population.vecs[indexes[0]].first;
			const std::vector<double>& x_r2 = population.vecs[indexes[1]].first;
			const std::vector<double>& x_r3 = population.vecs[indexes[2]].first;
			const std::vector<double>& x_r4 = population.vecs[indexes[3]].first;

			for (size_t i = 0; i < n; ++i) {
				xm[i] = (x_r1[i] + x_r2[i] + x_r3[i] + x_r4[i]) * 0.25;
				dm[i] = dm_rand * rho1 * (x_best[i] - x_r1[i]);
			}

			gsr_func(gsr, delx, xs, rho2, x_best, population.worst_vec.first, x_cur, x_r1, dm, xm, 1);
			dm_rand = rand_num();
			for (size_t i = 0; i < n; ++i) {
				dm[i] = dm_rand * rho1 * (x_best[i] - x_r1[i]);
				x1[i] = x_cur[i] - gsr[i] + dm[i];
			}

			dm_rand = rand_num();
			for (size_t i = 0; i < n; ++i) {
				dm[i] = dm_rand * rho1 * (x_r1[i] - x_r2[i]);
			}
			gsr_func(gsr, delx, xs, rho2, x_best, population.worst_vec.first, x_cur, x_r1, dm, xm, 2);

			dm_rand = rand_num();
			for (size_t i = 0; i < n; ++i) {
				dm[i] = dm_rand * rho1 * (x_r1[i] - x_r2[i]);
				x2[i] = x_best[i] - gsr[i] + dm[i];
			}

			rho1 = alpha * (2 * rand_num() - 1.0);
			double ra = rand_num();
			double rb = rand_num();

			for (size_t i = 0; i < n; ++i) {
				double x3 = x_cur[i] - rho1 * (x2[i] - x1[i]);
				x_next[i] = ra * (rb * x1[i] + (1 - rb) * x2[i]) + (1 - ra) * x3;
				x_next[i] = std::max(-TH, std::min(TH, x_next[i]));
			}


			if (rand_num() < PR) {
				double L1 = (rand_num() < 0.5) ? 0.0 : 1.0;
				double u1 = L1 * 2.0 * rand_num() + (1.0 - L1);
				double u2 = L1 * rand_num() + (1.0 - L1);
				double u3 = L1 * rand_num() + (1.0 - L1);

				double nu2 = rand_num();
				(void)nu2;
				const std::vector<double>& x_p = population.vecs[gen_random_index()].first;
				for (size_t i = 0; i < n; ++i) {
					x_rand[i] = TH * (2.0 * rand_num() - 1.0);
				}
				double L2 = (rand_num() < 0.5) ? 0.0 : 1.0;

				for (size_t i = 0; i < n; ++i) {
					x_mk[i] = L2 * x_p[i] + (1.0 - L2) * x_rand[i];
				}

				const std::vector<double>& Y = (rand_num() < 0.5) ? x_next : x_best;
				double f1 = rand_neg_one_to_one();
				double f2 = rand_neg_one_to_one();

				for (size_t i = 0; i < n; ++i) {
					x_next[i] = Y[i] + f1 * (u1 * x_best[i] - u2 * x_mk[i]) + f2 * rho1 * (u3 * (x2[i] - x1[i]) + u2 * (x_r1[i] - x_r2[i])) * 0.5;
					x_next[i] = std::max(-TH, std::min(TH, x_next[i]));
				}
			}

			trial.second = population.calculateOf(block, x_next, bit, quality);
			population.update(trial, cur_vec);
		}
	}

	block = population.embedVec(block, population.vecs[population.best_ind].first);
}