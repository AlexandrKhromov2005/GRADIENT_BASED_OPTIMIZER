#ifndef POPULATION_H
#define POPULATION_H

#include <array>
#include <vector>
#include <utility>
#include "config.h"
#include "random_utils.h"
#include <opencv2/opencv.hpp>
#include <cmath>
#include "block_metrics.h"

using VecOf = std::pair<std::vector<double>, double>;

class Population {
public:
	std::vector<VecOf> vecs;
	VecOf worst_vec;
	size_t best_ind;

	Population();
	void initOf(const cv::Mat& block, uchar bit, int quality);
	cv::Mat apply_vec(const cv::Mat &block, const std::vector<double>& vec);
	double calculateOf(const cv::Mat &block, const std::vector<double>& vec, uchar bit, int quality);
	void update(const VecOf& trial, size_t vec_ind);
};



#endif // POPULATION_H
