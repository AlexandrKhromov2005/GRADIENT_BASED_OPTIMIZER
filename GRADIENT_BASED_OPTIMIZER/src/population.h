#ifndef POPULATION_H
#define POPULATION_H

#include <array>
#include <utility>
#include "config.h"
#include "random_utils.h"
#include <opencv2/opencv.hpp>
#include <cmath>
#include "block_metrics.h"

using VecOf = std::pair<std::array<double, VEC_SIZE>, double>;

class Population {
public:
	std::array<VecOf, POP_SIZE> vecs;
	VecOf worst_vec;
	size_t best_ind;

	Population();
	void initOf(const cv::Mat& block, uchar bit);
	cv::Mat apply_vec(const cv::Mat &block, std::array<double, VEC_SIZE> vec);
	double calculateOf(const cv::Mat &block, const std::array<double, VEC_SIZE>& vec, uchar bit);
	void update(VecOf trial , size_t vec_ind);
	cv::Mat emulateJPEG70(const cv::Mat& block);

};



#endif // POPULATION_H
