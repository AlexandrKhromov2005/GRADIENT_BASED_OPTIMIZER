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
#include <cstdint>

using VecOf = std::pair<std::vector<double>, double>;

enum class AttackType {
	NONE,           // No attack (original objective function)
	JPEG70,         // JPEG compression quality 70
	JPEG80,         // JPEG compression quality 80
	CONTRAST        // Contrast increase
};

class Population {
public:
	std::vector<VecOf> vecs;
	VecOf worst_vec;
	size_t best_ind;
	AttackType attack_type;

	Population();
	Population(AttackType attack);
	void initOf(const cv::Mat& block, uchar bit, int quality);
	cv::Mat apply_vec(const cv::Mat &block, const std::vector<double>& vec);
	double calculateOf(const cv::Mat &block, const std::vector<double>& vec, uchar bit, int quality);
	void update(const VecOf& trial, size_t vec_ind);

	// Block with the given change vector applied (DCT -> apply -> IDCT -> round), as written to the image.
	cv::Mat embedVec(const cv::Mat& block, const std::vector<double>& vec);

private:
	// Per-block data that does not depend on the change vector; filled on first use.
	void prepare(const cv::Mat& block);
	void modifiedPixels(const std::vector<double>& vec, uint8_t* out) const;

	bool prepared = false;
	uint8_t orig_pixels[64];
	double orig_dct[64];
	std::vector<int> zone_idx, reg0_idx, reg1_idx;  // row * 8 + col
};



#endif // POPULATION_H
