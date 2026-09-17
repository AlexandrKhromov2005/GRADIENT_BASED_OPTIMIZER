#ifndef EMBEDDING_CORE_H
#define EMBEDDING_CORE_H

#include <opencv2/opencv.hpp>
#include <vector>
#include "population.h"

// Attack type the quadrant (row, col) of the 4x4 grid is optimized for.
// Pattern: NONE/JPEG70/NONE/JPEG70 / CONTRAST/JPEG80/CONTRAST/JPEG80 (repeated)
AttackType quadrantAttackType(int row, int col);

// Base algorithm: embed wm_bits[i % WM_SIZE] into every 8x8 block of a grayscale image.
cv::Mat embedBits(const cv::Mat& gray, const std::vector<int>& wm_bits);

// Base algorithm: per-bit count of blocks voting for 1 (one vote per watermark copy).
std::vector<int> extractVotes(const cv::Mat& gray);

// Modified algorithm: 1024x1024 image, 16 quadrants 256x256, each optimized for its attack type.
cv::Mat embedBitsQuadrants(const cv::Mat& gray_1024, const std::vector<int>& wm_bits);

// Modified algorithm: per-bit count of votes for 1 among the quadrants of the given attack type.
std::vector<int> extractVotesQuadrants(const cv::Mat& gray_1024, AttackType attack_type);

#endif // EMBEDDING_CORE_H
