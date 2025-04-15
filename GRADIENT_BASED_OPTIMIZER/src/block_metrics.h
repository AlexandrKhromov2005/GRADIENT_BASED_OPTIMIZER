#ifndef BLOCK_METRICS_H
#define BLOCK_METRICS_H

#include <opencv2/core.hpp>

double calculateMSE(const cv::Mat& block1, const cv::Mat& block2);
double calculatePSNR(const cv::Mat& block1, const cv::Mat& block2);
double calc_s_zero(const cv::Mat& block);
double calc_s_one(const cv::Mat& block);

#endif // BLOCK_METRICS_H