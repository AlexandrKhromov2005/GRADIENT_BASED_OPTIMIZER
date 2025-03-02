#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <limits>
#include "block_metrics.h"

double calculateMSE(const cv::Mat& block1, const cv::Mat& block2) {
    double mse = 0.0;
    cv::Mat bloc1dbl, bloc2dbl;
    block1.convertTo(bloc1dbl, CV_64F);
    block2.convertTo(bloc2dbl, CV_64F);

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            double diff = (bloc1dbl.at<double>(i, j)) - (bloc2dbl.at<double>(i, j));
            mse += (diff * diff);
        }
    }
    mse /= 64.0;
    return mse;
}

double calculatePSNR(const cv::Mat& block1, const cv::Mat& block2) {
    double mse = calculateMSE(block1, block2);
    if (mse == 0) {
        return 100.0;
    }
    double psnr = 10.0 * std::log10((255.0 * 255.0) / mse);
    return psnr;
}

double calc_s_zero(const cv::Mat& block) {
    constexpr std::array<std::pair<std::size_t, std::size_t>, 11> REG0 = { {
        {7, 1}, {6, 1}, {5, 1},
        {5, 3}, {4, 3}, {3, 3},
        {3, 5}, {2, 5}, {1, 5},
        {1, 7}, {0, 7}
    } };

    double sum = 0.0;
    for (const auto& coord : REG0) {
        int row = static_cast<int>(coord.first);
        int col = static_cast<int>(coord.second);
        double value = block.at<double>(row, col);
        sum += std::fabs(value);
    }

    return sum;
}

double calc_s_one(const cv::Mat& block) {
    constexpr std::array<std::pair<std::size_t, std::size_t>, 11> REG1 = { {
        {7, 0}, {6, 0},
        {6, 2}, {5, 2}, {4, 2},
        {4, 4}, {3, 4}, {2, 4},
        {2, 6}, {1, 6}, {0, 6}
    } };

    double sum = 0.0;
    for (const auto& coord : REG1) {
        int row = static_cast<int>(coord.first);
        int col = static_cast<int>(coord.second);
        float value = block.at<double>(row, col);
        sum += std::fabs(value);
    }

    return sum;
}
