#ifndef DATASET_GENERATION_H
#define DATASET_GENERATION_H

#include <opencv2/opencv.hpp>
#include <string>

cv::Mat jpeg_attack(const cv::Mat& image, int quality = 70);
cv::Mat contrast_increase_attack(const cv::Mat& image, double alpha = 1.2);
void generate_dataset(double tau_max = 10.0);

#endif // DATASET_GENERATION_H