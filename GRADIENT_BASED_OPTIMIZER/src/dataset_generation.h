#ifndef DATASET_GENERATION_H
#define DATASET_GENERATION_H

#include <opencv2/opencv.hpp>
#include <string>

// Attack functions
cv::Mat jpeg_attack(const cv::Mat& image, int quality = 70);
cv::Mat contrast_increase_attack(const cv::Mat& image, double alpha = 1.2);

// Original dataset generation function
void generate_dataset(double tau_max = 10.0);

// New functions with classifier integration
uchar extract_bit_from_block(const cv::Mat& block);

#ifdef TORCH_AVAILABLE
void generate_dataset_with_classifier(double tau_max = 10.0);
bool initialize_classifier_for_dataset();
#endif

#endif // DATASET_GENERATION_H