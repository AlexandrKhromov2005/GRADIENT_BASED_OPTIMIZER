#ifndef ATTACKS_H
#define ATTACKS_H

#include <opencv2/opencv.hpp>

cv::Mat brightnessIncrease(const cv::Mat& image, int value);
cv::Mat brightnessDecrease(const cv::Mat& image, int value);
cv::Mat contrastIncrease(const cv::Mat& image, double alpha);
cv::Mat contrastDecrease(const cv::Mat& image, double alpha);
cv::Mat saltPepperNoise(const cv::Mat& image, double noiseProb);
cv::Mat speckleNoise(const cv::Mat& image, double noiseStddev);
cv::Mat histogramEqualization(const cv::Mat& image);
cv::Mat sharpening(const cv::Mat& image);
cv::Mat jpegCompression(const cv::Mat& image, int quality);
cv::Mat gaussianFiltering(const cv::Mat& image, int ksize);
cv::Mat medianFiltering(const cv::Mat& image, int ksize);
cv::Mat averageFiltering(const cv::Mat& image, int ksize);
cv::Mat cropFromCorner(const cv::Mat& image, int cropSize);
cv::Mat cropFromCenter(const cv::Mat& image, int cropSize);
cv::Mat cropFromEdge(const cv::Mat& image, int cropSize);

#endif
