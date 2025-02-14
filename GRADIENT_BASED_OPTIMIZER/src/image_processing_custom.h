#ifndef IMAGE_PROCESSING_CUSTOM_H
#define IMAGE_PROCESSING_CUSTOM_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

// Function to read an image in grayscale mode
cv::Mat readImage(const std::string& filename);

// Function to write an image to a file
bool writeImage(const std::string& filename, const cv::Mat& image);

// Function to split an image into 8x8 blocks
std::vector<cv::Mat> splitInto8x8Blocks(const cv::Mat& image);

// Function to merge 8x8 blocks into a single image
cv::Mat merge8x8Blocks(const std::vector<cv::Mat>& blocks, int rows, int cols);

// Function to convert a watermark (black and white image) into an array of 0 and 1
std::vector<int> convertWatermarkToBinary(const cv::Mat& watermark);

// Function to restore the watermark from an array of 0 and 1
cv::Mat convertBinaryToWatermark(const std::vector<int>& binaryData);

#endif // IMAGE_PROCESSING_CUSTOM_H
