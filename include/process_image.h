#ifndef PROCESS_IMAGE_H
#define PROCESS_IMAGE_H

#include <vector>
#include <opencv2/opencv.hpp>
#include "smart_block.h"

std::vector<Block> readImageAndSplitToBlocks(const std::string& imagePath);
cv::Mat reconstructImageFromBlocks(const std::vector<Block>& blocks, int rows, int cols);
std::vector<uchar> readCvzAndConvertToVector(const std::string& imagePath);
cv::Mat reconstructCvzFromVector(const std::vector<uchar>& cvzVector);
std::string addPrefixToFilename(const std::string& imagePath, const std::string& prefix);

#endif //PROCESS_IMAGE_H
