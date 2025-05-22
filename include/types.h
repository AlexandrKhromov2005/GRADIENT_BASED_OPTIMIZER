#ifndef TYPES_H
#define TYPES_H

#include <opencv2/opencv.hpp>

using blockDouble = cv::Mat_<double>;
using populationVector = cv::Mat_<double>;
using Mask = std::vector<std::pair<std::size_t, std::size_t>>;
using MaskCollection = std::vector<Mask>;

#endif //TYPES_H