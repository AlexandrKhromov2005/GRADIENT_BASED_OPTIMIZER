#pragma once

#include <opencv2/opencv.hpp>
#include "quantization_tables.h"

cv::Mat compress_block(cv::Mat& original_block, int quality);