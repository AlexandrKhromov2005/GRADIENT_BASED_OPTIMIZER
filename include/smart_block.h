#ifndef SMART_BLOCK_H
#define SMART_BLOCK_H

#include <opencv2/opencv.hpp>
#include "types.h"
#include "patterns.h"

struct Block {
    const blockDouble   spatialDomain_;
    const blockDouble   frequencyDomain_;
    blockDouble         spatialDomain;
    blockDouble         frequencyDomain;

    Block(cv::Mat_<double> spatialDomain_);
    double psnr();
    double calcAbsSum(const Mask& mask);
};

#endif // SMART_BLOCK_H