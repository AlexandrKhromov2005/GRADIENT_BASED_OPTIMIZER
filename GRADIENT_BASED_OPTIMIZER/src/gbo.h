#ifndef GBO_H
#define GBO_H

#include <opencv2/opencv.hpp>
#include <array>
#include "config.h"
#include <cmath>
#include "random_utils.h"
#include "population.h"
#include <algorithm>

class GBO {
public:
	uchar bit;
	cv::Mat &block;
	void main_loop();
	GBO(uchar bit, cv::Mat &block) : bit(bit), block(block) {}

};

#endif // GBO_H
