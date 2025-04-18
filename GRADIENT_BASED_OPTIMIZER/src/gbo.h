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
	int index;
	cv::Mat &block;
	void main_loop();
	GBO(uchar bit, cv::Mat &block, int index) : bit(bit), block(block), index(index) {}

};

#endif // GBO_H
