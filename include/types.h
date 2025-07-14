#ifndef TYPES_H
#define TYPES_H

#include <opencv2/opencv.hpp>
#include <vector>
#include "config.h"

using blockDouble = cv::Mat_<double>;
using populationVector = cv::Mat_<double>;
using Mask = std::vector<std::pair<std::size_t, std::size_t>>;
using MaskCollection = std::vector<Mask>;
using GBOVector = std::vector<double>;

void copyMaskedToVectorFast(const populationVector& src, const Mask& mask, GBOVector& dest);
void applyMaskedToPopulationVectorFast(const GBOVector& source, const Mask& mask, populationVector& dest);
GBOVector operator+(const GBOVector& a, const GBOVector& b);
GBOVector operator-(const GBOVector& a, const GBOVector& b);
GBOVector operator*(const GBOVector& a, const GBOVector& b);
GBOVector operator*(const GBOVector& a, double scalar);
GBOVector operator*(double scalar, const GBOVector& a);
GBOVector operator/(const GBOVector& a, const GBOVector& b);
GBOVector abs(const GBOVector& vec);
GBOVector operator+(const GBOVector& vec, double scalar);
GBOVector operator+(double scalar, const GBOVector& vec);
void clampVector(GBOVector& vec, double th = TH);
void fillRandom(GBOVector& vec);

#endif //TYPES_H