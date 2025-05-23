#ifndef TYPES_H
#define TYPES_H

#include <opencv2/opencv.hpp>

using blockDouble = cv::Mat_<double>;
using populationVector = cv::Mat_<double>;
using Mask = std::vector<std::pair<std::size_t, std::size_t>>;
using MaskCollection = std::vector<Mask>;

populationVector& operator+(populationVector& a, const std::pair<const populationVector&, const Mask&>& op);
populationVector& operator*(populationVector& vec, const std::pair<double, const Mask&>& op);
populationVector& operator-(populationVector& a, const std::pair<const populationVector&, const Mask&>& op);

populationVector absPopulationVectorFast(const populationVector& vec);
populationVector safeDivideMasked(const populationVector& a, const populationVector& b, const Mask& mask, double default_value = 0.0);
#endif //TYPES_H