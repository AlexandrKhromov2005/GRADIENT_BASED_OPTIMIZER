#include "../include/types.h"
#include <algorithm>
#include <execution>
#include <random>
#include <chrono>

//copyMaskedToVectorFast populationVector ==> GBOVector
void copyMaskedToVectorFast(const populationVector& src,
                           const Mask& mask,
                           std::vector<double>& dest) {
    dest.resize(mask.size());
    double* dest_ptr = dest.data();
    
    cv::parallel_for_(cv::Range(0, static_cast<int>(mask.size())), [&](const cv::Range& range) {
        for (int i = range.start; i < range.end; ++i) {
            const auto& [row, col] = mask[i];
            dest_ptr[i] = src(row, col);
        }
    });
}

//applyMaskedToPopulationVectorFast GBOVector ==> populationVector
void applyMaskedToPopulationVectorFast(const GBOVector& source,
                                     const Mask& mask,
                                     populationVector& dest) {
    double* dest_ptr = dest.ptr<double>();
    const int cols = dest.cols;
    
    cv::parallel_for_(cv::Range(0, static_cast<int>(mask.size())), [&](const cv::Range& range) {
        for (int i = range.start; i < range.end; ++i) {
            const auto& [row, col] = mask[i];
            dest_ptr[row * cols + col] = source[i];
        }
    });
}

//operator+ GBOVector + GBOVector
GBOVector operator+(const GBOVector& a, const GBOVector& b) {
    GBOVector result(a.size());
    std::transform(std::execution::par_unseq, 
                  a.begin(), a.end(), b.begin(), 
                  result.begin(), std::plus<>());
    return result;
}

//operator- GBOVector - GBOVector
GBOVector operator-(const GBOVector& a, const GBOVector& b) {
    GBOVector result(a.size());
    std::transform(std::execution::par_unseq, 
                  a.begin(), a.end(), b.begin(), 
                  result.begin(), std::minus<>());
    return result;
}

//operator* GBOVector * GBOVector
GBOVector operator*(const GBOVector& a, const GBOVector& b) {
    GBOVector result(a.size());
    std::transform(std::execution::par_unseq, 
                  a.begin(), a.end(), b.begin(), 
                  result.begin(), std::multiplies<>());
    return result;
}

//operator* GBOVector * scalar
GBOVector operator*(const GBOVector& a, double scalar) {
    GBOVector result(a.size());
    std::transform(std::execution::par_unseq, 
                  a.begin(), a.end(), 
                  result.begin(), 
                  [scalar](double val) { return val * scalar; });
    return result;
}

//operator* scalar * GBOVector
GBOVector operator*(double scalar, const GBOVector& a) {
    return a * scalar;
}

//operator/ GBOVector / GBOVector
GBOVector operator/(const GBOVector& a, const GBOVector& b) {
    GBOVector result(a.size());
    std::transform(std::execution::par_unseq, 
                  a.begin(), a.end(), b.begin(), 
                  result.begin(), [](double x, double y) { 
                      return y != 0.0 ? x / y : 0.0; 
                  });
    return result;
}

//abs returns absolute value of each element
GBOVector abs(const GBOVector& vec) {
    GBOVector result(vec.size());
    std::transform(std::execution::par_unseq, 
                  vec.begin(), vec.end(), 
                  result.begin(), 
                  [](double val) { return std::abs(val); });
    return result;
}

////operator+ GBOVector + scalar
GBOVector operator+(const GBOVector& vec, double scalar) {
    GBOVector result(vec.size());
    std::transform(std::execution::par_unseq,
                  vec.begin(), vec.end(),
                  result.begin(),
                  [scalar](double val) { return val + scalar; });
    return result;
}

//operator+ scalar + GBOVector
GBOVector operator+(double scalar, const GBOVector& vec) {
    return vec + scalar; 
}

//clampVector clamps all values of vector
void clampVector(GBOVector& vec, double th = TH) {
    std::transform(std::execution::par_unseq,
                  vec.begin(), vec.end(),
                  vec.begin(),
                  [th](double val) {
                      return std::clamp(val, -TH, TH);
                  });
}

//fillRandom returns random GBOVector
void fillRandom(GBOVector& vec) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-TH, TH);

    for (auto& val : vec) {
        val = dist(gen);
    }
}