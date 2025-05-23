#include "../include/types.h"

//Operator+ for element-wise addition by mask
populationVector& operator+(populationVector& a, const std::pair<const populationVector&, const Mask&>& op) {
    const auto& [b, mask] = op;
    
    cv::parallel_for_(cv::Range(0, static_cast<int>(mask.size())), [&](const cv::Range& range) {
        for (int i = range.start; i < range.end; ++i) {
            const auto& [row, col] = mask[i];
            a(row, col) += b(row, col);  // Используем operator(), а не указатели
        }
    });
    return a;
}

//Operator* for multiplication by a scalar by mask
populationVector& operator*(populationVector& vec, const std::pair<double, const Mask&>& op) {
    const auto& [scalar, mask] = op;
    
    cv::parallel_for_(cv::Range(0, static_cast<int>(mask.size())), [&](const cv::Range& range) {
        for (int i = range.start; i < range.end; ++i) {
            const auto& [row, col] = mask[i];
            vec(row, col) *= scalar;
        }
    });
    return vec;
}
//Operator- for mask subtraction
populationVector& operator-(populationVector& a, const std::pair<const populationVector&, const Mask&>& op) {
    const auto& [b, mask] = op;
    CV_Assert(a.size() == b.size());
    
    cv::parallel_for_(cv::Range(0, static_cast<int>(mask.size())), [&](const cv::Range& range) {
        for (int i = range.start; i < range.end; ++i) {
            const auto& [row, col] = mask[i];
            a(row, col) -= b(row, col);  
        }
    });
    return a;
}

//safeDivideMasked for mask division
populationVector safeDivideMasked(
    const populationVector& a,
    const populationVector& b,
    const Mask& mask,
    double default_value = 0.0
) {
    populationVector result(a.size(), default_value);

    const double* a_ptr = a.ptr<double>();
    const double* b_ptr = b.ptr<double>();
    double* res_ptr = result.ptr<double>();
    const int cols = a.cols;

    cv::parallel_for_(cv::Range(0, static_cast<int>(mask.size())), [&](const cv::Range& range) {
        for (int i = range.start; i < range.end; ++i) {
            const auto& [row, col] = mask[i];
            const int idx = row * cols + col;
            res_ptr[idx] = (b_ptr[idx] != 0.0) ? (a_ptr[idx] / b_ptr[idx]) : default_value;
        }
    });

    return result;
}

//absPopulationVectorFast returns absolute value
populationVector absPopulationVectorFast(const populationVector& vec) {
    populationVector result(vec.rows, vec.cols);
    const double* src = vec.ptr<double>();
    double* dst = result.ptr<double>();
    const int total = vec.rows * vec.cols;

    cv::parallel_for_(cv::Range(0, total), [&](const cv::Range& range) {
        for (int i = range.start; i < range.end; ++i) {
            dst[i] = std::abs(src[i]);  
        }
    });
    
    return result;
}