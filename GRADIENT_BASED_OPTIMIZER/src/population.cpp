#include "population.h"
#include <iostream>
#include <sstream>

Population::Population() {
    for (size_t i = 0; i < POP_SIZE; ++i) {
        for (size_t j = 0; j < VEC_SIZE; ++j) {
            vecs[i].first[j] = TH * (2.0 * rand_num() - 1.0);
        }
        vecs[i].second = DBL_MAX;
    }
    best_ind = 0;
    worst_vec.first = vecs[0].first;
    worst_vec.second = -DBL_MAX;
}

cv::Mat Population::apply_vec(const cv::Mat& block, std::array<double, VEC_SIZE> vec) {
    constexpr std::array<std::pair<std::size_t, std::size_t>, 22> ZONE0 = { {
        {6, 0}, {5, 1}, {4, 2}, {3, 3},
        {2, 4}, {1, 5}, {0, 6}, {0, 7},
        {1, 6}, {2, 5}, {3, 4}, {4, 3},
        {5, 2}, {6, 1}, {7, 0}, {7, 1},
        {6, 2}, {5, 3}, {4, 4}, {3, 5},
        {2, 6}, {1, 7}
    } };

    cv::Mat new_block = block.clone();

    //std::ostringstream debugStream;
    //debugStream << "apply_vec: ";

    for (size_t i = 0; i < VEC_SIZE; ++i) {
        int row = static_cast<int>(ZONE0[i].first);
        int col = static_cast<int>(ZONE0[i].second);

        double original_val = block.at<double>(row, col);
        double computed_val = SIGN(original_val) * std::fabs(std::fabs(original_val) + vec[i]);
        new_block.at<double>(row, col) = computed_val;


        //debugStream << "[" << row << "," << col << "] " << original_val << "->" << computed_val << "  ";
    }
    //std::cout << debugStream.str() << std::endl;
    //cv::Mat test1, test2, test3, test4;
    //cv::idct(block, test1);
    //cv::idct(new_block, test2);
    //test1.convertTo(test3, CV_8U);
    //test2.convertTo(test4, CV_8U);
    //double psnr = calculatePSNR(test3, test4);
    //std::cout << "app psnr: " << psnr << std::endl;

    return new_block;
}

cv::Mat Population::emulateJPEG70(const cv::Mat& block) {
    const int quality = 70;
    static const std::array<std::array<int, 8>, 8> std_quantization_matrix = { {
        {16, 11, 10, 16, 24, 40, 51, 61},
        {12, 12, 14, 19, 26, 58, 60, 55},
        {14, 13, 16, 24, 40, 57, 69, 56},
        {14, 17, 22, 29, 51, 87, 80, 62},
        {18, 22, 37, 56, 68, 109, 103, 77},
        {24, 35, 55, 64, 81, 104, 113, 92},
        {49, 64, 78, 87, 103, 121, 120, 101},
        {72, 92, 95, 98, 112, 100, 103, 99}
    } };

    double scale_factor = (quality < 50) ? 5000.0 / quality : 200.0 - 2.0 * quality;

    std::array<std::array<int, 8>, 8> quantization_matrix;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            int q = static_cast<int>(std_quantization_matrix[i][j] * scale_factor / 100.0);
            quantization_matrix[i][j] = (q > 0) ? q : 1;
        }
    }

    cv::Mat blockDouble;
    block.convertTo(blockDouble, CV_64F);
    cv::Mat dctBlock;
    cv::dct(blockDouble, dctBlock);

    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            dctBlock.at<double>(i, j) = std::round(dctBlock.at<double>(i, j) / quantization_matrix[i][j]) * quantization_matrix[i][j];
        }
    }

    cv::Mat idctBlock;
    cv::idct(dctBlock, idctBlock);
    cv::Mat result;
    idctBlock.convertTo(result, CV_8U);

    return result;
}

double Population::calculateOf(const cv::Mat& block, const std::array<double, VEC_SIZE>& vec, uchar bit) {
    cv::Mat blockDouble;
    block.convertTo(blockDouble, CV_64F);
    cv::Mat DCTblock;
    cv::dct(blockDouble, DCTblock);
    cv::Mat newDCTblock = apply_vec(DCTblock, vec);

    cv::Mat newblockDouble;
    cv::idct(newDCTblock, newblockDouble);
    cv::Mat newblock;
    newblockDouble.convertTo(newblock, CV_8U);

    // Применяем атаку JPEG70
    cv::Mat attackedBlock = emulateJPEG70(newblock);

    // Вычисляем DCT атакованного блока
    cv::Mat attackedBlockDouble;
    attackedBlock.convertTo(attackedBlockDouble, CV_64F);
    cv::Mat attackedDCTblock;
    cv::dct(attackedBlockDouble, attackedDCTblock);

    double s0 = calc_s_zero(attackedDCTblock);
    double s1 = calc_s_one(attackedDCTblock);
    double psnr = calculatePSNR(block, newblock);

    if (s0 < 0.001 || std::isnan(s0) || std::isinf(s0)) s0 = 0.001;
    if (s1 < 0.001 || std::isnan(s1) || std::isinf(s1)) s1 = 0.001;

    double val = (bit == 0) ? s1 / s0 : s0 / s1;
    return val - 0.01 * psnr;
}


void Population::initOf(const cv::Mat& block, uchar bit) {
    double ofbest = DBL_MAX;
    double ofworst = -DBL_MAX;

    size_t ibest = 0, iworst = 0;
    for (size_t i = 0; i < POP_SIZE; ++i) {
        vecs[i].second = calculateOf(block, vecs[i].first, bit);
        if (vecs[i].second > ofworst) {
            ofworst = vecs[i].second;
            iworst = i;
        }
        if (vecs[i].second < ofbest) {
            ofbest = vecs[i].second;
            ibest = i;
        }
    }

    best_ind = ibest;
    worst_vec.first = vecs[iworst].first;
    worst_vec.second = vecs[iworst].second;

}

void Population::update(VecOf trial, size_t vec_ind) {
    if (trial.second < vecs[vec_ind].second) {
        vecs[vec_ind] = trial;
        if (vecs[vec_ind].second < vecs[best_ind].second) {
            best_ind = vec_ind;
        }
    }
    else if (vecs[vec_ind].second > worst_vec.second) {
        worst_vec = vecs[vec_ind];
    }
}