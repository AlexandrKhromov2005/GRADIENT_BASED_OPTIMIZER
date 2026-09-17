#include "population.h"
#include "embedding_schemes.h"
#include "block_kernels.h"
#include <cstring>
#include <iostream>
#include <sstream>

Population::Population() : attack_type(AttackType::NONE) {
    vecs.resize(POP_SIZE);
    for (size_t i = 0; i < POP_SIZE; ++i) {
        vecs[i].first.resize(CURRENT_VEC_SIZE);
        for (size_t j = 0; j < CURRENT_VEC_SIZE; ++j) {
            vecs[i].first[j] = TH * (2.0 * rand_num() - 1.0);
        }
        vecs[i].second = DBL_MAX;
    }
    best_ind = 0;
    worst_vec.first = vecs[0].first;
    worst_vec.second = -DBL_MAX;
}

Population::Population(AttackType attack) : attack_type(attack) {
    vecs.resize(POP_SIZE);
    for (size_t i = 0; i < POP_SIZE; ++i) {
        vecs[i].first.resize(CURRENT_VEC_SIZE);
        for (size_t j = 0; j < CURRENT_VEC_SIZE; ++j) {
            vecs[i].first[j] = TH * (2.0 * rand_num() - 1.0);
        }
        vecs[i].second = DBL_MAX;
    }
    best_ind = 0;
    worst_vec.first = vecs[0].first;
    worst_vec.second = -DBL_MAX;
}

cv::Mat Population::apply_vec(const cv::Mat& block, const std::vector<double>& vec) {
    const auto& ZONE0 = getCurrentZONE0();

    cv::Mat new_block = block.clone();

    // Use full scheme size
    size_t max_elements = std::min(vec.size(), ZONE0.size());
    for (size_t i = 0; i < max_elements; ++i) {
        int row = ZONE0[i].first;
        int col = ZONE0[i].second;

        double original_val = block.at<double>(row, col);
        double computed_val = SIGN(original_val) * std::fabs(std::fabs(original_val) + vec[i]);
        new_block.at<double>(row, col) = computed_val;
    }

    return new_block;
}



void Population::prepare(const cv::Mat& block) {
    CV_Assert(block.type() == CV_8UC1 && block.rows == 8 && block.cols == 8);
    kernels::loadBlock(block, orig_pixels);
    // Once per block, so cv::dct costs nothing here. It is kept on purpose: coefficients that
    // are zero in exact arithmetic come out as +-1e-13 noise, and SIGN() of that noise decides
    // the sign of the embedded coefficient - so the very same transform must produce it.
    cv::Mat block_double, block_dct;
    block.convertTo(block_double, CV_64F);
    cv::dct(block_double, block_dct);
    for (int r = 0; r < 8; ++r) std::memcpy(orig_dct + 8 * r, block_dct.ptr<double>(r), 8 * sizeof(double));

    auto flatten = [](const std::vector<std::pair<int, int>>& coords, std::vector<int>& idx) {
        idx.clear();
        for (const auto& c : coords) idx.push_back(c.first * 8 + c.second);
    };
    flatten(getCurrentZONE0(), zone_idx);
    flatten(getCurrentREG0(), reg0_idx);
    flatten(getCurrentREG1(), reg1_idx);
    prepared_data = block.data;
    prepared = true;
}

void Population::modifiedPixels(const std::vector<double>& vec, uint8_t* out) const {
    double coefs[64], pixels[64];
    std::memcpy(coefs, orig_dct, sizeof(coefs));
    const size_t max_elements = std::min(vec.size(), zone_idx.size());
    for (size_t i = 0; i < max_elements; ++i) {
        const double original_val = orig_dct[zone_idx[i]];
        coefs[zone_idx[i]] = SIGN(original_val) * std::fabs(std::fabs(original_val) + vec[i]);
    }
    kernels::idct8x8(coefs, pixels);
    if (!kernels::roundToU8(pixels, out)) {
        // A pixel sits on a rounding boundary: let the reference transform decide, so the
        // result equals cv::idct + convertTo(CV_8U) in every case, not just almost surely.
        cv::Mat reference;
        cv::idct(cv::Mat(8, 8, CV_64F, coefs), reference);
        kernels::roundToU8(reference.ptr<double>(), out);
    }
}

cv::Mat Population::embedVec(const cv::Mat& block, const std::vector<double>& vec) {
    if (!prepared || block.data != prepared_data) prepare(block);
    uint8_t pixels[64];
    modifiedPixels(vec, pixels);
    cv::Mat result(8, 8, CV_8U);
    kernels::storeBlock(pixels, result);
    return result;
}

double Population::calculateOf(const cv::Mat& block, const std::vector<double>& vec, uchar bit, int quality) {
    if (!prepared || block.data != prepared_data) prepare(block);

    uint8_t embedded[64], attacked_buf[64];
    modifiedPixels(vec, embedded);

    // Apply attack if specified
    const uint8_t* attacked = embedded;
    if (attack_type == AttackType::JPEG70) {
        kernels::jpegRoundTrip(embedded, attacked_buf, 70);
        attacked = attacked_buf;
    } else if (attack_type == AttackType::JPEG80) {
        kernels::jpegRoundTrip(embedded, attacked_buf, 80);
        attacked = attacked_buf;
    } else if (attack_type == AttackType::CONTRAST) {
        kernels::contrastU8(embedded, attacked_buf, 1.1);
        attacked = attacked_buf;
    }

    // Calculate s0 and s1 from attacked block (or original if no attack)
    double attacked_pixels[64], attacked_dct[64];
    for (int i = 0; i < 64; ++i) attacked_pixels[i] = attacked[i];
    kernels::dct8x8(attacked_pixels, attacked_dct);

    double s0 = 0.0, s1 = 0.0;
    for (int idx : reg0_idx) s0 += std::fabs(attacked_dct[idx]);
    for (int idx : reg1_idx) s1 += std::fabs(attacked_dct[idx]);

    // PSNR between attacked and original block
    double mse = 0.0;
    for (int i = 0; i < 64; ++i) {
        const double diff = static_cast<double>(orig_pixels[i]) - static_cast<double>(attacked[i]);
        mse += diff * diff;
    }
    mse /= 64.0;
    const double psnr = (mse == 0) ? 100.0 : 10.0 * std::log10((255.0 * 255.0) / mse);

    if (s0 < 0.001 || std::isnan(s0) || std::isinf(s0)) s0 = 0.001;
    if (s1 < 0.001 || std::isnan(s1) || std::isinf(s1)) s1 = 0.001;

    double val = (bit == 0) ? s1 / s0 : s0 / s1;
    return val - 0.01 * psnr;
}


void Population::initOf(const cv::Mat& block, uchar bit, int quality) {
    double ofbest = DBL_MAX;
    double ofworst = -DBL_MAX;

    size_t ibest = 0, iworst = 0;
    for (size_t i = 0; i < POP_SIZE; ++i) {
        vecs[i].second = calculateOf(block, vecs[i].first, bit, quality);
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

void Population::update(const VecOf& trial, size_t vec_ind) {
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