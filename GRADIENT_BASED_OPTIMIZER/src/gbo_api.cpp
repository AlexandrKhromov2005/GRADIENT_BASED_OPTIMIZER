#include "gbo_api.h"
#include "embedding_schemes.h"
#include "image_processing_custom.h"
#include "image_metrics.h"
#include "block_metrics.h"
#include "attacks.h"
#include "gbo.h"
#include "config.h"
#include "jpeg/quantization_tables.h"

#include <stdexcept>

namespace gbo {

static bool s_initialized = false;

// ---- Initialization --------------------------------------------------------

bool init(const std::string& schemes_json_path) {
    auto& mgr = EmbeddingSchemeManager::getInstance();
    bool ok = mgr.loadSchemes(schemes_json_path);
    if (ok) {
        initialize_quantization_mats();
        s_initialized = true;
    }
    return ok;
}

bool setScheme(const std::string& scheme_id) {
    auto& mgr = EmbeddingSchemeManager::getInstance();
    const auto* scheme = mgr.getScheme(scheme_id);
    if (!scheme) return false;
    mgr.setCurrentScheme(scheme_id);
    return true;
}

std::vector<std::string> availableSchemes() {
    return EmbeddingSchemeManager::getInstance().getAvailableSchemes();
}

// ---- Watermark embedding / extraction --------------------------------------

cv::Mat embedWatermark(const cv::Mat& image,
                       const cv::Mat& watermark) {
    if (!s_initialized) {
        throw std::runtime_error("gbo::init() must be called before embedWatermark()");
    }
    cv::Mat gray;
    if (image.channels() > 1) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }

    std::vector<cv::Mat> blocks = splitInto8x8Blocks(gray);
    std::vector<int> wm_bits = convertWatermarkToBinary(watermark);

    initialize_quantization_mats();

    for (size_t i = 0; i < blocks.size(); ++i) {
        GBO optimizer(static_cast<uchar>(wm_bits[i % WM_SIZE]), blocks[i]);
        optimizer.main_loop();
    }

    return merge8x8Blocks(blocks, gray.rows, gray.cols);
}

cv::Mat extractWatermark(const cv::Mat& image) {
    if (!s_initialized) {
        throw std::runtime_error("gbo::init() must be called before extractWatermark()");
    }
    cv::Mat gray;
    if (image.channels() > 1) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image;
    }

    std::vector<cv::Mat> blocks = splitInto8x8Blocks(gray);
    std::vector<int> wm_vec(WM_SIZE, 0);

    for (size_t i = 0; i < blocks.size(); ++i) {
        cv::Mat dbl_block;
        blocks[i].convertTo(dbl_block, CV_64F);
        cv::Mat dct_block;
        cv::dct(dbl_block, dct_block);
        double s0 = calc_s_zero(dct_block);
        double s1 = calc_s_one(dct_block);
        if (s0 < s1) {
            ++wm_vec[i % WM_SIZE];
        }
    }

    for (size_t i = 0; i < WM_SIZE; ++i) {
        switch (wm_vec[i]) {
        case 0: case 1: wm_vec[i] = 0; break;
        case 3: case 4: wm_vec[i] = 1; break;
        default:        wm_vec[i] = rand() % 2; break;
        }
    }

    return convertBinaryToWatermark(wm_vec);
}

// ---- Metrics ---------------------------------------------------------------

double computeMSE(const cv::Mat& a, const cv::Mat& b) {
    return ::computeMSE(a, b);
}

double computePSNR(const cv::Mat& a, const cv::Mat& b) {
    return ::computePSNR(a, b);
}

double computeSSIM(const cv::Mat& a, const cv::Mat& b) {
    return ::computeSSIM(a, b);
}

double computeNCC(const cv::Mat& a, const cv::Mat& b) {
    return ::computeNCC(a, b);
}

double computeBER(const cv::Mat& original_wm, const cv::Mat& extracted_wm) {
    return ::computeBER(original_wm, extracted_wm);
}

// ---- Attack simulation -----------------------------------------------------

cv::Mat attackJPEG(const cv::Mat& image, int quality) {
    return jpegCompression(image, quality);
}

cv::Mat attackBrightnessIncrease(const cv::Mat& image, int value) {
    return brightnessIncrease(image, value);
}

cv::Mat attackBrightnessDecrease(const cv::Mat& image, int value) {
    return brightnessDecrease(image, value);
}

cv::Mat attackContrastIncrease(const cv::Mat& image, double alpha) {
    return contrastIncrease(image, alpha);
}

cv::Mat attackContrastDecrease(const cv::Mat& image, double alpha) {
    return contrastDecrease(image, alpha);
}

cv::Mat attackSaltPepper(const cv::Mat& image, double noise_prob) {
    return saltPepperNoise(image, noise_prob);
}

cv::Mat attackMedianFilter(const cv::Mat& image, int ksize) {
    return medianFiltering(image, ksize);
}

cv::Mat attackGaussianFilter(const cv::Mat& image, int ksize) {
    return gaussianFiltering(image, ksize);
}

} // namespace gbo
