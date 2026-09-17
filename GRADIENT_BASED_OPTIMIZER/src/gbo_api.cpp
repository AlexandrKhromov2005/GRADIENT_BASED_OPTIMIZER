#include "gbo_api.h"
#include "embedding_schemes.h"
#include "image_processing_custom.h"
#include "image_metrics.h"
#include "block_metrics.h"
#include "attacks.h"
#include "gbo.h"
#include "embedding_core.h"
#include "config.h"
#include "jpeg/quantization_tables.h"

#include <stdexcept>

namespace gbo {

static bool s_initialized = false;

// Grayscale view/copy of a caller-supplied image; rejects everything the algorithm cannot take.
static cv::Mat toGray(const cv::Mat& image) {
    const int channels = image.channels();
    if (image.empty() || image.dims != 2 || image.depth() != CV_8U ||
        (channels != 1 && channels != 3 && channels != 4)) {
        throw std::invalid_argument("image must be a non-empty 8-bit image with 1, 3 or 4 channels");
    }
    if (channels == 1) return image;
    cv::Mat gray;
    cv::cvtColor(image, gray, channels == 3 ? cv::COLOR_BGR2GRAY : cv::COLOR_BGRA2GRAY);
    return gray;
}

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

// ---- Execution control -----------------------------------------------------

void setThreads(unsigned threads) { setEmbeddingThreads(threads); }

void setSeed(uint64_t seed) { setEmbeddingSeed(seed); }

void clearSeed() { clearEmbeddingSeed(); }

// ---- Watermark embedding / extraction --------------------------------------

cv::Mat embedWatermark(const cv::Mat& image,
                       const cv::Mat& watermark) {
    if (!s_initialized) {
        throw std::runtime_error("gbo::init() must be called before embedWatermark()");
    }
    const cv::Mat gray = toGray(image);
    if (watermark.type() != CV_8UC1) {
        throw std::invalid_argument("watermark must be a single-channel 8-bit image");
    }
    std::vector<int> wm_bits = convertWatermarkToBinary(watermark);

    initialize_quantization_mats();

    return embedBits(gray, wm_bits);
}

cv::Mat extractWatermark(const cv::Mat& image) {
    if (!s_initialized) {
        throw std::runtime_error("gbo::init() must be called before extractWatermark()");
    }
    const cv::Mat gray = toGray(image);

    std::vector<int> wm_vec = extractVotes(gray);

    // Majority vote over the copies of each bit. A 512x512 image holds 4 copies (0-1 votes -> 0,
    // 3-4 -> 1, 2 -> coin flip); other sizes hold a different number, possibly uneven per bit.
    const size_t blocks = static_cast<size_t>(gray.rows / 8) * (gray.cols / 8);
    for (size_t i = 0; i < WM_SIZE; ++i) {
        const int copies = static_cast<int>(blocks / WM_SIZE + (i < blocks % WM_SIZE ? 1 : 0));
        const int votes = wm_vec[i];
        if (2 * votes == copies) wm_vec[i] = (copies == 0) ? 0 : rand() % 2;
        else wm_vec[i] = (2 * votes > copies) ? 1 : 0;
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
