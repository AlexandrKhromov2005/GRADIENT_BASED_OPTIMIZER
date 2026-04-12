#ifndef GBO_API_H
#define GBO_API_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#ifdef _WIN32
#   ifdef GBO_BUILDING_LIBRARY
#       define GBO_API __declspec(dllexport)
#   elif defined(GBO_USING_SHARED)
#       define GBO_API __declspec(dllimport)
#   else
#       define GBO_API
#   endif
#else
#   ifdef GBO_BUILDING_LIBRARY
#       define GBO_API __attribute__((visibility("default")))
#   else
#       define GBO_API
#   endif
#endif

namespace gbo {

// ---- Initialization --------------------------------------------------------

/// Load embedding schemes from a JSON file and initialize internal tables.
/// Must be called once before any embed/extract operation.
/// @param schemes_json_path  Path to embedding_schemes.json
/// @return true on success.
GBO_API bool init(const std::string& schemes_json_path = "embedding_schemes.json");

/// Select the active embedding scheme by its identifier
/// (e.g. "scheme1", "scheme2", "scheme3").
/// @return true if the scheme exists and was activated.
GBO_API bool setScheme(const std::string& scheme_id);

/// Return a list of all available scheme identifiers.
GBO_API std::vector<std::string> availableSchemes();

// ---- Watermark embedding / extraction --------------------------------------

/// Embed a binary watermark into a grayscale cover image.
/// Internally runs the GBO optimizer (POP_SIZE x ITERATIONS generations)
/// on every 8x8 block.  A single call performs one full embedding pass.
/// @param image       Grayscale cover image (CV_8U).
/// @param watermark   Binary watermark image (black/white, 32x32 recommended).
/// @return Watermarked grayscale image (CV_8U), same size as input.
GBO_API cv::Mat embedWatermark(const cv::Mat& image,
                               const cv::Mat& watermark);

/// Extract a binary watermark from a (possibly attacked) watermarked image.
/// @param image  Watermarked grayscale image (CV_8U).
/// @return Extracted binary watermark image.
GBO_API cv::Mat extractWatermark(const cv::Mat& image);

// ---- Image quality metrics -------------------------------------------------

/// Mean Squared Error between two images.
GBO_API double computeMSE(const cv::Mat& a, const cv::Mat& b);

/// Peak Signal-to-Noise Ratio (dB).
GBO_API double computePSNR(const cv::Mat& a, const cv::Mat& b);

/// Structural Similarity Index.
GBO_API double computeSSIM(const cv::Mat& a, const cv::Mat& b);

/// Normalized Cross-Correlation.
GBO_API double computeNCC(const cv::Mat& a, const cv::Mat& b);

/// Bit Error Rate between original and extracted watermarks.
GBO_API double computeBER(const cv::Mat& original_wm, const cv::Mat& extracted_wm);

// ---- Attack simulation -----------------------------------------------------

/// Apply JPEG compression attack.
GBO_API cv::Mat attackJPEG(const cv::Mat& image, int quality);

/// Apply brightness increase.
GBO_API cv::Mat attackBrightnessIncrease(const cv::Mat& image, int value);

/// Apply brightness decrease.
GBO_API cv::Mat attackBrightnessDecrease(const cv::Mat& image, int value);

/// Apply contrast increase.
GBO_API cv::Mat attackContrastIncrease(const cv::Mat& image, double alpha);

/// Apply contrast decrease.
GBO_API cv::Mat attackContrastDecrease(const cv::Mat& image, double alpha);

/// Apply salt-and-pepper noise.
GBO_API cv::Mat attackSaltPepper(const cv::Mat& image, double noise_prob);

/// Apply median filtering.
GBO_API cv::Mat attackMedianFilter(const cv::Mat& image, int ksize);

/// Apply Gaussian filtering.
GBO_API cv::Mat attackGaussianFilter(const cv::Mat& image, int ksize);

} // namespace gbo

#endif // GBO_API_H
