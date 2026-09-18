#ifndef GBO_API_H
#define GBO_API_H

#include <opencv2/opencv.hpp>
#include <cstdint>
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
/// Must be called before any embed/extract operation; calling it again replaces the schemes.
/// @param schemes_json_path  Path to embedding_schemes.json (format: see README, section 6)
/// @return true on success; false if the file cannot be read or is rejected by the parser
///         (the reason is printed to std::cerr), in which case nothing changes.
GBO_API bool init(const std::string& schemes_json_path = "embedding_schemes.json");

/// Select the active embedding scheme by its identifier
/// (e.g. "scheme1", "scheme2", "scheme3").
/// @return true if the scheme exists and was activated.
GBO_API bool setScheme(const std::string& scheme_id);

/// Return a list of all available scheme identifiers.
GBO_API std::vector<std::string> availableSchemes();

// ---- Execution control -----------------------------------------------------

/// Number of worker threads for embedding. 0 (default) = all hardware threads;
/// the GBO_THREADS environment variable is honoured when this is left at 0.
/// The embedded result never depends on the thread count.
GBO_API void setThreads(unsigned threads);

/// Make embedding reproducible: calling setSeed(s) and then embedWatermark() with the
/// same image and watermark always gives the same watermarked image (call setSeed again
/// before every embedding you want to reproduce). By default every call uses a fresh
/// random seed. Do not run embeddings concurrently while a seed is fixed.
GBO_API void setSeed(uint64_t seed);

/// Return to non-deterministic seeding (the default).
GBO_API void clearSeed();

// ---- Watermark embedding / extraction --------------------------------------

/// Embed a binary watermark into a grayscale cover image.
/// Internally runs the GBO optimizer (POP_SIZE x ITERATIONS generations)
/// on every 8x8 block, blocks being processed in parallel (see setThreads).
/// A single call performs one full embedding pass.
/// @param image       Cover image, 8 bits per channel, 1, 3 or 4 channels (colour is converted
///                    to grayscale). Pixels of a border narrower than 8 are left unchanged.
/// @param watermark   Binary watermark image (CV_8UC1, black/white, 32x32 recommended).
/// @return Watermarked grayscale image (CV_8UC1), same size as input.
/// @throws std::invalid_argument if the image is empty or of another type, or if the
///         watermark is not CV_8UC1 or holds fewer than 1024 pixels (32x32).
/// @note Do not call setScheme() while an embedding or extraction is in progress.
GBO_API cv::Mat embedWatermark(const cv::Mat& image,
                               const cv::Mat& watermark);

/// Extract a binary watermark from a (possibly attacked) watermarked image.
/// Each bit is a majority vote over its copies (4 in a 512x512 image); a tie is resolved
/// at random.
/// @param image  Watermarked image, 8 bits per channel, 1, 3 or 4 channels.
/// @return Extracted binary watermark image (32x32, CV_8UC1).
/// @throws std::invalid_argument if the image is empty or of another type.
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
