#ifndef BLOCK_KERNELS_H
#define BLOCK_KERNELS_H

// Allocation-free kernels for a single 8x8 block. They compute exactly the same
// quantities as the OpenCV-based reference code (cv::dct / cv::idct / convertTo /
// imencode+imdecode), only without per-call cv::Mat, DCT-plan and codec set-up costs.

#include <cstdint>
#include <opencv2/core.hpp>

namespace kernels {

// Orthonormal 2D DCT-II of an 8x8 block, row-major (same transform as cv::dct).
void dct8x8(const double* in, double* out);

// Orthonormal 2D inverse DCT of an 8x8 block, row-major (same transform as cv::idct).
void idct8x8(const double* in, double* out);

// Same rounding as cv::Mat::convertTo(CV_8U): round half to even, saturate to [0, 255].
void roundToU8(const double* in, uint8_t* out);

// Baseline JPEG compression + decompression of one grayscale 8x8 block.
// Bit-exact emulation of libjpeg(-turbo) with the default integer DCT (what
// cv::imencode / cv::imdecode run for an 8x8 image), without building a JPEG stream.
void jpegRoundTrip(const uint8_t* in, uint8_t* out, int quality);

// True when jpegRoundTrip() reproduces cv::imencode + cv::imdecode on this machine
// (checked once on a set of probe blocks; when false callers must use the codec).
bool jpegEmulationIsExact();

// out[i] = the value cv::Mat::convertTo(-1, alpha, 0) gives for in[i] (contrast change).
void contrastU8(const uint8_t* in, uint8_t* out, double alpha);

// Copy helpers between cv::Mat (CV_8U, 8x8) and flat arrays.
void loadBlock(const cv::Mat& block, uint8_t* out);
void storeBlock(const uint8_t* in, cv::Mat& block);

} // namespace kernels

#endif // BLOCK_KERNELS_H
