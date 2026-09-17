// Differential test of the 8x8 kernels against the OpenCV code they replace.

#include "fuzz_common.h"
#include "block_kernels.h"

#include <cmath>
#include <opencv2/imgcodecs.hpp>

namespace {

void codecRoundTrip(const uint8_t* in, uint8_t* out, int quality) {
    cv::Mat block(8, 8, CV_8UC1);
    kernels::storeBlock(in, block);
    std::vector<uchar> encoded;
    FUZZ_CHECK(cv::imencode(".jpg", block, encoded, {cv::IMWRITE_JPEG_QUALITY, quality}));
    const cv::Mat decoded = cv::imdecode(encoded, cv::IMREAD_GRAYSCALE);
    FUZZ_CHECK(decoded.rows == 8 && decoded.cols == 8 && decoded.type() == CV_8UC1);
    kernels::loadBlock(decoded, out);
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    fuzz::initLibrary();
    FUZZ_CHECK(kernels::jpegEmulationIsExact());
    fuzz::Input in(data, size);
    const int quality = in.range(-20, 130);
    const double alpha = in.take<double>();
    const bool raw_doubles = in.take<uint8_t>() & 1;
    uint8_t pixels[64];
    in.fill(pixels, 64);

    // JPEG: emulation == codec for every block and quality. jpegRoundTrip clamps the quality
    // to [1, 100]; the reference gets the clamped value.
    uint8_t emulated[64], reference[64];
    kernels::jpegRoundTrip(pixels, emulated, quality);
    codecRoundTrip(pixels, reference, std::min(100, std::max(1, quality)));
    FUZZ_CHECK(std::memcmp(emulated, reference, 64) == 0);

    // Contrast: LUT == convertTo for any finite gain.
    if (std::isfinite(alpha)) {
        uint8_t mapped[64];
        kernels::contrastU8(pixels, mapped, alpha);
        cv::Mat source(8, 8, CV_8UC1, pixels), expected;
        source.convertTo(expected, -1, alpha, 0);
        FUZZ_CHECK(std::memcmp(mapped, expected.data, 64) == 0);
    }

    // DCT / IDCT against the definition, and the round trip.
    double values[64], fast[64], slow[64], back[64];
    for (int i = 0; i < 64; ++i) values[i] = pixels[i];
    kernels::dct8x8(values, fast);
    kernels::dct8x8Matrix(values, slow);
    for (int i = 0; i < 64; ++i) FUZZ_CHECK(std::fabs(fast[i] - slow[i]) < 1e-9);
    kernels::idct8x8(fast, back);
    for (int i = 0; i < 64; ++i) FUZZ_CHECK(std::fabs(back[i] - values[i]) < 1e-9);
    kernels::idct8x8Matrix(fast, slow);
    for (int i = 0; i < 64; ++i) FUZZ_CHECK(std::fabs(back[i] - slow[i]) < 1e-9);

    // Rounding == convertTo(CV_8U). Values come either from the range the algorithm produces
    // or straight from the fuzzer (huge, NaN, infinite).
    double real[64];
    for (int i = 0; i < 64; ++i) {
        real[i] = raw_doubles ? in.take<double>() : in.take<int16_t>() / 64.0;
    }
    uint8_t rounded[64];
    kernels::roundToU8(real, rounded);
    cv::Mat expected;
    cv::Mat(8, 8, CV_64F, real).convertTo(expected, CV_8U);
    FUZZ_CHECK(std::memcmp(rounded, expected.data, 64) == 0);
    return 0;
}
