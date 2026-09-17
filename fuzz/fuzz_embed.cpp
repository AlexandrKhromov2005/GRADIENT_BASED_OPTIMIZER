// Embedding into a small arbitrary image with an arbitrary watermark, base and quadrant
// variants. Checks the documented contract: same size out, CV_8UC1, reproducible under a
// fixed seed regardless of the thread count, and the embedded bits can be read back.

#include "fuzz_common.h"
#include "embedding_core.h"
#include "image_processing_custom.h"

namespace {

bool sameImage(const cv::Mat& a, const cv::Mat& b) {
    if (a.empty() || b.empty()) return a.empty() && b.empty();  // clone() of a 0xN image is 0x0
    return a.size() == b.size() && a.type() == b.type() && cv::norm(a, b, cv::NORM_INF) == 0;
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    fuzz::initLibrary();
    fuzz::Input in(data, size);
    fuzz::pickScheme(in);
    const uint64_t seed = in.take<uint64_t>();
    const unsigned threads = 1 + in.take<uint8_t>() % 4;
    const bool quadrants = in.take<uint8_t>() & 1;
    const cv::Mat watermark = fuzz::takeMat(in, 34, fuzz::kGrayOrColor, 4);
    // Up to 5x5 blocks (base) or 4x4 one-block quadrants; GBO costs ~1200 objective
    // evaluations per block, so the image has to stay small.
    const cv::Mat image = fuzz::takeMat(in, quadrants ? 35 : 42, fuzz::kAnyType, sizeof(fuzz::kAnyType) / sizeof(int));

    const bool watermark_ok = watermark.type() == CV_8UC1 && watermark.total() >= 1024;
    // embedBitsQuadrants is an internal entry point: gray input only, may be empty.
    const bool image_ok = quadrants ? image.type() == CV_8UC1 : fuzz::isSupportedImage(image);
    const cv::Mat image_before = image.clone();
    try {
        cv::Mat first, second;
        if (quadrants) {
            const std::vector<int> bits = convertWatermarkToBinary(watermark);
            gbo::setThreads(threads);
            gbo::setSeed(seed);
            first = embedBitsQuadrants(image, bits);
            gbo::setThreads(1);
            gbo::setSeed(seed);
            second = embedBitsQuadrants(image, bits);
        } else {
            gbo::setThreads(threads);
            gbo::setSeed(seed);
            first = gbo::embedWatermark(image, watermark);
            gbo::setThreads(1);
            gbo::setSeed(seed);
            second = gbo::embedWatermark(image, watermark);
        }
        FUZZ_CHECK(watermark_ok && image_ok);
        FUZZ_CHECK(sameImage(first, second));
        FUZZ_CHECK(first.empty() == image.empty());
        if (!first.empty()) FUZZ_CHECK(first.rows == image.rows && first.cols == image.cols);
        if (!first.empty()) {
            FUZZ_CHECK(first.type() == CV_8UC1);
            gbo::extractWatermark(first);
            // Pixels outside the grid of full blocks are the cover pixels.
            if (!quadrants && image.channels() == 1) {
                const int grid_rows = image.rows / 8 * 8, grid_cols = image.cols / 8 * 8;
                for (int r = 0; r < image.rows; ++r)
                    for (int c = (r < grid_rows ? grid_cols : 0); c < image.cols; ++c)
                        FUZZ_CHECK(first.at<uchar>(r, c) == image.at<uchar>(r, c));
            }
        }
    } catch (const std::exception& e) {
        if (watermark_ok && image_ok) std::fprintf(stderr, "valid input rejected: %s\n", e.what());
        FUZZ_CHECK(!(watermark_ok && image_ok));
    }
    FUZZ_CHECK(sameImage(image, image_before));  // the input is never written to
    gbo::clearSeed();
    return 0;
}
