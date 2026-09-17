// Extraction from an arbitrary image: any size, type and memory layout.
// An unsupported image may be rejected with an exception, a supported one may not. Crashes,
// sanitizer reports and hangs are failures either way.

#include "fuzz_common.h"
#include "embedding_core.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    fuzz::initLibrary();
    fuzz::Input in(data, size);
    fuzz::pickScheme(in);
    const int attack = in.take<uint8_t>() % 4;
    const bool equal_is_one = in.take<uint8_t>() & 1;
    const size_t max_blocks = in.take<uint16_t>();
    const int tile_rows = in.range(1, 8), tile_cols = in.range(1, 8);
    cv::Mat image = fuzz::takeMat(in, 96, fuzz::kAnyType, sizeof(fuzz::kAnyType) / sizeof(int));
    // Tiling reaches the sizes that hold several watermark copies (up to 768x768) without
    // needing that many input bytes.
    if (!image.empty()) image = cv::repeat(image, tile_rows, tile_cols);

    try {
        const cv::Mat wm = gbo::extractWatermark(image);
        FUZZ_CHECK(fuzz::isSupportedImage(image));
        FUZZ_CHECK(wm.rows == 32 && wm.cols == 32 && wm.type() == CV_8UC1);
        for (int r = 0; r < 32; ++r)
            for (int c = 0; c < 32; ++c) FUZZ_CHECK(wm.at<uchar>(r, c) == 0 || wm.at<uchar>(r, c) == 255);
    } catch (const std::exception& e) {
        if (fuzz::isSupportedImage(image)) std::fprintf(stderr, "valid input rejected: %s\n", e.what());
        FUZZ_CHECK(!fuzz::isSupportedImage(image));
    }

    try {
        const std::vector<int> bits = extractBlockBits(image, max_blocks, equal_is_one);
        FUZZ_CHECK(bits.size() <= max_blocks);
        FUZZ_CHECK(bits.size() <= static_cast<size_t>(image.rows / 8) * (image.cols / 8));
        for (int bit : bits) FUZZ_CHECK(bit == 0 || bit == 1);
        // The whole-image path and the single-block path must agree.
        if (!bits.empty()) {
            FUZZ_CHECK(bits[0] == extractBitFromBlock(image(cv::Rect(0, 0, 8, 8)), equal_is_one));
        }
    } catch (const std::exception&) {
        FUZZ_CHECK(image.type() != CV_8UC1);
    }

    try {
        const std::vector<int> votes = extractVotesQuadrants(image, static_cast<AttackType>(attack));
        FUZZ_CHECK(votes.size() == 1024);
        for (int v : votes) FUZZ_CHECK(v >= 0 && v <= 4);
    } catch (const std::exception&) {
        FUZZ_CHECK(image.type() != CV_8UC1);
    }
    return 0;
}
