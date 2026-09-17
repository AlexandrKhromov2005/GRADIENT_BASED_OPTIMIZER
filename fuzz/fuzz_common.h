#ifndef FUZZ_COMMON_H
#define FUZZ_COMMON_H

// Helpers shared by the libFuzzer targets: turning raw fuzzer bytes into values and images.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/utils/logger.hpp>

#include "gbo_api.h"

namespace fuzz {

// Aborts with a message; libFuzzer records the input as a crash.
#define FUZZ_CHECK(cond)                                                          \
    do {                                                                          \
        if (!(cond)) {                                                            \
            std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); \
            std::abort();                                                         \
        }                                                                         \
    } while (0)

class Input {
public:
    Input(const uint8_t* data, size_t size) : p_(data), n_(size) {}

    // Next value of a trivially copyable type; zero bytes once the input is exhausted.
    template <typename T>
    T take() {
        T value{};
        const size_t k = n_ < sizeof(T) ? n_ : sizeof(T);
        if (k > 0) std::memcpy(&value, p_, k);
        p_ += k;
        n_ -= k;
        return value;
    }

    int range(int lo, int hi) { return lo + static_cast<int>(take<uint16_t>() % static_cast<unsigned>(hi - lo + 1)); }

    void fill(uint8_t* out, size_t count) {
        const size_t k = n_ < count ? n_ : count;
        if (k > 0) std::memcpy(out, p_, k);
        if (count > k) std::memset(out + k, 0, count - k);
        p_ += k;
        n_ -= k;
    }

private:
    const uint8_t* p_;
    size_t n_;
};

// An image whose size, type and memory layout come from the fuzzer. `types` lists the
// cv::Mat types to choose from. Sides are in [0, max_side].
inline cv::Mat takeMat(Input& in, int max_side, const int* types, int type_count) {
    const int rows = in.range(0, max_side), cols = in.range(0, max_side);
    const int type = types[in.take<uint8_t>() % type_count];
    const bool as_roi = in.take<uint8_t>() & 1;  // non-continuous view into a larger image
    if (rows == 0 || cols == 0) return cv::Mat(rows, cols, type);

    cv::Mat image;
    if (as_roi) {
        cv::Mat parent(rows + 3, cols + 5, type, cv::Scalar::all(0));
        image = parent(cv::Rect(2, 1, cols, rows));
    } else {
        image.create(rows, cols, type);
    }
    for (int r = 0; r < rows; ++r) in.fill(image.ptr<uint8_t>(r), static_cast<size_t>(cols) * image.elemSize());
    return image;
}

const int kAnyType[] = {CV_8UC1, CV_8UC1, CV_8UC3, CV_8UC4, CV_8UC2, CV_16UC1, CV_32FC1, CV_64FC1, CV_8SC1};
const int kGrayOrColor[] = {CV_8UC1, CV_8UC1, CV_8UC1, CV_8UC3};

// Images the API documents as accepted: 8 bits per channel, gray or colour, not empty.
inline bool isSupportedImage(const cv::Mat& image) {
    return !image.empty() && image.depth() == CV_8U && image.channels() != 2;  // takeMat makes 1-4 channels
}

inline void initLibrary() {
    static const bool ok = [] {
        // The library reports rejected input on std::cerr / the OpenCV log; at thousands of
        // runs per second that only hides the sanitizer reports.
        std::cerr.rdbuf(nullptr);
        std::cout.rdbuf(nullptr);
        cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
        cv::redirectError([](int, const char*, const char*, const char*, int, void*) { return 0; });
        const char* path = std::getenv("GBO_SCHEMES");
        return gbo::init(path ? path : "embedding_schemes.json");
    }();
    if (!ok) {
        std::fprintf(stderr, "cannot load embedding_schemes.json (run from the repo root or set GBO_SCHEMES)\n");
        std::abort();
    }
}

inline void pickScheme(Input& in) {
    static const std::vector<std::string> ids = gbo::availableSchemes();
    if (!ids.empty()) gbo::setScheme(ids[in.take<uint8_t>() % ids.size()]);
}

} // namespace fuzz

#endif // FUZZ_COMMON_H
