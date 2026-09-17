// Metrics and attack simulation on arbitrary image pairs and parameters.

#include "fuzz_common.h"

#include <cmath>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    fuzz::initLibrary();
    fuzz::Input in(data, size);
    const int which = in.take<uint8_t>() % 13;
    const int int_param = in.take<int32_t>();
    const double real_param = in.take<double>();
    const bool same_shape = in.take<uint8_t>() & 1;
    const cv::Mat a = fuzz::takeMat(in, 48, fuzz::kAnyType, sizeof(fuzz::kAnyType) / sizeof(int));
    cv::Mat b;
    if (same_shape && !a.empty()) {
        b.create(a.size(), a.type());
        for (int r = 0; r < b.rows; ++r) in.fill(b.ptr<uint8_t>(r), static_cast<size_t>(b.cols) * b.elemSize());
    } else {
        b = fuzz::takeMat(in, 48, fuzz::kAnyType, sizeof(fuzz::kAnyType) / sizeof(int));
    }

    // Calls that have no reason to fail: 8-bit gray input, matching shapes, valid kernel size.
    const bool gray = !a.empty() && a.type() == CV_8UC1;
    const int ksize = int_param % 64;
    bool must_succeed = gray;
    if (which <= 4) must_succeed = gray && b.type() == CV_8UC1 && a.size() == b.size();
    if (which >= 8 && which <= 10) must_succeed = gray && std::isfinite(real_param);
    if (which == 11) must_succeed = gray && (ksize == 3 || ksize == 5);
    if (which == 12) must_succeed = gray && ksize > 0 && ksize % 2 == 1;

    try {
        cv::Mat out;
        switch (which) {
        case 0: gbo::computeMSE(a, b); break;
        case 1: gbo::computePSNR(a, b); break;
        case 2: gbo::computeSSIM(a, b); break;
        case 3: gbo::computeNCC(a, b); break;
        case 4: {
            const double ber = gbo::computeBER(a, b);
            FUZZ_CHECK(ber == -1 || std::isnan(ber) || (ber >= 0.0 && ber <= 1.0));
            break;
        }
        case 5: out = gbo::attackJPEG(a, int_param); break;
        case 6: out = gbo::attackBrightnessIncrease(a, int_param); break;
        case 7: out = gbo::attackBrightnessDecrease(a, int_param); break;
        case 8: out = gbo::attackContrastIncrease(a, real_param); break;
        case 9: out = gbo::attackContrastDecrease(a, real_param); break;
        case 10: out = gbo::attackSaltPepper(a, real_param); break;
        case 11: out = gbo::attackMedianFilter(a, ksize); break;
        case 12: out = gbo::attackGaussianFilter(a, ksize); break;
        }
        if (which >= 5 && must_succeed) FUZZ_CHECK(out.size() == a.size() && out.type() == a.type());
    } catch (const std::exception& e) {
        if (must_succeed) std::fprintf(stderr, "valid call %d rejected: %s\n", which, e.what());
        FUZZ_CHECK(!must_succeed);
    }
    return 0;
}
