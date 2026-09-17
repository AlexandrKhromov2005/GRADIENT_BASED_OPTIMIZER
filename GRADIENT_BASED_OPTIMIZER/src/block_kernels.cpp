#include "block_kernels.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <opencv2/imgcodecs.hpp>

namespace kernels {

// ---- DCT ---------------------------------------------------------------------

namespace {

struct DctTable {
    double c[8][8];  // c[u][x] = a(u) * cos((2x + 1) * u * pi / 16)
    DctTable() {
        const double pi = 3.14159265358979323846;
        for (int u = 0; u < 8; ++u) {
            const double a = (u == 0) ? std::sqrt(1.0 / 8.0) : std::sqrt(2.0 / 8.0);
            for (int x = 0; x < 8; ++x) {
                c[u][x] = a * std::cos((2 * x + 1) * u * pi / 16.0);
            }
        }
    }
};

const DctTable& dctTable() {
    static const DctTable table;
    return table;
}

} // namespace

// Matrix form, straight from the definition. Used to cross-check the fast transforms.
void dct8x8Matrix(const double* in, double* out) {
    const auto& c = dctTable().c;
    double tmp[64];
    for (int r = 0; r < 8; ++r) {          // rows: tmp = in * C^T
        const double* row = in + 8 * r;
        for (int v = 0; v < 8; ++v) {
            double s = 0.0;
            for (int x = 0; x < 8; ++x) s += row[x] * c[v][x];
            tmp[8 * r + v] = s;
        }
    }
    for (int u = 0; u < 8; ++u) {          // columns: out = C * tmp
        for (int v = 0; v < 8; ++v) {
            double s = 0.0;
            for (int r = 0; r < 8; ++r) s += c[u][r] * tmp[8 * r + v];
            out[8 * u + v] = s;
        }
    }
}

void idct8x8Matrix(const double* in, double* out) {
    const auto& c = dctTable().c;
    double tmp[64];
    for (int u = 0; u < 8; ++u) {          // rows: tmp = in * C
        const double* row = in + 8 * u;
        for (int x = 0; x < 8; ++x) {
            double s = 0.0;
            for (int v = 0; v < 8; ++v) s += row[v] * c[v][x];
            tmp[8 * u + x] = s;
        }
    }
    for (int r = 0; r < 8; ++r) {          // columns: out = C^T * tmp
        for (int x = 0; x < 8; ++x) {
            double s = 0.0;
            for (int u = 0; u < 8; ++u) s += c[u][r] * tmp[8 * u + x];
            out[8 * r + x] = s;
        }
    }
}

namespace {

// Loeffler-Ligtenberg-Moschytz factorization (12 multiplications per 1-D transform), in
// double precision. Each 1-D pass is scaled by sqrt(8), so a 2-D transform is scaled by 8.
struct LlmConstants {
    double k0_298631336, k0_390180644, k0_541196100, k0_765366865, k0_899976223, k1_175875602,
           k1_501321110, k1_847759065, k1_961570560, k2_053119869, k2_562915447, k3_072711026;
    LlmConstants() {
        const double pi = 3.14159265358979323846, r2 = std::sqrt(2.0);
        const double c1 = std::cos(pi / 16), c2 = std::cos(2 * pi / 16), c3 = std::cos(3 * pi / 16),
                     c5 = std::cos(5 * pi / 16), c6 = std::cos(6 * pi / 16), c7 = std::cos(7 * pi / 16);
        k0_298631336 = r2 * (-c1 + c3 + c5 - c7);
        k2_053119869 = r2 * (c1 + c3 - c5 + c7);
        k3_072711026 = r2 * (c1 + c3 + c5 - c7);
        k1_501321110 = r2 * (c1 + c3 - c5 - c7);
        k0_899976223 = r2 * (c3 - c7);
        k2_562915447 = r2 * (c1 + c3);
        k1_961570560 = r2 * (c3 + c5);
        k0_390180644 = r2 * (c3 - c5);
        k1_175875602 = r2 * c3;
        k0_541196100 = r2 * c6;
        k0_765366865 = r2 * (c2 - c6);
        k1_847759065 = r2 * (c2 + c6);
    }
};

const LlmConstants& llm() {
    static const LlmConstants constants;
    return constants;
}

inline void llmForward(const LlmConstants& k, const double* in, double* out, int stride) {
    const double tmp0 = in[0] + in[7 * stride], tmp7 = in[0] - in[7 * stride];
    const double tmp1 = in[stride] + in[6 * stride], tmp6 = in[stride] - in[6 * stride];
    const double tmp2 = in[2 * stride] + in[5 * stride], tmp5 = in[2 * stride] - in[5 * stride];
    const double tmp3 = in[3 * stride] + in[4 * stride], tmp4 = in[3 * stride] - in[4 * stride];

    const double tmp10 = tmp0 + tmp3, tmp13 = tmp0 - tmp3;
    const double tmp11 = tmp1 + tmp2, tmp12 = tmp1 - tmp2;

    out[0] = tmp10 + tmp11;
    out[4 * stride] = tmp10 - tmp11;
    const double e = (tmp12 + tmp13) * k.k0_541196100;
    out[2 * stride] = e + tmp13 * k.k0_765366865;
    out[6 * stride] = e - tmp12 * k.k1_847759065;

    const double z5 = (tmp4 + tmp5 + tmp6 + tmp7) * k.k1_175875602;
    const double z1 = (tmp4 + tmp7) * -k.k0_899976223;
    const double z2 = (tmp5 + tmp6) * -k.k2_562915447;
    const double z3 = (tmp4 + tmp6) * -k.k1_961570560 + z5;
    const double z4 = (tmp5 + tmp7) * -k.k0_390180644 + z5;

    out[7 * stride] = tmp4 * k.k0_298631336 + z1 + z3;
    out[5 * stride] = tmp5 * k.k2_053119869 + z2 + z4;
    out[3 * stride] = tmp6 * k.k3_072711026 + z2 + z3;
    out[stride] = tmp7 * k.k1_501321110 + z1 + z4;
}

inline void llmInverse(const LlmConstants& k, const double* in, double* out, int stride) {
    const double e = (in[2 * stride] + in[6 * stride]) * k.k0_541196100;
    const double even2 = e - in[6 * stride] * k.k1_847759065;
    const double even3 = e + in[2 * stride] * k.k0_765366865;
    const double even0 = in[0] + in[4 * stride], even1 = in[0] - in[4 * stride];

    const double tmp10 = even0 + even3, tmp13 = even0 - even3;
    const double tmp11 = even1 + even2, tmp12 = even1 - even2;

    const double i7 = in[7 * stride], i5 = in[5 * stride], i3 = in[3 * stride], i1 = in[stride];
    const double z5 = (i7 + i5 + i3 + i1) * k.k1_175875602;
    const double z1 = (i7 + i1) * -k.k0_899976223;
    const double z2 = (i5 + i3) * -k.k2_562915447;
    const double z3 = (i7 + i3) * -k.k1_961570560 + z5;
    const double z4 = (i5 + i1) * -k.k0_390180644 + z5;

    const double odd0 = i7 * k.k0_298631336 + z1 + z3;
    const double odd1 = i5 * k.k2_053119869 + z2 + z4;
    const double odd2 = i3 * k.k3_072711026 + z2 + z3;
    const double odd3 = i1 * k.k1_501321110 + z1 + z4;

    out[0] = tmp10 + odd3;
    out[7 * stride] = tmp10 - odd3;
    out[stride] = tmp11 + odd2;
    out[6 * stride] = tmp11 - odd2;
    out[2 * stride] = tmp12 + odd1;
    out[5 * stride] = tmp12 - odd1;
    out[3 * stride] = tmp13 + odd0;
    out[4 * stride] = tmp13 - odd0;
}

} // namespace

void dct8x8(const double* in, double* out) {
    const LlmConstants& k = llm();
    double tmp[64];
    for (int r = 0; r < 8; ++r) llmForward(k, in + 8 * r, tmp + 8 * r, 1);
    for (int c = 0; c < 8; ++c) llmForward(k, tmp + c, out + c, 8);
    for (int i = 0; i < 64; ++i) out[i] *= 0.125;
}

void idct8x8(const double* in, double* out) {
    const LlmConstants& k = llm();
    double tmp[64];
    for (int c = 0; c < 8; ++c) llmInverse(k, in + c, tmp + c, 8);
    for (int r = 0; r < 8; ++r) llmInverse(k, tmp + 8 * r, out + 8 * r, 1);
    for (int i = 0; i < 64; ++i) out[i] *= 0.125;
}

bool roundToU8(const double* in, uint8_t* out) {
    bool unambiguous = true;
    for (int i = 0; i < 64; ++i) {
        const int r = cvRound(in[i]);  // round half to even, exactly what convertTo(CV_8U) uses
        if (std::fabs(in[i] - r) > 0.5 - 1e-9) unambiguous = false;
        out[i] = static_cast<uint8_t>(r < 0 ? 0 : (r > 255 ? 255 : r));
    }
    return unambiguous;
}

// ---- JPEG --------------------------------------------------------------------

namespace {

// Annex K luminance table, natural (row-major) order.
const int kStdLuminance[64] = {
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99
};

struct QuantTables {
    int q[101][64];
    QuantTables() {
        for (int quality = 1; quality <= 100; ++quality) {
            // jpeg_quality_scaling + jpeg_add_quant_table(force_baseline = TRUE)
            const int scale = (quality < 50) ? 5000 / quality : 200 - quality * 2;
            for (int i = 0; i < 64; ++i) {
                long v = (static_cast<long>(kStdLuminance[i]) * scale + 50L) / 100L;
                if (v <= 0L) v = 1L;
                if (v > 255L) v = 255L;
                q[quality][i] = static_cast<int>(v);
            }
        }
        std::memset(q[0], 0, sizeof(q[0]));
    }
};

const QuantTables& quantTables() {
    static const QuantTables tables;
    return tables;
}

// jfdctint.c / jidctint.c ("islow") constants, CONST_BITS = 13, PASS1_BITS = 2
constexpr int CONST_BITS = 13;
constexpr int PASS1_BITS = 2;
constexpr int32_t FIX_0_298631336 = 2446;
constexpr int32_t FIX_0_390180644 = 3196;
constexpr int32_t FIX_0_541196100 = 4433;
constexpr int32_t FIX_0_765366865 = 6270;
constexpr int32_t FIX_0_899976223 = 7373;
constexpr int32_t FIX_1_175875602 = 9633;
constexpr int32_t FIX_1_501321110 = 12299;
constexpr int32_t FIX_1_847759065 = 15137;
constexpr int32_t FIX_1_961570560 = 16069;
constexpr int32_t FIX_2_053119869 = 16819;
constexpr int32_t FIX_2_562915447 = 20995;
constexpr int32_t FIX_3_072711026 = 25172;

// Relies on arithmetic right shift of negative values (true for GCC, Clang, MSVC; C++20 rule).
inline int32_t descale(int32_t x, int n) { return (x + (int32_t(1) << (n - 1))) >> n; }

// One 1-D pass of jpeg_fdct_islow over 8 values spaced by `stride`.
inline void fdctPass(int32_t* d, int stride, bool first) {
    const int32_t tmp0 = d[0] + d[7 * stride], tmp7 = d[0] - d[7 * stride];
    const int32_t tmp1 = d[stride] + d[6 * stride], tmp6 = d[stride] - d[6 * stride];
    const int32_t tmp2 = d[2 * stride] + d[5 * stride], tmp5 = d[2 * stride] - d[5 * stride];
    const int32_t tmp3 = d[3 * stride] + d[4 * stride], tmp4 = d[3 * stride] - d[4 * stride];

    const int32_t tmp10 = tmp0 + tmp3, tmp13 = tmp0 - tmp3;
    const int32_t tmp11 = tmp1 + tmp2, tmp12 = tmp1 - tmp2;

    const int shift = first ? CONST_BITS - PASS1_BITS : CONST_BITS + PASS1_BITS;
    if (first) {
        d[0] = (tmp10 + tmp11) * (1 << PASS1_BITS);
        d[4 * stride] = (tmp10 - tmp11) * (1 << PASS1_BITS);
    } else {
        d[0] = descale(tmp10 + tmp11, PASS1_BITS);
        d[4 * stride] = descale(tmp10 - tmp11, PASS1_BITS);
    }

    int32_t z1 = (tmp12 + tmp13) * FIX_0_541196100;
    d[2 * stride] = descale(z1 + tmp13 * FIX_0_765366865, shift);
    d[6 * stride] = descale(z1 + tmp12 * (-FIX_1_847759065), shift);

    z1 = tmp4 + tmp7;
    int32_t z2 = tmp5 + tmp6;
    int32_t z3 = tmp4 + tmp6;
    int32_t z4 = tmp5 + tmp7;
    const int32_t z5 = (z3 + z4) * FIX_1_175875602;

    const int32_t t4 = tmp4 * FIX_0_298631336;
    const int32_t t5 = tmp5 * FIX_2_053119869;
    const int32_t t6 = tmp6 * FIX_3_072711026;
    const int32_t t7 = tmp7 * FIX_1_501321110;
    z1 *= -FIX_0_899976223;
    z2 *= -FIX_2_562915447;
    z3 *= -FIX_1_961570560;
    z4 *= -FIX_0_390180644;
    z3 += z5;
    z4 += z5;

    d[7 * stride] = descale(t4 + z1 + z3, shift);
    d[5 * stride] = descale(t5 + z2 + z4, shift);
    d[3 * stride] = descale(t6 + z2 + z3, shift);
    d[stride] = descale(t7 + z1 + z4, shift);
}

// One 1-D pass of jpeg_idct_islow; reads in[k * stride], writes out[k * stride].
inline void idctPass(const int32_t* in, int32_t* out, int stride, int shift) {
    int32_t z2 = in[2 * stride], z3 = in[6 * stride];
    int32_t z1 = (z2 + z3) * FIX_0_541196100;
    int32_t tmp2 = z1 + z3 * (-FIX_1_847759065);
    int32_t tmp3 = z1 + z2 * FIX_0_765366865;

    z2 = in[0];
    z3 = in[4 * stride];
    int32_t tmp0 = (z2 + z3) * (int32_t(1) << CONST_BITS);
    int32_t tmp1 = (z2 - z3) * (int32_t(1) << CONST_BITS);

    const int32_t tmp10 = tmp0 + tmp3, tmp13 = tmp0 - tmp3;
    const int32_t tmp11 = tmp1 + tmp2, tmp12 = tmp1 - tmp2;

    tmp0 = in[7 * stride];
    tmp1 = in[5 * stride];
    tmp2 = in[3 * stride];
    tmp3 = in[stride];

    z1 = tmp0 + tmp3;
    z2 = tmp1 + tmp2;
    z3 = tmp0 + tmp2;
    int32_t z4 = tmp1 + tmp3;
    const int32_t z5 = (z3 + z4) * FIX_1_175875602;

    tmp0 *= FIX_0_298631336;
    tmp1 *= FIX_2_053119869;
    tmp2 *= FIX_3_072711026;
    tmp3 *= FIX_1_501321110;
    z1 *= -FIX_0_899976223;
    z2 *= -FIX_2_562915447;
    z3 *= -FIX_1_961570560;
    z4 *= -FIX_0_390180644;
    z3 += z5;
    z4 += z5;

    tmp0 += z1 + z3;
    tmp1 += z2 + z4;
    tmp2 += z2 + z3;
    tmp3 += z1 + z4;

    out[0] = descale(tmp10 + tmp3, shift);
    out[7 * stride] = descale(tmp10 - tmp3, shift);
    out[stride] = descale(tmp11 + tmp2, shift);
    out[6 * stride] = descale(tmp11 - tmp2, shift);
    out[2 * stride] = descale(tmp12 + tmp1, shift);
    out[5 * stride] = descale(tmp12 - tmp1, shift);
    out[3 * stride] = descale(tmp13 + tmp0, shift);
    out[4 * stride] = descale(tmp13 - tmp0, shift);
}

void jpegRoundTripCodec(const uint8_t* in, uint8_t* out, int quality) {
    cv::Mat block(8, 8, CV_8U);
    std::memcpy(block.data, in, 64);
    std::vector<uchar> encoded;
    if (!cv::imencode(".jpg", block, encoded, {cv::IMWRITE_JPEG_QUALITY, quality})) {
        throw std::runtime_error("JPEG encoding failed (OpenCV built without JPEG support?)");
    }
    cv::Mat decoded = cv::imdecode(encoded, cv::IMREAD_GRAYSCALE);
    if (decoded.rows != 8 || decoded.cols != 8 || decoded.type() != CV_8UC1) {
        throw std::runtime_error("JPEG decoding failed (OpenCV built without JPEG support?)");
    }
    for (int r = 0; r < 8; ++r) std::memcpy(out + 8 * r, decoded.ptr<uchar>(r), 8);
}

void jpegRoundTripEmulated(const uint8_t* in, uint8_t* out, int quality) {
    const int* q = quantTables().q[quality];
    int32_t d[64], ws[64], px[64];

    // Encoder: level shift, forward DCT (output scaled by 8), quantization.
    for (int i = 0; i < 64; ++i) d[i] = static_cast<int32_t>(in[i]) - 128;
    for (int r = 0; r < 8; ++r) fdctPass(d + 8 * r, 1, true);
    for (int c = 0; c < 8; ++c) fdctPass(d + c, 8, false);
    for (int i = 0; i < 64; ++i) {
        const int32_t qval = static_cast<int32_t>(q[i]) << 3;
        int32_t t = d[i];
        if (t < 0) {
            t = -t;
            t += qval >> 1;
            t = (t >= qval) ? t / qval : 0;
            t = -t;
        } else {
            t += qval >> 1;
            t = (t >= qval) ? t / qval : 0;
        }
        d[i] = t * q[i];  // decoder: dequantization
    }

    // Decoder: inverse DCT, level shift, range limit.
    for (int c = 0; c < 8; ++c) idctPass(d + c, ws + c, 8, CONST_BITS - PASS1_BITS);
    for (int r = 0; r < 8; ++r) idctPass(ws + 8 * r, px + 8 * r, 1, CONST_BITS + PASS1_BITS + 3);
    // Plain clamping equals libjpeg's range_limit table for px in [-512, 511]. Equality with the
    // codec on arbitrary blocks and qualities is what the start-up probe, --selftest and
    // fuzz_kernels check.
    for (int i = 0; i < 64; ++i) {
        const int32_t v = px[i] + 128;
        out[i] = static_cast<uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v));
    }
}

bool probeJpegEmulation() {
    // Deterministic probe blocks: smooth, textured and saturated content.
    uint32_t state = 0x12345678u;
    auto next = [&state]() { state = state * 1664525u + 1013904223u; return state >> 8; };
    for (int n = 0; n < 96; ++n) {
        uint8_t block[64], a[64], b[64];
        const int base = static_cast<int>(next() % 256), amp = 1 << (n % 8);
        for (int i = 0; i < 64; ++i) {
            int v = base + static_cast<int>(next() % (2 * amp + 1)) - amp + ((n & 1) ? (i % 8) * 3 : 0);
            block[i] = static_cast<uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v));
        }
        for (int quality : {1, 10, 30, 50, 70, 80, 90, 95, 100}) {
            jpegRoundTripEmulated(block, a, quality);
            jpegRoundTripCodec(block, b, quality);
            if (std::memcmp(a, b, 64) != 0) return false;
        }
    }
    return true;
}

} // namespace

bool jpegEmulationIsExact() {
    static const bool exact = [] {
        const bool ok = probeJpegEmulation();
        if (!ok) {
            std::cerr << "gbo: JPEG emulation differs from this OpenCV/libjpeg build; "
                         "using the codec for the JPEG attack (much slower)" << std::endl;
        }
        return ok;
    }();
    return exact;
}

void jpegRoundTrip(const uint8_t* in, uint8_t* out, int quality) {
    if (quality < 1) quality = 1;
    if (quality > 100) quality = 100;
    if (jpegEmulationIsExact()) {
        jpegRoundTripEmulated(in, out, quality);
    } else {
        jpegRoundTripCodec(in, out, quality);
    }
}

// ---- Contrast ----------------------------------------------------------------

void contrastU8(const uint8_t* in, uint8_t* out, double alpha) {
    struct Lut {
        double alpha = 0.0;
        bool valid = false;
        uint8_t map[256];
    };
    thread_local Lut lut;
    if (!lut.valid || lut.alpha != alpha) {
        cv::Mat ramp(1, 256, CV_8U), mapped;
        for (int i = 0; i < 256; ++i) ramp.at<uchar>(0, i) = static_cast<uchar>(i);
        ramp.convertTo(mapped, -1, alpha, 0);
        std::memcpy(lut.map, mapped.data, 256);
        lut.alpha = alpha;
        lut.valid = true;
    }
    for (int i = 0; i < 64; ++i) out[i] = lut.map[in[i]];
}

// ---- cv::Mat helpers ---------------------------------------------------------

void loadBlock(const cv::Mat& block, uint8_t* out) {
    for (int r = 0; r < 8; ++r) std::memcpy(out + 8 * r, block.ptr<uchar>(r), 8);
}

void storeBlock(const uint8_t* in, cv::Mat& block) {
    for (int r = 0; r < 8; ++r) std::memcpy(block.ptr<uchar>(r), in + 8 * r, 8);
}

} // namespace kernels
