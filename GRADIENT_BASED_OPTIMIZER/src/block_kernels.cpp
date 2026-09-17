#include "block_kernels.h"

#include <cmath>
#include <cstring>
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

void dct8x8(const double* in, double* out) {
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

void idct8x8(const double* in, double* out) {
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

void roundToU8(const double* in, uint8_t* out) {
    for (int i = 0; i < 64; ++i) {
        const double r = std::nearbyint(in[i]);  // default FP mode: half to even, as cvRound
        out[i] = static_cast<uint8_t>(r < 0.0 ? 0.0 : (r > 255.0 ? 255.0 : r));
    }
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
constexpr int64_t FIX_0_298631336 = 2446;
constexpr int64_t FIX_0_390180644 = 3196;
constexpr int64_t FIX_0_541196100 = 4433;
constexpr int64_t FIX_0_765366865 = 6270;
constexpr int64_t FIX_0_899976223 = 7373;
constexpr int64_t FIX_1_175875602 = 9633;
constexpr int64_t FIX_1_501321110 = 12299;
constexpr int64_t FIX_1_847759065 = 15137;
constexpr int64_t FIX_1_961570560 = 16069;
constexpr int64_t FIX_2_053119869 = 16819;
constexpr int64_t FIX_2_562915447 = 20995;
constexpr int64_t FIX_3_072711026 = 25172;

inline int64_t descale(int64_t x, int n) { return (x + (int64_t(1) << (n - 1))) >> n; }

// One 1-D pass of jpeg_fdct_islow over 8 values spaced by `stride`.
inline void fdctPass(int64_t* d, int stride, bool first) {
    const int64_t tmp0 = d[0] + d[7 * stride], tmp7 = d[0] - d[7 * stride];
    const int64_t tmp1 = d[stride] + d[6 * stride], tmp6 = d[stride] - d[6 * stride];
    const int64_t tmp2 = d[2 * stride] + d[5 * stride], tmp5 = d[2 * stride] - d[5 * stride];
    const int64_t tmp3 = d[3 * stride] + d[4 * stride], tmp4 = d[3 * stride] - d[4 * stride];

    const int64_t tmp10 = tmp0 + tmp3, tmp13 = tmp0 - tmp3;
    const int64_t tmp11 = tmp1 + tmp2, tmp12 = tmp1 - tmp2;

    const int shift = first ? CONST_BITS - PASS1_BITS : CONST_BITS + PASS1_BITS;
    if (first) {
        d[0] = (tmp10 + tmp11) * (1 << PASS1_BITS);
        d[4 * stride] = (tmp10 - tmp11) * (1 << PASS1_BITS);
    } else {
        d[0] = descale(tmp10 + tmp11, PASS1_BITS);
        d[4 * stride] = descale(tmp10 - tmp11, PASS1_BITS);
    }

    int64_t z1 = (tmp12 + tmp13) * FIX_0_541196100;
    d[2 * stride] = descale(z1 + tmp13 * FIX_0_765366865, shift);
    d[6 * stride] = descale(z1 + tmp12 * (-FIX_1_847759065), shift);

    z1 = tmp4 + tmp7;
    int64_t z2 = tmp5 + tmp6;
    int64_t z3 = tmp4 + tmp6;
    int64_t z4 = tmp5 + tmp7;
    const int64_t z5 = (z3 + z4) * FIX_1_175875602;

    const int64_t t4 = tmp4 * FIX_0_298631336;
    const int64_t t5 = tmp5 * FIX_2_053119869;
    const int64_t t6 = tmp6 * FIX_3_072711026;
    const int64_t t7 = tmp7 * FIX_1_501321110;
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
inline void idctPass(const int64_t* in, int64_t* out, int stride, int shift) {
    int64_t z2 = in[2 * stride], z3 = in[6 * stride];
    int64_t z1 = (z2 + z3) * FIX_0_541196100;
    int64_t tmp2 = z1 + z3 * (-FIX_1_847759065);
    int64_t tmp3 = z1 + z2 * FIX_0_765366865;

    z2 = in[0];
    z3 = in[4 * stride];
    int64_t tmp0 = (z2 + z3) * (int64_t(1) << CONST_BITS);
    int64_t tmp1 = (z2 - z3) * (int64_t(1) << CONST_BITS);

    const int64_t tmp10 = tmp0 + tmp3, tmp13 = tmp0 - tmp3;
    const int64_t tmp11 = tmp1 + tmp2, tmp12 = tmp1 - tmp2;

    tmp0 = in[7 * stride];
    tmp1 = in[5 * stride];
    tmp2 = in[3 * stride];
    tmp3 = in[stride];

    z1 = tmp0 + tmp3;
    z2 = tmp1 + tmp2;
    z3 = tmp0 + tmp2;
    int64_t z4 = tmp1 + tmp3;
    const int64_t z5 = (z3 + z4) * FIX_1_175875602;

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
    cv::imencode(".jpg", block, encoded, {cv::IMWRITE_JPEG_QUALITY, quality});
    cv::Mat decoded = cv::imdecode(encoded, cv::IMREAD_GRAYSCALE);
    for (int r = 0; r < 8; ++r) std::memcpy(out + 8 * r, decoded.ptr<uchar>(r), 8);
}

void jpegRoundTripEmulated(const uint8_t* in, uint8_t* out, int quality) {
    const int* q = quantTables().q[quality];
    int64_t d[64], ws[64], px[64];

    // Encoder: level shift, forward DCT (output scaled by 8), quantization.
    for (int i = 0; i < 64; ++i) d[i] = static_cast<int64_t>(in[i]) - 128;
    for (int r = 0; r < 8; ++r) fdctPass(d + 8 * r, 1, true);
    for (int c = 0; c < 8; ++c) fdctPass(d + c, 8, false);
    for (int i = 0; i < 64; ++i) {
        const int64_t qval = static_cast<int64_t>(q[i]) << 3;
        int64_t t = d[i];
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
    for (int i = 0; i < 64; ++i) {
        const int64_t v = px[i] + 128;
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
        for (int quality : {50, 70, 80, 90}) {
            jpegRoundTripEmulated(block, a, quality);
            jpegRoundTripCodec(block, b, quality);
            if (std::memcmp(a, b, 64) != 0) return false;
        }
    }
    return true;
}

} // namespace

bool jpegEmulationIsExact() {
    static const bool exact = probeJpegEmulation();
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
