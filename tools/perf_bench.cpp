// Reproducible speed / quality benchmark for the base and the quadrant (modified) algorithm.
// Prints timings, a hash of the watermarked image and quality metrics, so that two builds
// can be compared: identical seed + identical hash == identical computation.
#include <opencv2/opencv.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <random>
#include <cstring>
#include <string>
#include <vector>

#include "embedding_core.h"
#include "embedding_schemes.h"
#include "image_processing_custom.h"
#include "image_metrics.h"
#include "attacks.h"
#include "random_utils.h"
#include "config.h"
#include "block_kernels.h"
#include "block_metrics.h"
#include "jpeg/quantization_tables.h"

using Clock = std::chrono::steady_clock;
static double secondsSince(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

static uint64_t fnv1a(const cv::Mat& m) {
    uint64_t h = 1469598103934665603ULL;
    for (int r = 0; r < m.rows; ++r) {
        const uchar* p = m.ptr<uchar>(r);
        for (int c = 0; c < m.cols; ++c) { h ^= p[c]; h *= 1099511628211ULL; }
    }
    return h;
}

struct Attack { std::string name; std::function<cv::Mat(const cv::Mat&)> fn; };

static AttackType attackTypeFromName(const std::string& n) {
    if (n.find("JPEG") != std::string::npos) {
        if (n.find("70") != std::string::npos) return AttackType::JPEG70;
        if (n.find("80") != std::string::npos) return AttackType::JPEG80;
    }
    if (n.find("Contrast") != std::string::npos) return AttackType::CONTRAST;
    return AttackType::NONE;
}

// Compares the allocation-free kernels with the OpenCV reference on blocks of a real image
// (plus random perturbations): JPEG round trip and contrast must match bit for bit.
static int selfTest(const cv::Mat& gray, int rounds) {
    std::vector<cv::Mat> blocks = splitInto8x8Blocks(gray);
    long jpeg_checked = 0, jpeg_bad = 0, contrast_bad = 0, round_bad = 0;
    double max_dct_err = 0, max_idct_err = 0;
    std::mt19937 rng(7);
    std::printf("jpeg emulation probe: %s\n", kernels::jpegEmulationIsExact() ? "exact" : "MISMATCH -> codec fallback");
    for (int round = 0; round < rounds; ++round) {
        for (const cv::Mat& src : blocks) {
            cv::Mat block = src.clone();
            if (round > 0) {  // perturb like an embedding would (and sometimes much harder)
                const int amp = (round % 4 == 0) ? 64 : 1 + round % 6;
                for (int i = 0; i < 64; ++i) {
                    int v = block.data[i] + int(rng() % (2 * amp + 1)) - amp;
                    block.data[i] = cv::saturate_cast<uchar>(v);
                }
            }
            uint8_t in[64], out[64];
            kernels::loadBlock(block, in);
            for (int q : {70, 80, 90, 50, 30, 100, 1 + int(rng() % 100)}) {
                kernels::jpegRoundTrip(in, out, q);
                cv::Mat ref = jpegCompression(block, q);
                ++jpeg_checked;
                if (std::memcmp(out, ref.data, 64) != 0) ++jpeg_bad;
            }
            kernels::contrastU8(in, out, 1.1);
            if (std::memcmp(out, contrastIncrease(block, 1.1).data, 64) != 0) ++contrast_bad;

            cv::Mat dbl, ref_dct, ref_idct, ref_u8;
            block.convertTo(dbl, CV_64F);
            cv::dct(dbl, ref_dct);
            double mine[64], back[64];
            kernels::dct8x8(dbl.ptr<double>(), mine);
            for (int i = 0; i < 64; ++i) max_dct_err = std::max(max_dct_err, std::fabs(mine[i] - ref_dct.ptr<double>()[i]));
            kernels::dct8x8Matrix(dbl.ptr<double>(), back);
            for (int i = 0; i < 64; ++i) max_dct_err = std::max(max_dct_err, std::fabs(mine[i] - back[i]));
            for (int i = 0; i < 64; ++i) ref_dct.ptr<double>()[i] += (i > 20 && i < 43) ? (rng() % 2000) / 100.0 - 10.0 : 0.0;
            cv::idct(ref_dct, ref_idct);
            kernels::idct8x8(ref_dct.ptr<double>(), back);
            for (int i = 0; i < 64; ++i) max_idct_err = std::max(max_idct_err, std::fabs(back[i] - ref_idct.ptr<double>()[i]));
            kernels::idct8x8Matrix(ref_dct.ptr<double>(), mine);
            for (int i = 0; i < 64; ++i) max_idct_err = std::max(max_idct_err, std::fabs(back[i] - mine[i]));
            ref_idct.convertTo(ref_u8, CV_8U);
            kernels::roundToU8(ref_idct.ptr<double>(), out);
            if (std::memcmp(out, ref_u8.data, 64) != 0) ++round_bad;
            // The composition the embedding really uses: fast IDCT, rounding, and the
            // reference transform whenever a value is too close to a rounding boundary.
            if (!kernels::roundToU8(back, out)) kernels::roundToU8(ref_idct.ptr<double>(), out);
            if (std::memcmp(out, ref_u8.data, 64) != 0) ++round_bad;
            // Same again with every pixel forced onto an exact x.5 boundary.
            cv::Mat half = ref_u8.clone(); half.convertTo(half, CV_64F); half += 0.5;
            cv::Mat half_dct, half_idct, half_u8; cv::dct(half, half_dct); cv::idct(half_dct, half_idct);
            half_idct.convertTo(half_u8, CV_8U);
            kernels::idct8x8(half_dct.ptr<double>(), back);
            if (!kernels::roundToU8(back, out)) kernels::roundToU8(half_idct.ptr<double>(), out);
            if (std::memcmp(out, half_u8.data, 64) != 0) ++round_bad;
        }
    }
    // Extraction: fast path against the cv::dct reference, on clean and attacked images.
    long bits_checked = 0, bits_bad = 0;
    for (const cv::Mat& img : {gray, jpegCompression(gray, 70), jpegCompression(gray, 30), contrastIncrease(gray, 1.1),
                               medianFiltering(gray, 5), gaussianFiltering(gray, 5), sharpening(gray),
                               histogramEqualization(gray), brightnessIncrease(gray, 50)}) {
        for (const cv::Mat& block : splitInto8x8Blocks(img)) {
            cv::Mat dbl, dct;
            block.convertTo(dbl, CV_64F);
            cv::dct(dbl, dct);
            const int ref = (calc_s_zero(dct) < calc_s_one(dct)) ? 1 : 0;
            ++bits_checked;
            if (ref != extractBitFromBlock(block)) { ++bits_bad; std::printf("  mismatch: s0=%.17g s1=%.17g\n", calc_s_zero(dct), calc_s_one(dct)); }
        }
    }
    std::printf("extraction: %ld blocks, %ld bit mismatches\n", bits_checked, bits_bad);
    if (bits_bad) return 1;

    std::printf("jpeg: %ld round trips, %ld mismatches\ncontrast mismatches: %ld\nrounding mismatches: %ld\n"
                "max |dct - cv::dct| = %.3e, max |idct - cv::idct| = %.3e\n",
                jpeg_checked, jpeg_bad, contrast_bad, round_bad, max_dct_err, max_idct_err);
    return (jpeg_bad || contrast_bad || round_bad) ? 1 : 0;
}

static int run(int argc, char** argv) {
    std::string mode = "base", image = "images/lenna.png", wm = "images/watermark_32x32.png";
    std::string scheme = "scheme1", out;
    unsigned seed = 1;
    int repeat = 1, crop = 0, selftest = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() {
            if (i + 1 >= argc) { std::cerr << "missing value for " << a << "\n"; std::exit(2); }
            return std::string(argv[++i]);
        };
        if (a == "--mode") mode = next();
        else if (a == "--image") image = next();
        else if (a == "--wm") wm = next();
        else if (a == "--scheme") scheme = next();
        else if (a == "--seed") seed = std::stoul(next());
        else if (a == "--repeat") repeat = std::stoi(next());
        else if (a == "--crop") crop = std::stoi(next());
        else if (a == "--out") out = next();
        else if (a == "--selftest") selftest = std::stoi(next());
        else if (a == "--threads") setEmbeddingThreads(std::stoul(next()));
        else { std::cerr << "unknown option " << a << "\n"; return 2; }
    }

    std::cout.setstate(std::ios::failbit);  // silence library chatter
    auto& mgr = EmbeddingSchemeManager::getInstance();
    mgr.loadSchemes("embedding_schemes.json");
    mgr.setCurrentScheme(scheme);
    initialize_quantization_mats();
    std::cout.clear();

    cv::Mat gray = readImage(image);
    if (gray.empty()) return 1;
    if (crop > 0) gray = gray(cv::Rect(0, 0, crop, crop)).clone();
    if (selftest > 0) return selfTest(gray, selftest);
    const cv::Mat wm_img = readImage(wm);
    const std::vector<int> bits = convertWatermarkToBinary(wm_img);
    if (bits.size() != WM_SIZE) { std::cerr << "watermark must be 32x32\n"; return 1; }
    const bool quad = (mode == "quad");

    const std::vector<Attack> attacks = {
        {"No attack", [](const cv::Mat& m) { return m; }},
        {"Brightness increase", [](const cv::Mat& m) { return brightnessIncrease(m, 50); }},
        {"Contrast increase", [](const cv::Mat& m) { return contrastIncrease(m, 1.1); }},
        {"Contrast decrease", [](const cv::Mat& m) { return contrastDecrease(m, 0.9); }},
        {"Salt Pepper Noise", [](const cv::Mat& m) { srand(12345); return saltPepperNoise(m, 0.02); }},
        {"Histogram Equalization", [](const cv::Mat& m) { return histogramEqualization(m); }},
        {"Sharpening", [](const cv::Mat& m) { return sharpening(m); }},
        {"JPEG90", [](const cv::Mat& m) { return jpegCompression(m, 90); }},
        {"JPEG80", [](const cv::Mat& m) { return jpegCompression(m, 80); }},
        {"JPEG70", [](const cv::Mat& m) { return jpegCompression(m, 70); }},
        {"Gaussian Filtering", [](const cv::Mat& m) { return gaussianFiltering(m, 5); }},
        {"Median Filtering", [](const cv::Mat& m) { return medianFiltering(m, 5); }},
    };

    std::printf("mode=%s image=%s size=%dx%d scheme=%s blocks=%d threads=%u\n", mode.c_str(), image.c_str(),
                gray.cols, gray.rows, scheme.c_str(), (gray.rows / 8) * (gray.cols / 8), embeddingThreads());

    double sum_embed = 0, sum_extract = 0, sum_psnr = 0, sum_ssim = 0;
    std::vector<double> sum_ber(attacks.size(), 0.0);

    for (int r = 0; r < repeat; ++r) {
        setEmbeddingSeed(seed + r);
        auto t0 = Clock::now();
        cv::Mat marked = quad ? embedBitsQuadrants(gray, bits) : embedBits(gray, bits);
        const double t_embed = secondsSince(t0);

        const double psnr = computePSNR(gray, marked), ssim = computeSSIM(gray, marked);
        std::printf("run seed=%u embed_s=%.4f hash=%016llx psnr=%.10f ssim=%.10f\n", seed + r, t_embed,
                    (unsigned long long)fnv1a(marked), psnr, ssim);
        if (!out.empty() && r == 0) writeImage(out, marked);
        sum_embed += t_embed; sum_psnr += psnr; sum_ssim += ssim;

        double t_extract = 0;
        for (size_t a = 0; a < attacks.size(); ++a) {
            const cv::Mat attacked = attacks[a].fn(marked);
            auto t1 = Clock::now();
            std::vector<int> votes = quad ? extractVotesQuadrants(attacked, attackTypeFromName(attacks[a].name))
                                          : extractVotes(attacked);
            t_extract += secondsSince(t1);
            const int copies = quad ? 4 : (gray.rows / 8) * (gray.cols / 8) / WM_SIZE;
            int errors2 = 0, ties = 0;  // errors counted in half-bits: a tie is half an error
            for (size_t i = 0; i < WM_SIZE; ++i) {
                if (2 * votes[i] == copies) { ++ties; ++errors2; }
                else if ((2 * votes[i] > copies ? 1 : 0) != bits[i]) errors2 += 2;
            }
            const double ber = errors2 / (2.0 * WM_SIZE);
            sum_ber[a] += ber;
            std::printf("  %-24s ber=%.6f ties=%d\n", attacks[a].name.c_str(), ber, ties);
        }
        sum_extract += t_extract / attacks.size();
    }

    std::printf("MEAN over %d run(s): embed_s=%.4f extract_s=%.6f psnr=%.6f ssim=%.6f\n", repeat,
                sum_embed / repeat, sum_extract / repeat, sum_psnr / repeat, sum_ssim / repeat);
    for (size_t a = 0; a < attacks.size(); ++a)
        std::printf("  MEAN %-24s ber=%.6f\n", attacks[a].name.c_str(), sum_ber[a] / repeat);
    return 0;
}

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
