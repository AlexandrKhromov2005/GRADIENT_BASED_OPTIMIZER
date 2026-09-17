#include "embedding_core.h"
#include "embedding_schemes.h"
#include "image_processing_custom.h"
#include "block_kernels.h"
#include "random_utils.h"
#include "gbo.h"
#include "config.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <random>
#include <thread>

// ---- Execution control -------------------------------------------------------

namespace {

std::atomic<unsigned> g_threads{0};
std::atomic<bool> g_has_seed{false};
std::atomic<uint64_t> g_seed{0};

uint64_t nextBaseSeed() {
    if (g_has_seed.load()) {
        // Consecutive embeddings under one fixed seed still get different streams.
        return g_seed.fetch_add(0x9E3779B97F4A7C15ull);
    }
    std::random_device rd;
    return (static_cast<uint64_t>(rd()) << 32) ^ rd();
}

// Calls fn(i) for i in [0, count) on the configured number of threads. Work is handed out in
// small chunks, because the cost of a block varies a lot with its attack type.
void parallelFor(size_t count, size_t chunk, const std::function<void(size_t)>& fn) {
    const size_t max_useful = (count + chunk - 1) / chunk;
    const size_t threads = std::min<size_t>(embeddingThreads(), std::max<size_t>(max_useful, 1));
    if (threads <= 1) {
        for (size_t i = 0; i < count; ++i) fn(i);
        return;
    }
    std::atomic<size_t> next{0};
    auto worker = [&]() {
        for (;;) {
            const size_t begin = next.fetch_add(chunk);
            if (begin >= count) return;
            const size_t end = std::min(begin + chunk, count);
            for (size_t i = begin; i < end; ++i) fn(i);
        }
    };
    std::vector<std::thread> pool;
    pool.reserve(threads - 1);
    for (size_t t = 1; t < threads; ++t) pool.emplace_back(worker);
    worker();
    for (auto& t : pool) t.join();
}

} // namespace

void setEmbeddingThreads(unsigned threads) { g_threads.store(threads); }

unsigned embeddingThreads() {
    unsigned threads = g_threads.load();
    if (threads == 0) {
        if (const char* env = std::getenv("GBO_THREADS")) threads = static_cast<unsigned>(std::atoi(env));
    }
    if (threads == 0) threads = std::thread::hardware_concurrency();
    return std::max(threads, 1u);
}

void setEmbeddingSeed(uint64_t seed) {
    g_seed.store(seed);
    g_has_seed.store(true);
}

void clearEmbeddingSeed() { g_has_seed.store(false); }

// ---- Block-level work --------------------------------------------------------

void runEmbedJobs(std::vector<EmbedJob>& jobs) {
    // Touch every lazily built table before the workers start.
    EmbeddingSchemeManager::getInstance();
    kernels::jpegEmulationIsExact();

    const uint64_t base_seed = nextBaseSeed();
    parallelFor(jobs.size(), 8, [&](size_t i) {
        seed_random_stream(base_seed, i);
        GBO gbo(jobs[i].bit, *jobs[i].block, jobs[i].attack);
        gbo.main_loop();
    });
}

namespace {

// Indices (row * 8 + col) of the two coefficient regions of the current scheme.
struct Regions {
    std::vector<int> reg0, reg1;
    Regions() {
        for (const auto& c : getCurrentREG0()) reg0.push_back(c.first * 8 + c.second);
        for (const auto& c : getCurrentREG1()) reg1.push_back(c.first * 8 + c.second);
    }
};

inline int extractBitAt(const uchar* top_left, size_t step, const Regions& regions, bool equal_is_one) {
    double pixels[64], dct[64];
    for (int r = 0; r < 8; ++r) {
        const uchar* row = top_left + r * step;
        for (int c = 0; c < 8; ++c) pixels[8 * r + c] = row[c];
    }
    kernels::dct8x8(pixels, dct);
    double s0 = 0.0, s1 = 0.0;
    for (int idx : regions.reg0) s0 += std::fabs(dct[idx]);
    for (int idx : regions.reg1) s1 += std::fabs(dct[idx]);

    // S0 and S1 equal in exact arithmetic (flat or heavily quantized blocks): the bit is decided
    // by floating-point noise, so reproduce the noise of the reference transform, cv::dct.
    if (std::fabs(s0 - s1) <= 1e-9 * std::max(1.0, s0 + s1)) {
        cv::Mat block(8, 8, CV_64F, pixels), reference;
        cv::dct(block, reference);
        const double* ref = reference.ptr<double>();
        s0 = s1 = 0.0;
        for (int idx : regions.reg0) s0 += std::fabs(ref[idx]);
        for (int idx : regions.reg1) s1 += std::fabs(ref[idx]);
    }
    if (s0 == s1) return equal_is_one ? 1 : 0;
    return (s0 < s1) ? 1 : 0;
}

} // namespace

int extractBitFromBlock(const cv::Mat& block, bool equal_is_one) {
    CV_Assert(block.type() == CV_8UC1 && block.rows == 8 && block.cols == 8);
    return extractBitAt(block.ptr<uchar>(0), block.step, Regions(), equal_is_one);
}

std::vector<int> extractBlockBits(const cv::Mat& region, size_t max_blocks, bool equal_is_one) {
    CV_Assert(region.type() == CV_8UC1);
    const Regions regions;
    const int block_rows = region.rows / 8, block_cols = region.cols / 8;
    std::vector<int> bits;
    bits.reserve(std::min(max_blocks, static_cast<size_t>(block_rows) * block_cols));
    for (int br = 0; br < block_rows && bits.size() < max_blocks; ++br) {
        const uchar* row = region.ptr<uchar>(br * 8);
        for (int bc = 0; bc < block_cols && bits.size() < max_blocks; ++bc) {
            bits.push_back(extractBitAt(row + bc * 8, region.step, regions, equal_is_one));
        }
    }
    return bits;
}

void embedBlocks(std::vector<cv::Mat>& blocks, const std::vector<int>& wm_bits, AttackType attack, size_t max_blocks) {
    std::vector<EmbedJob> jobs;
    const size_t count = std::min(blocks.size(), max_blocks);
    jobs.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        jobs.push_back({&blocks[i], static_cast<uchar>(wm_bits[i % WM_SIZE]), attack});
    }
    runEmbedJobs(jobs);
}

// ---- Whole-image operations --------------------------------------------------

AttackType quadrantAttackType(int row, int col) {
    static const AttackType pattern[2][2] = {
        {AttackType::NONE, AttackType::JPEG70},
        {AttackType::CONTRAST, AttackType::JPEG80}
    };
    return pattern[row % 2][col % 2];
}

cv::Mat embedBits(const cv::Mat& gray, const std::vector<int>& wm_bits, AttackType attack) {
    std::vector<cv::Mat> blocks = splitInto8x8Blocks(gray);
    embedBlocks(blocks, wm_bits, attack);
    return merge8x8Blocks(blocks, gray.rows, gray.cols);
}

std::vector<int> extractVotes(const cv::Mat& gray) {
    std::vector<int> votes(WM_SIZE, 0);
    const std::vector<int> bits = extractBlockBits(gray);
    for (size_t i = 0; i < bits.size(); ++i) votes[i % WM_SIZE] += bits[i];
    return votes;
}

cv::Mat embedBitsQuadrants(const cv::Mat& gray_1024, const std::vector<int>& wm_bits) {
    const int qh = gray_1024.rows / 4, qw = gray_1024.cols / 4;
    std::vector<std::vector<cv::Mat>> quadrant_blocks(16);
    std::vector<EmbedJob> jobs;
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            std::vector<cv::Mat>& blocks = quadrant_blocks[row * 4 + col];
            blocks = splitInto8x8Blocks(gray_1024(cv::Rect(col * qw, row * qh, qw, qh)));
            const AttackType attack = quadrantAttackType(row, col);
            for (size_t i = 0; i < blocks.size(); ++i) {
                jobs.push_back({&blocks[i], static_cast<uchar>(wm_bits[i % WM_SIZE]), attack});
            }
        }
    }
    runEmbedJobs(jobs);

    cv::Mat result = gray_1024.clone();
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            merge8x8Blocks(quadrant_blocks[row * 4 + col], qh, qw)
                .copyTo(result(cv::Rect(col * qw, row * qh, qw, qh)));
        }
    }
    return result;
}

std::vector<int> extractVotesQuadrants(const cv::Mat& gray_1024, AttackType attack_type) {
    const int qh = gray_1024.rows / 4, qw = gray_1024.cols / 4;
    std::vector<int> votes(WM_SIZE, 0);
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            if (quadrantAttackType(row, col) != attack_type) continue;
            const std::vector<int> bits = extractBlockBits(gray_1024(cv::Rect(col * qw, row * qh, qw, qh)), WM_SIZE);
            for (size_t i = 0; i < bits.size(); ++i) votes[i] += bits[i];
        }
    }
    return votes;
}
