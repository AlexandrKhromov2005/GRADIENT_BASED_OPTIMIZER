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
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <random>
#include <thread>

// ---- Execution control -------------------------------------------------------

namespace {

std::atomic<unsigned> g_threads{0};
std::atomic<bool> g_has_seed{false};
std::atomic<uint64_t> g_seed{0};
std::atomic<uint64_t> g_seeded_calls{0};  // embedding passes since the last setEmbeddingSeed()

uint64_t nextBaseSeed() {
    if (g_has_seed.load()) {
        // The n-th pass after setEmbeddingSeed(s) always gets the same streams, and
        // different passes (e.g. the 16 quadrants of one image) get different ones.
        return g_seed.load() + g_seeded_calls.fetch_add(1) * 0x9E3779B97F4A7C15ull;
    }
    std::random_device rd;
    const uint64_t high = rd();
    const uint64_t low = rd();
    return (high << 32) ^ low;
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
    std::atomic<bool> failed{false};
    std::exception_ptr error;
    std::mutex error_mutex;
    auto worker = [&]() {
        try {
            for (;;) {
                const size_t begin = next.fetch_add(chunk);
                if (begin >= count || failed.load()) return;
                const size_t end = std::min(begin + chunk, count);
                for (size_t i = begin; i < end; ++i) fn(i);
            }
        } catch (...) {  // an exception must not escape a std::thread: hand it to the caller
            std::lock_guard<std::mutex> lock(error_mutex);
            if (!error) error = std::current_exception();
            failed.store(true);
        }
    };
    std::vector<std::thread> pool;
    pool.reserve(threads - 1);
    try {
        for (size_t t = 1; t < threads; ++t) pool.emplace_back(worker);
    } catch (...) {
        // Could not start every thread: the ones that did start (and this one) do the work.
    }
    worker();
    for (auto& t : pool) t.join();
    if (error) std::rethrow_exception(error);
}

} // namespace

void setEmbeddingThreads(unsigned threads) { g_threads.store(threads); }

unsigned embeddingThreads() {
    const unsigned hardware = std::max(std::thread::hardware_concurrency(), 1u);
    unsigned threads = g_threads.load();
    if (threads == 0) {
        if (const char* env = std::getenv("GBO_THREADS")) {
            const long value = std::strtol(env, nullptr, 10);
            if (value > 0) threads = static_cast<unsigned>(std::min<long>(value, 4096));
        }
    }
    if (threads == 0) threads = hardware;
    return std::min(threads, 4 * hardware);  // more threads than that only cost memory
}

void setEmbeddingSeed(uint64_t seed) {
    g_seed.store(seed);
    g_seeded_calls.store(0);
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
        fill(getCurrentREG0(), reg0);
        fill(getCurrentREG1(), reg1);
    }
    static void fill(const std::vector<std::pair<int, int>>& coords, std::vector<int>& idx) {
        for (const auto& c : coords) {
            CV_Assert(c.first >= 0 && c.first < 8 && c.second >= 0 && c.second < 8);
            idx.push_back(c.first * 8 + c.second);
        }
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
    if (wm_bits.size() < WM_SIZE) {
        throw std::invalid_argument("watermark must provide at least WM_SIZE bits");
    }
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

namespace {

// Writes the blocks (splitInto8x8Blocks order) back over `target`. Pixels outside the grid of
// full blocks - the border of an image whose side is not a multiple of 8 - are left alone.
void writeBlocks(const std::vector<cv::Mat>& blocks, cv::Mat target) {
    size_t index = 0;
    for (int r = 0; r + 8 <= target.rows; r += 8) {
        for (int c = 0; c + 8 <= target.cols && index < blocks.size(); c += 8) {
            blocks[index++].copyTo(target(cv::Rect(c, r, 8, 8)));
        }
    }
}

} // namespace

cv::Mat embedBits(const cv::Mat& gray, const std::vector<int>& wm_bits, AttackType attack) {
    CV_Assert(gray.type() == CV_8UC1);
    std::vector<cv::Mat> blocks = splitInto8x8Blocks(gray);
    embedBlocks(blocks, wm_bits, attack);
    cv::Mat result = gray.clone();
    writeBlocks(blocks, result);
    return result;
}

std::vector<int> extractVotes(const cv::Mat& gray) {
    std::vector<int> votes(WM_SIZE, 0);
    const std::vector<int> bits = extractBlockBits(gray);
    for (size_t i = 0; i < bits.size(); ++i) votes[i % WM_SIZE] += bits[i];
    return votes;
}

cv::Mat embedBitsQuadrants(const cv::Mat& gray_1024, const std::vector<int>& wm_bits) {
    if (wm_bits.size() < WM_SIZE) {
        throw std::invalid_argument("watermark must provide at least WM_SIZE bits");
    }
    CV_Assert(gray_1024.type() == CV_8UC1);
    if (gray_1024.empty()) return gray_1024.clone();
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
            writeBlocks(quadrant_blocks[row * 4 + col], result(cv::Rect(col * qw, row * qh, qw, qh)));
        }
    }
    return result;
}

std::vector<int> extractVotesQuadrants(const cv::Mat& gray_1024, AttackType attack_type) {
    CV_Assert(gray_1024.type() == CV_8UC1);
    const int qh = gray_1024.rows / 4, qw = gray_1024.cols / 4;
    std::vector<int> votes(WM_SIZE, 0);
    if (gray_1024.empty()) return votes;
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            if (quadrantAttackType(row, col) != attack_type) continue;
            const std::vector<int> bits = extractBlockBits(gray_1024(cv::Rect(col * qw, row * qh, qw, qh)), WM_SIZE);
            for (size_t i = 0; i < bits.size(); ++i) votes[i] += bits[i];
        }
    }
    return votes;
}
