#ifndef EMBEDDING_CORE_H
#define EMBEDDING_CORE_H

#include <opencv2/opencv.hpp>
#include <cstdint>
#include <vector>
#include "population.h"

// ---- Execution control -------------------------------------------------------

// Number of worker threads used by embedding / extraction. 0 = all hardware threads
// (default; can also be set with the GBO_THREADS environment variable).
void setEmbeddingThreads(unsigned threads);
unsigned embeddingThreads();

// Fixes the seed of all following embeddings, which makes them reproducible: the same
// sequence of embedding calls after setEmbeddingSeed(s) always gives the same images, and
// the result never depends on the number of threads. Call it again before an embedding to
// reproduce that embedding. Without a fixed seed every embedding draws a fresh seed from
// std::random_device.
//
// Threading contract: embedding calls may not run concurrently with each other under a fixed
// seed (which call gets which stream would be undefined), and the active scheme must not be
// changed (setCurrentScheme / loadSchemes) while an embedding or extraction is running.
void setEmbeddingSeed(uint64_t seed);
void clearEmbeddingSeed();

// ---- Block-level work --------------------------------------------------------

// One 8x8 block to embed a bit into. `block` is modified in place.
struct EmbedJob {
    cv::Mat* block;
    uchar bit;
    AttackType attack;
};

// Runs GBO for every job, in parallel. Each job gets its own random stream derived from
// (seed, job index), so the outcome does not depend on scheduling.
void runEmbedJobs(std::vector<EmbedJob>& jobs);

// Embeds wm_bits[i % WM_SIZE] into blocks[i] (in place) for the first `max_blocks` blocks.
// Blocks must be 8x8 CV_8UC1. Throws std::invalid_argument if wm_bits has fewer than WM_SIZE bits.
void embedBlocks(std::vector<cv::Mat>& blocks, const std::vector<int>& wm_bits,
                 AttackType attack = AttackType::NONE, size_t max_blocks = static_cast<size_t>(-1));

// Bit carried by one 8x8 block (CV_8U) under the current scheme: 1 if S0 < S1, else 0.
// `equal_is_one` selects the bit reported when S0 == S1 exactly.
int extractBitFromBlock(const cv::Mat& block, bool equal_is_one = false);

// Bits of the 8x8 blocks of a grayscale (CV_8U) image region, row-major block order.
std::vector<int> extractBlockBits(const cv::Mat& region, size_t max_blocks = static_cast<size_t>(-1),
                                  bool equal_is_one = false);

// ---- Whole-image operations --------------------------------------------------

// Attack type the quadrant (row, col) of the 4x4 grid is optimized for.
// Pattern: NONE/JPEG70/NONE/JPEG70 / CONTRAST/JPEG80/CONTRAST/JPEG80 (repeated)
AttackType quadrantAttackType(int row, int col);

// Base algorithm: embed wm_bits[i % WM_SIZE] into every 8x8 block of a grayscale image.
cv::Mat embedBits(const cv::Mat& gray, const std::vector<int>& wm_bits,
                  AttackType attack = AttackType::NONE);

// Base algorithm: per-bit count of blocks voting for 1 (one vote per watermark copy).
std::vector<int> extractVotes(const cv::Mat& gray);

// Modified algorithm: 4x4 grid of quadrants (256x256 for a 1024x1024 image), each
// optimized for its attack type; every quadrant carries one watermark copy.
cv::Mat embedBitsQuadrants(const cv::Mat& gray_1024, const std::vector<int>& wm_bits);

// Modified algorithm: per-bit count of votes for 1 among the quadrants of the given attack type.
std::vector<int> extractVotesQuadrants(const cv::Mat& gray_1024, AttackType attack_type);

#endif // EMBEDDING_CORE_H
