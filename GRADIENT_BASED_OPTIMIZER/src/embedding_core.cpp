#include "embedding_core.h"
#include "image_processing_custom.h"
#include "block_metrics.h"
#include "gbo.h"
#include "config.h"

AttackType quadrantAttackType(int row, int col) {
    static const AttackType pattern[2][2] = {
        {AttackType::NONE, AttackType::JPEG70},
        {AttackType::CONTRAST, AttackType::JPEG80}
    };
    return pattern[row % 2][col % 2];
}

static int extractBit(const cv::Mat& block) {
    cv::Mat dbl_block;
    block.convertTo(dbl_block, CV_64F);
    cv::Mat dct_block;
    cv::dct(dbl_block, dct_block);
    double s0 = calc_s_zero(dct_block);
    double s1 = calc_s_one(dct_block);
    return (s0 < s1) ? 1 : 0;
}

cv::Mat embedBits(const cv::Mat& gray, const std::vector<int>& wm_bits) {
    std::vector<cv::Mat> blocks = splitInto8x8Blocks(gray);
    for (size_t i = 0; i < blocks.size(); ++i) {
        GBO gbo(static_cast<uchar>(wm_bits[i % WM_SIZE]), blocks[i]);
        gbo.main_loop();
    }
    return merge8x8Blocks(blocks, gray.rows, gray.cols);
}

std::vector<int> extractVotes(const cv::Mat& gray) {
    std::vector<cv::Mat> blocks = splitInto8x8Blocks(gray);
    std::vector<int> votes(WM_SIZE, 0);
    for (size_t i = 0; i < blocks.size(); ++i) {
        votes[i % WM_SIZE] += extractBit(blocks[i]);
    }
    return votes;
}

cv::Mat embedBitsQuadrants(const cv::Mat& gray_1024, const std::vector<int>& wm_bits) {
    const int qh = gray_1024.rows / 4, qw = gray_1024.cols / 4;
    cv::Mat result = gray_1024.clone();
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            cv::Rect roi(col * qw, row * qh, qw, qh);
            cv::Mat quadrant = gray_1024(roi).clone();
            AttackType attack = quadrantAttackType(row, col);

            std::vector<cv::Mat> blocks = splitInto8x8Blocks(quadrant);
            for (size_t i = 0; i < blocks.size(); ++i) {
                GBO gbo(static_cast<uchar>(wm_bits[i % WM_SIZE]), blocks[i], attack);
                gbo.main_loop();
            }
            merge8x8Blocks(blocks, qh, qw).copyTo(result(roi));
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
            cv::Mat quadrant = gray_1024(cv::Rect(col * qw, row * qh, qw, qh)).clone();
            std::vector<cv::Mat> blocks = splitInto8x8Blocks(quadrant);
            for (size_t i = 0; i < WM_SIZE && i < blocks.size(); ++i) {
                votes[i] += extractBit(blocks[i]);
            }
        }
    }
    return votes;
}
