#include "compression.h"

cv::Mat compress_block(cv::Mat& original_block, int quality) {
    const auto& quant_array = quantization_tables[quality - 1];
    cv::Mat quant_table = qtable_to_mat(quant_array);

    cv::Mat block_double, quant_double;
    original_block.convertTo(block_double, CV_64F);
    quant_table.convertTo(quant_double, CV_64F);

    cv::Mat result_float;
    cv::divide(block_double, quant_double, result_float);

    cv::Mat rounded(8, 8, CV_32S);
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            rounded.at<int>(i, j) = static_cast<int>(
                std::round(result_float.at<float>(i, j))
                );
        }
    }

    return rounded * quant_table;
}