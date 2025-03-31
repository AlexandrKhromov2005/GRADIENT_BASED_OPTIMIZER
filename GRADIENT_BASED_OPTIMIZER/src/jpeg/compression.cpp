#include "compression.h"

cv::Mat compress_block(cv::Mat& original_block, int quality) {
    const cv::Mat& quant_table = quantization_mats[quality - 1];

    cv::Mat block_float;
    original_block.convertTo(block_float, CV_64F);

    cv::Mat dct_coeffs;
    cv::dct(block_float, dct_coeffs);

    cv::Mat quant_table_float;
    quant_table.convertTo(quant_table_float, CV_64F);

    cv::Mat quantized_coeffs;
    cv::divide(dct_coeffs, quant_table_float, quantized_coeffs);
    quantized_coeffs.convertTo(quantized_coeffs, CV_32S); 
    quantized_coeffs.convertTo(quantized_coeffs, CV_64F); 

    cv::Mat dequantized_coeffs;
    cv::multiply(quantized_coeffs, quant_table_float, dequantized_coeffs);

    cv::Mat reconstructed_block;
    cv::idct(dequantized_coeffs, reconstructed_block);

    cv::Mat final_block;
    reconstructed_block.convertTo(final_block, CV_8U, 1.0, 0.5);

    return final_block.clone();
}