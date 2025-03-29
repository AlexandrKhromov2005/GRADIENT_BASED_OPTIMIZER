#pragma once
#include <opencv2/opencv.hpp>
#include <array>
#include <vector>
#include <utility>
#include <type_traits>

constexpr std::array<std::array<int, 8>, 8> BASE_TABLE = { {
    {16, 11, 10, 16, 24, 40, 51, 61},
    {12, 12, 14, 19, 26, 58, 60, 55},
    {14, 13, 16, 24, 40, 57, 69, 56},
    {14, 17, 22, 29, 51, 87, 80, 62},
    {18, 22, 37, 56, 68, 109, 103, 77},
    {24, 35, 55, 64, 81, 104, 113, 92},
    {49, 64, 78, 87, 103, 121, 120, 101},
    {72, 92, 95, 98, 112, 100, 103, 99}
} };

constexpr int generate_quantization_value(int base, int quality) {
    const double value = (base * 100.0) / quality;
    return static_cast<int>(value + 0.5) < 1 ? 1 : static_cast<int>(value + 0.5);
}

constexpr std::array<std::array<int, 8>, 8> generate_quantization_table(int quality) {
    std::array<std::array<int, 8>, 8> quantTable{};
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            quantTable[i][j] = generate_quantization_value(BASE_TABLE[i][j], quality);
        }
    }
    return quantTable;
}

template<int Quality>
constexpr std::array<std::array<int, 8>, 8> generate_q_table() {
    return generate_quantization_table(Quality);
}

template<int... Qualities>
constexpr auto generate_tables_impl(std::integer_sequence<int, Qualities...>) {
    return std::array<std::array<std::array<int, 8>, 8>, sizeof...(Qualities)>{
        generate_quantization_table(Qualities + 1)...
    };
}

constexpr auto generate_all_quantization_tables() {
    return generate_tables_impl(std::make_integer_sequence<int, 100>{});
}

constexpr auto quantization_tables = generate_all_quantization_tables();

const std::vector<cv::Mat> get_quantization_tables() {
    std::vector<cv::Mat> tables;
    tables.reserve(100);

    for (int quality = 1; quality <= 100; ++quality) {
        const auto& quant_table = quantization_tables[quality - 1];
        cv::Mat mat(8, 8, CV_32S);

        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
                mat.at<int>(i, j) = quant_table[i][j];
            }
        }
        tables.push_back(mat.clone());
    }

    return tables;
}