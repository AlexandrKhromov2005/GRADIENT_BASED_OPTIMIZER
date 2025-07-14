#include <opencv2/opencv.hpp>
#include <iostream>
#include <immintrin.h>
#include <string>
#include "../include/embending_and_extracting.h"

int main() {
    const std::string& imagePath = "../images/lenna.png";
    const std::string& watermarkPath = "../images/watermark.png";
    size_t a = 0;
    size_t b = 0;

    embend(imagePath, watermarkPath, a, b);
    return 0;
}