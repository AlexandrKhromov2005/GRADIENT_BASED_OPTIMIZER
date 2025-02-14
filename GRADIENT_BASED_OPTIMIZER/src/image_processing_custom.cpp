#include <iostream>
#include <opencv2/opencv.hpp>
#include "image_processing_custom.h"

// Function to read an image in grayscale mode
cv::Mat readImage(const std::string& filename) {
    // Read the image in grayscale
    cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error: cannot open image: " << filename << std::endl;
    }
    return image;
}

// Function to write an image to a file
bool writeImage(const std::string& filename, const cv::Mat& image) {
    try {
        if (cv::imwrite(filename, image)) {
            return true;
        }
    }
    catch (const cv::Exception& ex) {
        std::cerr << "Error during saving: " << ex.what() << std::endl;
    }
    std::cerr << "Error: cannot save image: " << filename << std::endl;
    return false;
}


// Function to split an image into 8x8 blocks
std::vector<cv::Mat> splitInto8x8Blocks(const cv::Mat& image) {
    std::vector<cv::Mat> blocks;
    const int blockSize = 8;
    int rows = image.rows;
    int cols = image.cols;

    // If the image dimensions are not multiples of 8, crop the image to the nearest size divisible by 8
    int validRows = (rows / blockSize) * blockSize;
    int validCols = (cols / blockSize) * blockSize;

    for (int i = 0; i < validRows; i += blockSize) {
        for (int j = 0; j < validCols; j += blockSize) {
            cv::Rect roi(j, i, blockSize, blockSize);
            // Clone the ROI to create a separate cv::Mat object
            blocks.push_back(image(roi).clone());
        }
    }
    return blocks;
}

// Function to merge 8x8 blocks into a single image
cv::Mat merge8x8Blocks(const std::vector<cv::Mat>& blocks, int rows, int cols) {
    if (blocks.empty()) { // Проверка на пустой вектор
        std::cerr << "Error: No blocks to merge." << std::endl;
        return cv::Mat();
    }
    const int blockSize = 8;
    cv::Mat mergedImage(rows, cols, blocks[0].type());
    int blockIndex = 0;
    for (int i = 0; i < rows; i += blockSize) {
        for (int j = 0; j < cols; j += blockSize) {
            if (blockIndex < blocks.size()) {
                cv::Rect roi(j, i, blockSize, blockSize);
                blocks[blockIndex].copyTo(mergedImage(roi));
                blockIndex++;
            }
        }
    }
    return mergedImage;
}

// wm -> binary
std::vector<int> convertWatermarkToBinary(const cv::Mat& watermark) {
    std::vector<int> binaryData;
    if (watermark.channels() != 1) {
        std::cerr << "Error: Watermark must be a grayscale image." << std::endl;
        return binaryData;
    }

    for (int i = 0; i < watermark.rows; ++i) {
        for (int j = 0; j < watermark.cols; ++j) {
            // Черный пиксель (0) -> 1, белый (255) -> 0
            binaryData.push_back(watermark.at<uchar>(i, j) < 128 ? 1 : 0);
        }
    }
    return binaryData;
}

// binary -> wm
cv::Mat convertBinaryToWatermark(const std::vector<int>& binaryData) {
    cv::Mat watermark;
    if (binaryData.empty()) {
        std::cerr << "Error: Binary data is empty." << std::endl;
        return watermark;
    }
    if (binaryData.size() != 32 * 32) {
        std::cerr << "Error: Binary data size does not match rows*cols." << std::endl;
        return watermark;
    }

    watermark.create(32, 32, CV_8UC1);
    int index = 0;
    for (int i = 0; i < 32; ++i) {
        for (int j = 0; j < 32; ++j) {
            // 1 -> черный (0), 0 -> белый (255)
            watermark.at<uchar>(i, j) = (binaryData[index] == 1) ? 0 : 255;
            index++;
        }
    }
    return watermark;
}
