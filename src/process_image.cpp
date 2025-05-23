#include "../include/process_image.h"

//readImageAndSplitToBlocks for read image
std::vector<Block> readImageAndSplitToBlocks(const std::string& imagePath) {
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        throw std::runtime_error("Не удалось загрузить изображение");
    }

    if (image.rows % 8 != 0 || image.cols % 8 != 0) {
        throw std::runtime_error("Размеры изображения должны быть кратны 8");
    }

    std::vector<Block> blocks;

    for (int i = 0; i < image.rows; i += 8) {
        for (int j = 0; j < image.cols; j += 8) {
            cv::Mat blockMat = image(cv::Rect(j, i, 8, 8)).clone();

            blockMat.convertTo(blockMat, CV_64F);

            Block block(blockMat);

            blocks.push_back(block);
        }
    }

    return blocks;
}

//reconstructImageFromBlocks for write image
cv::Mat reconstructImageFromBlocks(const std::vector<Block>& blocks, int rows, int cols) {
    cv::Mat image(rows, cols, CV_64F);

    int blockRows = rows / 8;
    int blockCols = cols / 8;

    for (int i = 0; i < blockRows; ++i) {
        for (int j = 0; j < blockCols; ++j) {
            int blockIndex = i * blockCols + j;
            if (blockIndex >= blocks.size()) {
                break; 
            }

            const Block& block = blocks[blockIndex];

            for (int y = 0; y < 8; ++y) {
                for (int x = 0; x < 8; ++x) {
                    image.at<double>(i * 8 + y, j * 8 + x) = block.spatialDomain(y, x);
                }
            }
        }
    }

    cv::Mat image8u;
    image.convertTo(image8u, CV_8U, 255.0 / 65025.0); 

    return image8u;
}

//readCvzAndConvertToVector for read watermark
std::vector<uchar> readCvzAndConvertToVector(const std::string& imagePath) {
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        throw std::runtime_error("Не удалось загрузить изображение ЦВЗ");
    }

    if (image.rows != 32 || image.cols != 32) {
        throw std::runtime_error("Размеры изображения ЦВЗ должны быть 32x32");
    }

    cv::Mat binaryImage;
    cv::threshold(image, binaryImage, 128, 255, cv::THRESH_BINARY);

    std::vector<uchar> cvzVector;
    cvzVector.reserve(32 * 32);

    for (int i = 0; i < binaryImage.rows; ++i) {
        for (int j = 0; j < binaryImage.cols; ++j) {
            cvzVector.push_back(binaryImage.at<uchar>(i, j) == 255 ? 1 : 0);
        }
    }

    return cvzVector;
}

//reconstructCvzFromVector for write watermark
cv::Mat reconstructCvzFromVector(const std::vector<uchar>& cvzVector) {
    if (cvzVector.size() != 32 * 32) {
        throw std::runtime_error("Вектор ЦВЗ должен содержать 1024 элемента");
    }

    cv::Mat cvzImage(32, 32, CV_8U);

    for (int i = 0; i < 32; ++i) {
        for (int j = 0; j < 32; ++j) {
            cvzImage.at<uchar>(i, j) = cvzVector[i * 32 + j] ? 255 : 0;
        }
    }

    return cvzImage;
}

//addPrefixToFilename for adding prefix "wm_"
std::string addPrefixToFilename(const std::string& imagePath, const std::string& prefix) {
    size_t lastSlashPos = imagePath.find_last_of("/\\");
    if (lastSlashPos == std::string::npos) {
        return prefix + imagePath;
    } else {
        return imagePath.substr(0, lastSlashPos + 1) + prefix + imagePath.substr(lastSlashPos + 1);
    }
}