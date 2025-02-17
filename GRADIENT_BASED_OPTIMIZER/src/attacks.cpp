#include "attacks.h"


// Brightness Increase
cv::Mat brightnessIncrease(const cv::Mat& image, int value) {
    cv::Mat result;
    image.convertTo(result, -1, 1, value);  // Increase brightness by a constant value
    return result;
}

// Brightness Decrease
cv::Mat brightnessDecrease(const cv::Mat& image, int value) {
    cv::Mat result;
    image.convertTo(result, -1, 1, -value);  // Decrease brightness by a constant value
    return result;
}

// Contrast Increase
cv::Mat contrastIncrease(const cv::Mat& image, double alpha) {
    cv::Mat result;
    image.convertTo(result, -1, alpha, 0);  // Increase contrast
    return result;
}

// Contrast Decrease
cv::Mat contrastDecrease(const cv::Mat& image, double alpha) {
    cv::Mat result;
    image.convertTo(result, -1, alpha, 0);  // Decrease contrast
    return result;
}

// Salt-Pepper Noise
cv::Mat saltPepperNoise(const cv::Mat& image, double noiseProb) {
    cv::Mat result = image.clone();
    int numPixels = result.rows * result.cols;
    for (int i = 0; i < numPixels; i++) {
        if (rand() % 100 < noiseProb * 100) {
            int row = rand() % result.rows;
            int col = rand() % result.cols;
            if (rand() % 2 == 0) {
                result.at<uchar>(row, col) = 0;  // Salt
            }
            else {
                result.at<uchar>(row, col) = 255;  // Pepper
            }
        }
    }
    return result;
}

// Speckle Noise
cv::Mat speckleNoise(const cv::Mat& image, double noiseStddev) {
    cv::Mat noise = cv::Mat(image.size(), CV_64F);
    randn(noise, 0, noiseStddev);  // Generate Gaussian noise
    cv::Mat result;
    image.convertTo(result, CV_64F);
    result = result + noise;
    result.convertTo(result, image.type());
    return result;
}

// Histogram Equalization
cv::Mat histogramEqualization(const cv::Mat& image) {
    cv::Mat result;
    cv::equalizeHist(image, result);
    return result;
}

// Sharpening
cv::Mat sharpening(const cv::Mat& image) {
    cv::Mat kernel = (cv::Mat_<float>(3, 3) <<
        0, -1, 0,
        -1, 5, -1,
        0, -1, 0);
    cv::Mat result;
    cv::filter2D(image, result, image.depth(), kernel);
    return result;
}

// JPEG Compression
cv::Mat jpegCompression(const cv::Mat& image, int quality) {
    std::vector<int> compression_params = { cv::IMWRITE_JPEG_QUALITY, quality };
    std::vector<uchar> encoded_image;
    cv::imencode(".jpg", image, encoded_image, compression_params);
    return cv::imdecode(encoded_image, cv::IMREAD_GRAYSCALE);
}

// Gaussian Filtering
cv::Mat gaussianFiltering(const cv::Mat& image, int ksize) {
    cv::Mat result;
    cv::GaussianBlur(image, result, cv::Size(ksize, ksize), 0);
    return result;
}

// Median Filtering
cv::Mat medianFiltering(const cv::Mat& image, int ksize) {
    cv::Mat result;
    cv::medianBlur(image, result, ksize);
    return result;
}

// Average Filtering
cv::Mat averageFiltering(const cv::Mat& image, int ksize) {
    cv::Mat result;
    cv::blur(image, result, cv::Size(ksize, ksize));
    return result;
}

// Cropping from Corner
cv::Mat cropFromCorner(const cv::Mat& image, int cropSize) {
    cv::Rect region(0, 0, cropSize, cropSize);
    return image(region);
}

// Cropping from Center
cv::Mat cropFromCenter(const cv::Mat& image, int cropSize) {
    int x = (image.cols - cropSize) / 2;
    int y = (image.rows - cropSize) / 2;
    cv::Rect region(x, y, cropSize, cropSize);
    return image(region);
}

// Cropping from Edge
cv::Mat cropFromEdge(const cv::Mat& image, int cropSize) {
    cv::Rect region(image.cols - cropSize, image.rows - cropSize, cropSize, cropSize);
    return image(region);
}
