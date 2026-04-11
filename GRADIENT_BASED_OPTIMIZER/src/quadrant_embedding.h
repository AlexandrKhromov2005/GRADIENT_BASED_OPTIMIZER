#ifndef QUADRANT_EMBEDDING_H
#define QUADRANT_EMBEDDING_H

#include <opencv2/opencv.hpp>
#include <memory>
#include <string>
#include "quadrant_classifier.h"

// Wrapper class for quadrant-based watermark embedding with classifier
class QuadrantEmbedding {
public:
    // Initialize quadrant classifier
    static bool initializeQuadrantClassifier(const std::string& model_path, bool use_cuda = true);

    // Embed watermark into 1024x1024 image with 16 quadrants (pattern: 1-2-1-2/3-4-3-4/1-2-1-2/3-4-3-4)
    // Type 1: NONE, Type 2: JPEG70, Type 3: CONTRAST, Type 4: JPEG80
    // Returns embedded image
    static cv::Mat embedWatermarkQuadrants(const cv::Mat& image_1024x1024,
                                          const cv::Mat& watermark);

    // Embed watermark using bit vector (for dataset generation with random watermarks)
    // Same pattern as above but uses bit vector directly
    static cv::Mat embedWatermarkQuadrants(const cv::Mat& image_1024x1024,
                                          const std::vector<int>& wm_bits);

    // Extract watermark from 1024x1024 image using classifier to select attack type
    // Uses voting from 4 quadrants of selected type
    // Returns extracted watermark
    static cv::Mat extractWatermarkWithClassifier(const cv::Mat& image_1024x1024);

    // Get the initialized classifier (for direct access if needed)
    static std::shared_ptr<QuadrantClassifier> getClassifier();

private:
    static std::shared_ptr<QuadrantClassifier> classifier_;
    static bool classifier_ready_;
};

#endif // QUADRANT_EMBEDDING_H
