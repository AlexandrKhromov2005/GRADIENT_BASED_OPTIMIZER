#ifndef ATTACK_TYPE_EMBEDDING_H
#define ATTACK_TYPE_EMBEDDING_H

#include <opencv2/opencv.hpp>
#include <memory>
#include <string>
#include "attack_type_classifier.h"

// Wrapper class for watermark embedding/extraction with attack type classification
// Uses model_torchscript.pt to classify attack type and select best quadrant for extraction
class AttackTypeEmbedding {
public:
    // Initialize attack type classifier
    static bool initializeAttackClassifier(const std::string& model_path, bool use_cuda = true);

    // Embed watermark into 1024x1024 image with 16 quadrants
    // Same pattern as QuadrantEmbedding: NONE/JPEG70/NONE/JPEG70 / CONTRAST/JPEG80/CONTRAST/JPEG80
    static cv::Mat embedWatermarkQuadrants(const cv::Mat& image_1024x1024,
                                          const cv::Mat& watermark);

    // Embed using bit vector (for random watermarks)
    static cv::Mat embedWatermarkQuadrants(const cv::Mat& image_1024x1024,
                                          const std::vector<int>& wm_bits);

    // Extract watermark using attack type classifier
    // Classifier determines which quadrant type to use based on predicted attack
    static cv::Mat extractWatermarkWithClassifier(const cv::Mat& image_1024x1024);

    // Get the initialized classifier (for direct access)
    static std::shared_ptr<AttackTypeClassifier> getClassifier();

private:
    static std::shared_ptr<AttackTypeClassifier> classifier_;
    static bool classifier_ready_;
};

#endif // ATTACK_TYPE_EMBEDDING_H
