#ifndef EMBEDDING_WITH_CLASSIFIER_H
#define EMBEDDING_WITH_CLASSIFIER_H

#include <opencv2/opencv.hpp>
#include "embedding_schemes.h"
#include "gbo.h"

class EmbeddingWithClassifier {
public:
    // Initialize ensemble classifier
    static bool initializeClassifier(const std::vector<std::string>& model_paths,
                                   const std::vector<float>& thresholds,
                                   bool use_cuda = true);
    
    // Initialize single classifier
    static bool initializeSingleClassifier(const std::string& model_path,
                                         float threshold = 0.5f,
                                         bool use_cuda = true);
    
    // Embed a bit into an 8x8 block using classifier to select scheme
    static cv::Mat embedBitWithSchemeSelection(const cv::Mat& block_8x8, uchar bit_to_embed);
    
    // Extract a bit from an 8x8 block using classifier to predict scheme
    static uchar extractBitWithSchemePrediction(const cv::Mat& block_8x8);
    
private:
    static bool classifier_ready_;
};

#endif // EMBEDDING_WITH_CLASSIFIER_H