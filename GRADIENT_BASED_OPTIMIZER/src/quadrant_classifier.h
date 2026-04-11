#ifndef QUADRANT_CLASSIFIER_H
#define QUADRANT_CLASSIFIER_H

#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <memory>

class QuadrantClassifier {
private:
    torch::jit::script::Module model_;
    torch::Device device_;
    bool model_loaded_;

    // Preprocessing parameters (1024x1024 RGB input for UltraHighResForensicNet)
    // ImageNet normalization
    const std::vector<float> MEAN_{0.485f, 0.456f, 0.406f};
    const std::vector<float> STD_{0.229f, 0.224f, 0.225f};
    const int INPUT_WIDTH_ = 1024;   // Updated to 1024 for best_model_ultrahighres.pt
    const int INPUT_HEIGHT_ = 1024;  // Updated to 1024 for best_model_ultrahighres.pt

public:
    struct PredictionResult {
        int predicted_class;      // 0, 1, 2, or 3
        std::string class_name;   // ai_generated, authentic, heavily_edited, lightly_edited
        float confidence;
        float prob_scheme0;       // Probability for class 0 (ai_generated)
        float prob_scheme1;       // Probability for class 1 (authentic)
        float prob_scheme2;       // Probability for class 2 (heavily_edited)
        float prob_scheme3;       // Probability for class 3 (lightly_edited)

        // Attack type mapping (for watermark extraction)
        // We map forensic classes to attack types:
        // class 0 (ai_generated) -> NONE attack
        // class 1 (authentic) -> JPEG70 attack
        // class 2 (heavily_edited) -> CONTRAST attack
        // class 3 (lightly_edited) -> JPEG80 attack
        int getAttackType() const { return predicted_class; }
    };

    QuadrantClassifier(const std::string& model_path, bool use_cuda = true);

    ~QuadrantClassifier() = default;

    PredictionResult predict(const cv::Mat& image, bool use_tta = true);
    PredictionResult predict(const std::string& image_path, bool use_tta = true);

    std::vector<PredictionResult> predict_batch(const std::vector<std::string>& image_paths,
                                              bool use_tta = true);

    bool isLoaded() const { return model_loaded_; }

private:
    torch::Tensor preprocess_image(const cv::Mat& image);
    torch::Tensor apply_tta(const torch::Tensor& input);
    cv::Mat load_and_resize_image(const std::string& image_path);
};

#endif // QUADRANT_CLASSIFIER_H
