#ifndef ATTACK_TYPE_CLASSIFIER_H
#define ATTACK_TYPE_CLASSIFIER_H

#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>

// Classifier for attack type detection on 1024x1024 images
// Model: model_torchscript.pt
// Classes: 0=NoAttack, 1=JPG70, 2=JPG80, 3=ContrastIncrease
class AttackTypeClassifier {
public:
    struct PredictionResult {
        int predicted_class;        // 0-3
        std::string class_name;     // "NoAttack", "JPG70", "JPG80", "ContrastIncrease"
        float confidence;           // max probability
        float prob_noattack;        // Class 0 probability
        float prob_jpg70;           // Class 1 probability
        float prob_jpg80;           // Class 2 probability
        float prob_contrast;        // Class 3 probability

        // Get class name from index
        static std::string getClassName(int class_idx) {
            switch (class_idx) {
                case 0: return "NoAttack";
                case 1: return "JPG70";
                case 2: return "JPG80";
                case 3: return "ContrastIncrease";
                default: return "Unknown";
            }
        }
    };

    // Constructor
    AttackTypeClassifier(const std::string& model_path, bool use_cuda = true);

    // Destructor
    ~AttackTypeClassifier() = default;

    // Predict attack type for a single image
    PredictionResult predict(const cv::Mat& image, bool use_tta = true);

    // Check if model is loaded successfully
    bool isLoaded() const { return model_loaded_; }

private:
    // Preprocess image for inference
    torch::Tensor preprocess_image(const cv::Mat& image);

    // Apply Test-Time Augmentation (horizontal flip)
    torch::Tensor apply_tta(const torch::Tensor& input);

    // Model and device
    torch::jit::script::Module model_;
    torch::Device device_;
    bool model_loaded_;

    // ImageNet normalization parameters
    const std::vector<float> MEAN_{0.485f, 0.456f, 0.406f};
    const std::vector<float> STD_{0.229f, 0.224f, 0.225f};

    // Input dimensions
    const int INPUT_WIDTH_ = 1024;
    const int INPUT_HEIGHT_ = 1024;
};

#endif // ATTACK_TYPE_CLASSIFIER_H
