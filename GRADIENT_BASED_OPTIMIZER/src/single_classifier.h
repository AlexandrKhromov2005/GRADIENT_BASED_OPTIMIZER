#ifndef SINGLE_CLASSIFIER_H
#define SINGLE_CLASSIFIER_H

#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <memory>

class SingleClassifier {
private:
    torch::jit::script::Module model_;
    float threshold_;
    torch::Device device_;
    bool model_loaded_;
    
    // Preprocessing parameters
    const std::vector<float> MEAN_{0.485f};
    const std::vector<float> STD_{0.229f};
    const int INPUT_WIDTH_ = 192;
    const int INPUT_HEIGHT_ = 192;
    
public:
    struct PredictionResult {
        int predicted_class;
        std::string class_name;
        float confidence;
        float prob_scheme0;
        float prob_scheme1;
        float threshold;
    };
    
    SingleClassifier(const std::string& model_path,
                    float threshold = 0.5f,
                    bool use_cuda = true);
    
    ~SingleClassifier() = default;
    
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

#endif // SINGLE_CLASSIFIER_H