#include "quadrant_classifier.h"
#include <iostream>
#include <algorithm>
#include <numeric>

QuadrantClassifier::QuadrantClassifier(const std::string& model_path, bool use_cuda)
    : device_(use_cuda && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
      model_loaded_(false) {

    std::cout << "🚀 Initializing QuadrantClassifier..." << std::endl;
    std::cout << "📱 Device: " << (device_.is_cuda() ? "CUDA" : "CPU") << std::endl;

    // Load model
    try {
        model_ = torch::jit::load(model_path, device_);
        model_.eval();
        model_loaded_ = true;

        std::cout << "✅ Loaded model: " << model_path << std::endl;
        std::cout << "🎯 HighResForensicNet: 384x384 RGB input, 4 classes" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "❌ Error loading " << model_path << ": " << e.what() << std::endl;
        model_loaded_ = false;
    }
}

torch::Tensor QuadrantClassifier::preprocess_image(const cv::Mat& image) {
    // Convert BGR to RGB (OpenCV loads as BGR)
    cv::Mat rgb_image;
    if (image.channels() == 3) {
        cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);
    } else if (image.channels() == 1) {
        // Convert grayscale to RGB
        cv::cvtColor(image, rgb_image, cv::COLOR_GRAY2RGB);
    } else {
        rgb_image = image.clone();
    }

    // Resize to 384x384
    cv::Mat resized;
    cv::resize(rgb_image, resized, cv::Size(INPUT_WIDTH_, INPUT_HEIGHT_));

    // Convert to float and normalize to [0, 1]
    resized.convertTo(resized, CV_32FC3, 1.0/255.0);

    // Split channels and apply ImageNet normalization
    std::vector<cv::Mat> channels(3);
    cv::split(resized, channels);

    for (int c = 0; c < 3; c++) {
        channels[c] = (channels[c] - MEAN_[c]) / STD_[c];
    }

    cv::Mat normalized;
    cv::merge(channels, normalized);

    // Create tensor from normalized image
    // OpenCV format: [height, width, channels]
    // PyTorch format: [batch, channels, height, width]
    auto tensor = torch::from_blob(
        normalized.data,
        {1, INPUT_HEIGHT_, INPUT_WIDTH_, 3},
        torch::kFloat32
    );

    // Permute from [1, H, W, C] to [1, C, H, W]
    tensor = tensor.permute({0, 3, 1, 2}).contiguous();
    tensor = tensor.to(device_);

    return tensor;
}

torch::Tensor QuadrantClassifier::apply_tta(const torch::Tensor& input) {
    std::vector<torch::Tensor> predictions;

    // Original image
    auto pred = torch::softmax(model_.forward({input}).toTensor(), 1);
    predictions.push_back(pred);

    // Horizontal flip
    auto flipped = torch::flip(input, {3});
    auto pred_flip = torch::softmax(model_.forward({flipped}).toTensor(), 1);
    predictions.push_back(pred_flip);

    // Average TTA
    auto stacked = torch::stack(predictions, 0);
    return torch::mean(stacked, 0);
}

cv::Mat QuadrantClassifier::load_and_resize_image(const std::string& image_path) {
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        throw std::runtime_error("Failed to load image: " + image_path);
    }
    return image;
}

QuadrantClassifier::PredictionResult QuadrantClassifier::predict(const cv::Mat& image, bool use_tta) {
    if (!model_loaded_) {
        throw std::runtime_error("Model not loaded!");
    }

    // Preprocess
    auto input_tensor = preprocess_image(image);

    // Get prediction
    torch::Tensor pred;

    if (use_tta) {
        pred = apply_tta(input_tensor);
    } else {
        pred = torch::softmax(model_.forward({input_tensor}).toTensor(), 1);
    }

    // Extract prediction data
    auto pred_data = pred.cpu().contiguous();
    float* pred_ptr = pred_data.data_ptr<float>();

    // Result
    PredictionResult result;
    result.prob_scheme0 = pred_ptr[0];
    result.prob_scheme1 = pred_ptr[1];
    result.prob_scheme2 = pred_ptr[2];
    result.prob_scheme3 = pred_ptr[3];

    // Find class with maximum probability
    std::vector<float> probs = {result.prob_scheme0, result.prob_scheme1,
                                result.prob_scheme2, result.prob_scheme3};
    result.predicted_class = std::max_element(probs.begin(), probs.end()) - probs.begin();
    result.confidence = probs[result.predicted_class];

    // Map to class name
    switch (result.predicted_class) {
        case 0: result.class_name = "scheme_0"; break;
        case 1: result.class_name = "scheme_1"; break;
        case 2: result.class_name = "scheme_2"; break;
        case 3: result.class_name = "scheme_3"; break;
        default: result.class_name = "unknown"; break;
    }

    return result;
}

QuadrantClassifier::PredictionResult QuadrantClassifier::predict(const std::string& image_path, bool use_tta) {
    cv::Mat image = load_and_resize_image(image_path);
    return predict(image, use_tta);
}

std::vector<QuadrantClassifier::PredictionResult>
QuadrantClassifier::predict_batch(const std::vector<std::string>& image_paths, bool use_tta) {
    std::vector<PredictionResult> results;
    results.reserve(image_paths.size());

    for (const auto& path : image_paths) {
        try {
            auto result = predict(path, use_tta);
            results.push_back(result);

            std::cout << "✅ " << path << ": " << result.class_name
                      << " (" << (result.confidence * 100) << "%)" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "❌ Error for " << path << ": " << e.what() << std::endl;
        }
    }

    return results;
}
