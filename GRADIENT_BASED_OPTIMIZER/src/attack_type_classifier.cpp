#include "attack_type_classifier.h"
#include <iostream>

AttackTypeClassifier::AttackTypeClassifier(const std::string& model_path, bool use_cuda)
    : device_(use_cuda && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
      model_loaded_(false) {

    try {
        std::cout << "🔄 Loading AttackTypeClassifier model: " << model_path << std::endl;

        // Load TorchScript model
        model_ = torch::jit::load(model_path, device_);
        model_.eval();

        model_loaded_ = true;

        std::cout << "✅ AttackTypeClassifier loaded successfully" << std::endl;
        std::cout << "   Device: " << (device_.is_cuda() ? "CUDA" : "CPU") << std::endl;
        std::cout << "   Input size: " << INPUT_WIDTH_ << "x" << INPUT_HEIGHT_ << std::endl;
        std::cout << "   Classes: NoAttack, JPG70, JPG80, ContrastIncrease" << std::endl;

    } catch (const c10::Error& e) {
        std::cerr << "❌ Error loading AttackTypeClassifier model: " << e.what() << std::endl;
        model_loaded_ = false;
    }
}

torch::Tensor AttackTypeClassifier::preprocess_image(const cv::Mat& image) {
    cv::Mat rgb_image;

    // Convert to RGB if needed
    if (image.channels() == 1) {
        // Grayscale -> RGB (replicate channel)
        cv::cvtColor(image, rgb_image, cv::COLOR_GRAY2RGB);
    } else if (image.channels() == 3) {
        // BGR -> RGB
        cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);
    } else {
        rgb_image = image.clone();
    }

    // Resize to 1024x1024
    cv::Mat resized;
    if (rgb_image.rows != INPUT_HEIGHT_ || rgb_image.cols != INPUT_WIDTH_) {
        cv::resize(rgb_image, resized, cv::Size(INPUT_WIDTH_, INPUT_HEIGHT_));
    } else {
        resized = rgb_image;
    }

    // Convert to float [0, 1]
    resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);

    // Split channels for normalization
    std::vector<cv::Mat> channels(3);
    cv::split(resized, channels);

    // Apply ImageNet normalization per channel
    for (int c = 0; c < 3; c++) {
        channels[c] = (channels[c] - MEAN_[c]) / STD_[c];
    }

    // Merge back
    cv::Mat normalized;
    cv::merge(channels, normalized);

    // Convert to tensor [1, H, W, C]
    auto tensor = torch::from_blob(
        normalized.data,
        {1, INPUT_HEIGHT_, INPUT_WIDTH_, 3},
        torch::kFloat32
    ).clone();

    // Permute to [1, C, H, W] (PyTorch format)
    tensor = tensor.permute({0, 3, 1, 2}).contiguous();

    // Move to device
    tensor = tensor.to(device_);

    return tensor;
}

torch::Tensor AttackTypeClassifier::apply_tta(const torch::Tensor& input) {
    std::vector<torch::Tensor> predictions;

    // Original image
    auto output_original = model_.forward({input}).toTensor();
    auto pred_original = torch::softmax(output_original, 1);
    predictions.push_back(pred_original);

    // Horizontal flip
    auto flipped = torch::flip(input, {3});  // Flip along width dimension
    auto output_flip = model_.forward({flipped}).toTensor();
    auto pred_flip = torch::softmax(output_flip, 1);
    predictions.push_back(pred_flip);

    // Average predictions
    auto stacked = torch::stack(predictions, 0);
    auto averaged = torch::mean(stacked, 0);

    return averaged;
}

AttackTypeClassifier::PredictionResult AttackTypeClassifier::predict(
    const cv::Mat& image, bool use_tta) {

    PredictionResult result;

    if (!model_loaded_) {
        std::cerr << "❌ Model not loaded, cannot predict" << std::endl;
        result.predicted_class = -1;
        result.class_name = "Error";
        result.confidence = 0.0f;
        return result;
    }

    try {
        // Preprocess image
        auto input_tensor = preprocess_image(image);

        // Forward pass with optional TTA
        torch::Tensor probabilities;
        {
            torch::NoGradGuard no_grad;
            if (use_tta) {
                probabilities = apply_tta(input_tensor);
            } else {
                auto output = model_.forward({input_tensor}).toTensor();
                probabilities = torch::softmax(output, 1);
            }
        }

        // Move to CPU and get data
        probabilities = probabilities.to(torch::kCPU);
        auto probs_accessor = probabilities.accessor<float, 2>();

        // Extract probabilities
        result.prob_noattack = probs_accessor[0][0];
        result.prob_jpg70 = probs_accessor[0][1];
        result.prob_jpg80 = probs_accessor[0][2];
        result.prob_contrast = probs_accessor[0][3];

        // Find predicted class (argmax)
        auto max_prob = probabilities.argmax(1);
        result.predicted_class = max_prob.item<int>();

        // Get class name
        result.class_name = PredictionResult::getClassName(result.predicted_class);

        // Confidence is the max probability
        result.confidence = std::max({
            result.prob_noattack,
            result.prob_jpg70,
            result.prob_jpg80,
            result.prob_contrast
        });

    } catch (const c10::Error& e) {
        std::cerr << "❌ Error during prediction: " << e.what() << std::endl;
        result.predicted_class = -1;
        result.class_name = "Error";
        result.confidence = 0.0f;
    }

    return result;
}
