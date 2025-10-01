#include "single_classifier.h"
#include <iostream>
#include <algorithm>
#include <numeric>

SingleClassifier::SingleClassifier(const std::string& model_path,
                                 float threshold,
                                 bool use_cuda)
    : device_(use_cuda && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
      threshold_(threshold),
      model_loaded_(false) {
    
    std::cout << "🚀 Инициализация SingleClassifier..." << std::endl;
    std::cout << "📱 Устройство: " << (device_.is_cuda() ? "CUDA" : "CPU") << std::endl;
    
    // Загружаем модель
    try {
        model_ = torch::jit::load(model_path, device_);
        model_.eval();
        model_loaded_ = true;
        
        std::cout << "✅ Загружена модель: " << model_path 
                  << " (порог: " << threshold_ << ")" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "❌ Ошибка загрузки " << model_path << ": " << e.what() << std::endl;
        model_loaded_ = false;
    }
}

torch::Tensor SingleClassifier::preprocess_image(const cv::Mat& image) {
    // Конвертация в градации серого если нужно
    cv::Mat gray_image;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
    } else {
        gray_image = image.clone();
    }
    
    // Изменение размера
    cv::Mat resized;
    cv::resize(gray_image, resized, cv::Size(INPUT_WIDTH_, INPUT_HEIGHT_));
    
    // Конвертация в тензор
    resized.convertTo(resized, CV_32F, 1.0/255.0);  // Нормализация 0-1
    
    auto tensor = torch::from_blob(resized.data, {1, 1, INPUT_HEIGHT_, INPUT_WIDTH_}, torch::kFloat);
    tensor = tensor.to(device_);
    
    // Нормализация
    tensor = (tensor - MEAN_[0]) / STD_[0];
    
    return tensor;
}

torch::Tensor SingleClassifier::apply_tta(const torch::Tensor& input) {
    std::vector<torch::Tensor> predictions;
    
    // Оригинальное изображение
    auto pred = torch::softmax(model_.forward({input}).toTensor(), 1);
    predictions.push_back(pred);
    
    // Горизонтальный поворот
    auto flipped = torch::flip(input, {3});
    auto pred_flip = torch::softmax(model_.forward({flipped}).toTensor(), 1);
    predictions.push_back(pred_flip);
    
    // Усреднение TTA
    auto stacked = torch::stack(predictions, 0);
    return torch::mean(stacked, 0);
}

cv::Mat SingleClassifier::load_and_resize_image(const std::string& image_path) {
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        throw std::runtime_error("Не удалось загрузить изображение: " + image_path);
    }
    return image;
}

SingleClassifier::PredictionResult SingleClassifier::predict(const cv::Mat& image, bool use_tta) {
    if (!model_loaded_) {
        throw std::runtime_error("Модель не загружена!");
    }
    
    // Предобработка
    auto input_tensor = preprocess_image(image);
    
    // Получаем предсказание
    torch::Tensor pred;
    
    if (use_tta) {
        pred = apply_tta(input_tensor);
    } else {
        pred = torch::softmax(model_.forward({input_tensor}).toTensor(), 1);
    }
    
    // Извлекаем данные предсказания
    auto pred_data = pred.cpu().contiguous();
    float* pred_ptr = pred_data.data_ptr<float>();
    
    // Результат
    PredictionResult result;
    result.prob_scheme0 = pred_ptr[0];
    result.prob_scheme1 = pred_ptr[1];
    result.predicted_class = (result.prob_scheme1 > threshold_) ? 1 : 0;
    result.class_name = (result.predicted_class == 0) ? "scheme_0" : "scheme_1";
    result.confidence = (result.predicted_class == 0) ? result.prob_scheme0 : result.prob_scheme1;
    result.threshold = threshold_;
    
    return result;
}

SingleClassifier::PredictionResult SingleClassifier::predict(const std::string& image_path, bool use_tta) {
    cv::Mat image = load_and_resize_image(image_path);
    return predict(image, use_tta);
}

std::vector<SingleClassifier::PredictionResult> 
SingleClassifier::predict_batch(const std::vector<std::string>& image_paths, bool use_tta) {
    std::vector<PredictionResult> results;
    results.reserve(image_paths.size());
    
    for (const auto& path : image_paths) {
        try {
            auto result = predict(path, use_tta);
            results.push_back(result);
            
            std::cout << "✅ " << path << ": " << result.class_name 
                      << " (" << (result.confidence * 100) << "%)" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "❌ Ошибка для " << path << ": " << e.what() << std::endl;
            // Добавляем пустой результат или пропускаем
        }
    }
    
    return results;
}