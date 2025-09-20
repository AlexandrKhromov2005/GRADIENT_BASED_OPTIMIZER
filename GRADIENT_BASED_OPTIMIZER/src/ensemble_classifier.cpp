
#include "ensemble_classifier.h"
#include <iostream>
#include <algorithm>
#include <numeric>

EnsembleClassifier::EnsembleClassifier(const std::vector<std::string>& model_paths,
                                     const std::vector<float>& thresholds,
                                     bool use_cuda)
    : device_(use_cuda && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
      thresholds_(thresholds) {
    
    std::cout << "🚀 Инициализация EnsembleClassifier..." << std::endl;
    std::cout << "📱 Устройство: " << (device_.is_cuda() ? "CUDA" : "CPU") << std::endl;
    
    // Загружаем все модели
    for (size_t i = 0; i < model_paths.size(); ++i) {
        try {
            auto model = torch::jit::load(model_paths[i], device_);
            model.eval();
            models_.push_back(std::move(model));
            
            std::cout << "✅ Загружена модель: " << model_paths[i] 
                      << " (порог: " << thresholds_[i] << ")" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "❌ Ошибка загрузки " << model_paths[i] << ": " << e.what() << std::endl;
        }
    }
    
    std::cout << "🎯 Загружено " << models_.size() << " моделей" << std::endl;
}

torch::Tensor EnsembleClassifier::preprocess_image(const cv::Mat& image) {
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

torch::Tensor EnsembleClassifier::apply_tta(const torch::Tensor& input, torch::jit::script::Module& model) {
    std::vector<torch::Tensor> predictions;
    
    // Оригинальное изображение
    auto pred = torch::softmax(model.forward({input}).toTensor(), 1);
    predictions.push_back(pred);
    
    // Горизонтальный поворот
    auto flipped = torch::flip(input, {3});
    auto pred_flip = torch::softmax(model.forward({flipped}).toTensor(), 1);
    predictions.push_back(pred_flip);
    
    // Усреднение TTA
    auto stacked = torch::stack(predictions, 0);
    return torch::mean(stacked, 0);
}

cv::Mat EnsembleClassifier::load_and_resize_image(const std::string& image_path) {
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        throw std::runtime_error("Не удалось загрузить изображение: " + image_path);
    }
    return image;
}

EnsembleClassifier::PredictionResult EnsembleClassifier::predict(const cv::Mat& image, bool use_tta) {
    if (models_.empty()) {
        throw std::runtime_error("Нет загруженных моделей!");
    }
    
    // Предобработка
    auto input_tensor = preprocess_image(image);
    
    // Получаем предсказания от всех моделей
    std::vector<torch::Tensor> all_predictions;
    std::vector<std::vector<float>> individual_preds;
    
    for (size_t i = 0; i < models_.size(); ++i) {
        torch::Tensor pred;
        
        if (use_tta) {
            pred = apply_tta(input_tensor, models_[i]);
        } else {
            pred = torch::softmax(models_[i].forward({input_tensor}).toTensor(), 1);
        }
        
        all_predictions.push_back(pred);
        
        // Сохраняем индивидуальные предсказания
        auto pred_data = pred.cpu().contiguous();
        float* pred_ptr = pred_data.data_ptr<float>();
        individual_preds.push_back({pred_ptr[0], pred_ptr[1]});
    }
    
    // Усредняем предсказания ансамбля
    auto ensemble_pred = torch::mean(torch::stack(all_predictions, 0), 0);
    auto pred_data = ensemble_pred.cpu().contiguous();
    float* pred_ptr = pred_data.data_ptr<float>();
    
    // Вычисляем средний порог
    float avg_threshold = std::accumulate(thresholds_.begin(), thresholds_.end(), 0.0f) / thresholds_.size();
    
    // Результат
    PredictionResult result;
    result.prob_scheme0 = pred_ptr[0];
    result.prob_scheme1 = pred_ptr[1];
    result.predicted_class = (result.prob_scheme1 > avg_threshold) ? 1 : 0;
    result.class_name = (result.predicted_class == 0) ? "scheme_0" : "scheme_1";
    result.confidence = (result.predicted_class == 0) ? result.prob_scheme0 : result.prob_scheme1;
    result.threshold = avg_threshold;
    result.ensemble_size = static_cast<int>(models_.size());
    result.individual_predictions = individual_preds;
    
    return result;
}

EnsembleClassifier::PredictionResult EnsembleClassifier::predict(const std::string& image_path, bool use_tta) {
    cv::Mat image = load_and_resize_image(image_path);
    return predict(image, use_tta);
}

std::vector<EnsembleClassifier::PredictionResult> 
EnsembleClassifier::predict_batch(const std::vector<std::string>& image_paths, bool use_tta) {
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
