
#include "ensemble_classifier.h"
#include <iostream>

int main() {
    try {
        // Пути к моделям и их пороги
        std::vector<std::string> model_paths = {
            "best_scheme_classifier_torchscript.pt",
            "ensemble_model_1_torchscript.pt"
        };
        
        std::vector<float> thresholds = {0.510f, 0.510f};  // Пороги из метаданных
        
        // Создание ансамбля
        EnsembleClassifier ensemble(model_paths, thresholds, true);  // true = использовать CUDA
        
        // Предсказание для одного изображения
        auto result = ensemble.predict("test_image.jpg", true);  // true = использовать TTA
        
        std::cout << "Класс: " << result.class_name << std::endl;
        std::cout << "Уверенность: " << (result.confidence * 100) << "%" << std::endl;
        std::cout << "Scheme_0: " << (result.prob_scheme0 * 100) << "%" << std::endl;
        std::cout << "Scheme_1: " << (result.prob_scheme1 * 100) << "%" << std::endl;
        
        // Пакетная обработка
        std::vector<std::string> image_paths = {"img1.jpg", "img2.jpg", "img3.jpg"};
        auto batch_results = ensemble.predict_batch(image_paths, true);
        
        for (const auto& res : batch_results) {
            std::cout << "Результат: " << res.class_name << " (" << (res.confidence * 100) << "%)" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Ошибка: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
