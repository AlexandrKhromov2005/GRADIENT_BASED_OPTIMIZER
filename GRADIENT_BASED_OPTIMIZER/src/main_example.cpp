
#include "single_classifier.h"
#include <iostream>

int main() {
    try {
        // Путь к модели и её порог
        std::string model_path = "final_model.pt";
        float threshold = 0.5f;
        
        // Создание single classifier
        SingleClassifier classifier(model_path, threshold, true);  // true = использовать CUDA
        
        // Предсказание для одного изображения
        auto result = classifier.predict("test_image.jpg", true);  // true = использовать TTA
        
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
