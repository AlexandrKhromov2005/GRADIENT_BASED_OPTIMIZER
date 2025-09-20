# C++ Integration Guide for Ensemble Classifier

## 📋 Файлы проекта:
- `ensemble_classifier.h` - заголовочный файл
- `ensemble_classifier.cpp` - реализация класса  
- `main_example.cpp` - пример использования
- `CMakeLists.txt` - файл сборки

## 🔧 Требования:
- C++14 или новее
- PyTorch C++ (libtorch)
- OpenCV 4.x
- CUDA (опционально)

## 📦 Установка зависимостей:

### Ubuntu/Debian:
```bash
# OpenCV
sudo apt update
sudo apt install libopencv-dev

# PyTorch C++
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcu118.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.0.0+cu118.zip
```

### Windows:
1. Установите OpenCV через vcpkg или скачайте готовую сборку
2. Скачайте libtorch с официального сайта PyTorch

## 🚀 Сборка:
```bash
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
make -j4
```

## 💡 Использование:
```cpp
#include "ensemble_classifier.h"

// Создание ансамбля
std::vector<std::string> models = {"best_scheme_classifier_torchscript.pt", "ensemble_model_1_torchscript.pt"};
std::vector<float> thresholds = {0.510f, 0.510f};
EnsembleClassifier ensemble(models, thresholds, true);

// Предсказание
auto result = ensemble.predict("image.jpg", true);
std::cout << "Класс: " << result.class_name << std::endl;
```

## 🎯 Производительность:
- Одна модель: ~20-50ms
- Ансамбль (2 модели): ~40-100ms  
- TTA добавляет ~2x времени
- GPU ускоряет в ~3-5 раз

## 📊 Преобразованные модели:
- best_scheme_classifier_torchscript.pt (порог: 0.5100000000000001)
- ensemble_model_1_torchscript.pt (порог: 0.5100000000000001)

## 🐛 Отладка:
1. Проверьте пути к моделям
2. Убедитесь что CUDA доступна (если используете)
3. Проверьте размер входного изображения (должно быть любое, автоматически изменится до 192x192)
4. Проверьте формат изображения (поддерживаются стандартные форматы OpenCV)
