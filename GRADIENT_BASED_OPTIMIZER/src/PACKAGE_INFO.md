# 📦 Пакет для C++ интеграции ансамбля нейронных сетей

## 📁 Содержимое папки:

### 🤖 **Модели PyTorch (.pt для C++)**
- `best_scheme_classifier_torchscript.pt` - основная модель (11.0 MB, точность 73.8%)
- `ensemble_model_1_torchscript.pt` - вторая модель для ансамбля (11.1 MB, точность 73.8%)

### 📋 **Метаданные моделей**  
- `best_scheme_classifier_metadata.txt` - параметры первой модели
- `ensemble_model_1_metadata.txt` - параметры второй модели

### 💻 **C++ исходный код**
- `ensemble_classifier.h` - заголовочный файл класса
- `ensemble_classifier.cpp` - реализация класса EnsembleClassifier  
- `main_example.cpp` - пример использования

### 🔧 **Сборка проекта**
- `CMakeLists.txt` - файл конфигурации CMake
- `README_CPP.md` - подробная инструкция по установке и использованию

### 📄 **Научная документация**
- `scientific_ensemble_description.md` - полное научное описание ансамбля

---

## 🚀 Быстрый старт:

### 1. Установка зависимостей:
```bash
# Ubuntu/Debian
sudo apt install libopencv-dev

# Скачать libtorch
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcu118.zip
unzip libtorch-*.zip
```

### 2. Сборка:
```bash
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
make -j4
```

### 3. Использование:
```cpp
#include "ensemble_classifier.h"

std::vector<std::string> models = {
    "best_scheme_classifier_torchscript.pt",
    "ensemble_model_1_torchscript.pt"
};
std::vector<float> thresholds = {0.510f, 0.510f};

EnsembleClassifier ensemble(models, thresholds, true);
auto result = ensemble.predict("image.jpg", true);

std::cout << "Класс: " << result.class_name << std::endl;
std::cout << "Уверенность: " << (result.confidence * 100) << "%" << std::endl;
```

---

## 📊 Характеристики:

| Параметр | Значение |
|----------|----------|
| **Точность одной модели** | 73.8% |
| **Точность ансамбля** | 75.2% |
| **Время инференса** | 50ms (1 модель), 200ms (ансамбль+TTA) |
| **Размер входа** | 192×192 пикселей, grayscale |
| **Классы** | scheme_0, scheme_1 |
| **Порог** | 0.510 |

---

## ✅ Что включено:
- ✅ Готовые модели в формате TorchScript (.pt)
- ✅ Полный C++ код с документацией
- ✅ Система сборки CMake  
- ✅ Примеры использования
- ✅ Научное описание методологии
- ✅ Поддержка GPU/CUDA
- ✅ Test Time Augmentation
- ✅ Batch processing

**Размер архива**: ~23 MB  
**Требования**: C++14, PyTorch C++, OpenCV  
**Поддержка**: Linux, Windows, macOS