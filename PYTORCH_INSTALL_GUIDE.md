# 🔧 Установка PyTorch для классификатора

## 📦 Автоматическая установка PyTorch C++

### Linux/Ubuntu:
```bash
# 1. Скачайте PyTorch C++ (CPU версия)
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcpu.zip

# 2. Распакуйте
unzip libtorch-cxx11-abi-shared-with-deps-2.0.0+cpu.zip

# 3. Установите CMAKE_PREFIX_PATH
export CMAKE_PREFIX_PATH=/path/to/libtorch:$CMAKE_PREFIX_PATH

# 4. Пересоберите проект
cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
make -j4
```

### CUDA версия (для GPU ускорения):
```bash
# Для CUDA 11.8
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcu118.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.0.0+cu118.zip

# Сборка с CUDA
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
make -j4
```

## 🚀 Проверка установки

После установки PyTorch у вас будут доступны:

1. **Режим классификатора**:
   ```bash
   ./gradient_based_optimizer --classifier
   ```

2. **Пример интеграции**:
   ```bash
   ./gradient_based_optimizer --example
   ```

3. **Отдельный исполняемый файл**:
   ```bash
   ./classifier_example
   ```

## 📁 Требуемые файлы моделей

Убедитесь, что в корневой директории проекта есть файлы:
- `best_scheme_classifier_torchscript.pt`
- `ensemble_model_1_torchscript.pt`

## ✅ Проверка успешной установки

Если PyTorch установлен правильно, при сборке вы увидите:
```
-- ✅ PyTorch found - classifier integration enabled
-- 🎯 Classifier example executable will be built
```

Если PyTorch не найден:
```
-- ⚠️ PyTorch not found - classifier integration disabled
-- 🔧 Building without classifier integration
```

## 🐛 Устранение проблем

### Ошибка CMake не находит Torch:
```bash
# Проверьте путь к libtorch
ls /path/to/libtorch/share/cmake/Torch/

# Установите точный путь
cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..
```

### Ошибки связывания:
```bash
# Проверьте LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/path/to/libtorch/lib:$LD_LIBRARY_PATH

# Или добавьте в CMakeLists.txt
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
```

### CUDA проблемы:
```bash
# Проверьте версию CUDA
nvcc --version

# Скачайте соответствующую версию PyTorch
# Для CUDA 11.8: cu118
# Для CUDA 12.1: cu121
```

## 🎯 Режимы работы с классификатором

После установки PyTorch доступны:

1. **`--classifier`** - Генерация датасета с автоматическим выбором схем
2. **`--example`** - Демонстрация работы классификатора
3. **`classifier_example`** - Подробное тестирование интеграции

Без PyTorch работают стандартные режимы:
- **`--dataset`** - Обычная генерация датасета
- **`--test`** - Тестовый режим
- **По умолчанию** - Основной режим работы