# 🚀 Полная инструкция по запуску эксперимента с final_model.pt

## 📋 Предварительные требования

### Системные требования:
- Linux система с g++ и cmake
- OpenCV библиотеки
- PyTorch C++ (libtorch) - устанавливается автоматически
- Минимум 2GB свободного места

### Проверка системы:
```bash
# Проверка компилятора
g++ --version

# Проверка cmake
cmake --version

# Проверка OpenCV
pkg-config --modversion opencv4
```

## 🛠️ Подготовка

### 1. Переход в директорию проекта:
```bash
cd /home/alex/projects/GRADIENT_BASED_OPTIMIZER
```

### 2. Проверка файлов:
```bash
# Проверка модели
ls -la final_model*.pt

# Проверка изображений
ls images/*.png | wc -l

# Проверка скриптов
ls -la *.sh
```

### 3. Установка прав на выполнение:
```bash
chmod +x run_classifier_experiment.sh
chmod +x clean_experiment_results.sh
```

## 🎯 Запуск экспериментов

### Автоматический запуск (рекомендуется):
```bash
./run_classifier_experiment.sh
```

Скрипт предложит:
1. **Быстрый тест** (~2-5 минут) - для проверки работоспособности
2. **Полный эксперимент** (~15-30 минут) - все изображения
3. **Ограниченный набор** (~5-10 минут) - 5 изображений
4. **Пользовательский режим** - ручная настройка

### Ручной запуск:

#### Быстрый тест:
```bash
./build/gradient_based_optimizer --dataset-classifier --test
```

#### Полный эксперимент:
```bash
./build/gradient_based_optimizer --dataset-classifier
```

#### Только сборка без запуска:
```bash
mkdir -p build && cd build
cmake ..
make gradient_based_optimizer -j$(nproc)
cd ..
```

## 📊 Анализ результатов

### Автоматический анализ:
```bash
# Результаты показываются после завершения эксперимента
# Или можно запустить анализ вручную:

scheme2=$(find dataset_classifier/scheme2_selected -name "*.png" | wc -l)
scheme3=$(find dataset_classifier/scheme3_selected -name "*.png" | wc -l)
correct=$(find dataset_classifier/extraction_correct -name "*.png" | wc -l)
incorrect=$(find dataset_classifier/extraction_incorrect -name "*.png" | wc -l)

echo "Scheme2: $scheme2, Scheme3: $scheme3"
echo "Правильно: $correct, Неправильно: $incorrect"
echo "Точность: $(echo "scale=2; $correct * 100 / ($correct + $incorrect)" | bc -l)%"
```

### Структура результатов:
```
dataset_classifier/
├── scheme2_selected/      # Блоки, классифицированные как Scheme2
├── scheme3_selected/      # Блоки, классифицированные как Scheme3
├── extraction_correct/    # Блоки с правильным извлечением
└── extraction_incorrect/  # Блоки с ошибками извлечения
```

## 🧹 Очистка результатов

### Интерактивная очистка:
```bash
./clean_experiment_results.sh
```

### Быстрая очистка только результатов классификатора:
```bash
rm -rf dataset_classifier/
```

### Полная очистка всех результатов:
```bash
rm -rf dataset/ dataset_classifier/ images_backup_* images_quick/ images_full/
```

## ⚡ Быстрые команды

### Полный цикл "очистка → эксперимент → анализ":
```bash
# Очистка
rm -rf dataset_classifier/

# Запуск быстрого теста
./build/gradient_based_optimizer --dataset-classifier --test

# Анализ
echo "Результаты:"
find dataset_classifier/ -name "*.png" | wc -l
```

### Мониторинг выполнения:
```bash
# В другом терминале, пока идет эксперимент:
watch -n 5 'find dataset_classifier/ -name "*.png" | wc -l'
```

## 🔧 Решение проблем

### Ошибка "PyTorch not found":
```bash
cd /tmp
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcpu.zip
```

### Ошибка "final_model_torchscript.pt not found":
```bash
cp best_scheme_classifier_torchscript.pt final_model_torchscript.pt
```

### Ошибка компиляции:
```bash
# Очистка сборки
rm -rf build/
mkdir build && cd build
cmake ..
make clean
make gradient_based_optimizer -j$(nproc)
```

### Ошибка "images not found":
```bash
# Проверка изображений
ls images/*.png
# Должно показать минимум watermark.png и другие изображения
```

## 📈 Интерпретация результатов

### Хорошие показатели:
- **Точность > 95%** - отличный результат
- **Точность 85-95%** - хороший результат  
- **Точность < 85%** - требует анализа

### Анализ выбора схем:
- **Balanced (50/50)** - классификатор различает типы блоков
- **Biased (90/10)** - модель имеет предпочтения
- **Extreme (100/0)** - возможно, нужна калибровка

### Файлы с результатами:
- `EXPERIMENT_RESULTS.md` - детальный анализ
- `FINAL_EXPERIMENT_SUMMARY.md` - итоговый отчет
- `dataset_classifier/` - все данные эксперимента

## 🎉 Успешное завершение

После успешного эксперимента вы получите:
1. ✅ Высокую точность извлечения (>95%)
2. 📊 Детальную статистику по схемам
3. 📁 Все промежуточные результаты
4. 📋 Готовые отчеты для анализа

**Время выполнения:**
- Быстрый тест: 2-5 минут
- Ограниченный (5 изображений): 5-10 минут  
- Полный эксперимент: 15-30 минут

**Готово к использованию!** 🚀