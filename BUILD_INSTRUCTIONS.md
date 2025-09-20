# 🔧 Инструкция по сборке проекта

## 📋 Быстрая сборка (выберите один вариант)

### 🟢 Базовая сборка (БЕЗ классификатора)
```bash
mkdir build && cd build
cmake ..
make -j4
```

### 🤖 Полная сборка (С классификатором)
```bash
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=~/libtorch ..
make -j4
```

## 🎯 Результат сборки

### При базовой сборке:
- ✅ `gradient_based_optimizer` - основной исполняемый файл
- ⚠️ Режимы `--classifier` и `--example` недоступны

### При полной сборке:
- ✅ `gradient_based_optimizer` - основной исполняемый файл
- ✅ `classifier_example` - тестирование классификатора
- ✅ Все режимы доступны

## 🚀 Использование

### Базовые режимы (работают всегда):
```bash
./build/gradient_based_optimizer --help           # Справка
./build/gradient_based_optimizer --test           # Тест (1 итерация)
./build/gradient_based_optimizer --dataset        # Генерация датасета
./build/gradient_based_optimizer                  # Основной режим (10 итераций)
```

### Режимы с классификатором (только при полной сборке):
```bash
./build/gradient_based_optimizer --example        # Демо классификатора
./build/gradient_based_optimizer --classifier     # Генерация с ИИ
./build/classifier_example                        # Подробный тест
```

## 🛠️ Устранение проблем

### ❌ Ошибка: PyTorch not found
**Решение**: Используйте базовую сборку или установите PyTorch:
```bash
# Скачать PyTorch (если нужен классификатор)
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcpu.zip
unzip libtorch-*.zip -d ~

# Пересобрать с PyTorch
cd build && rm -rf *
cmake -DCMAKE_PREFIX_PATH=~/libtorch ..
make -j4
```

### ❌ Ошибка: Classifier mode requires PyTorch
**Решение**: Пересоберите проект с флагом `-DCMAKE_PREFIX_PATH=~/libtorch`

### ❌ Ошибка сборки
**Решение**: Очистите и пересоберите:
```bash
cd build && rm -rf *
cmake -DCMAKE_PREFIX_PATH=~/libtorch ..  # или без этого флага для базовой сборки
make -j4
```

## 📁 Требуемые файлы

### Для базовой работы:
- Изображения в папке `images/` (lenna.png, airplane.png и т.д.)
- Водяной знак `images/watermark.png`

### Для классификатора (дополнительно):
- `best_scheme_classifier_torchscript.pt`
- `ensemble_model_1_torchscript.pt`

## 💡 Рекомендации

### Для разработки:
```bash
# Используйте базовую сборку для быстрой разработки
cmake .. && make -j4
```

### Для полного функционала:
```bash
# Используйте полную сборку с классификатором
cmake -DCMAKE_PREFIX_PATH=~/libtorch .. && make -j4
```

### Проверка успешности сборки:
```bash
# Должно показать доступные режимы
./build/gradient_based_optimizer --help
```

## 🎯 Пример полного процесса

```bash
# 1. Перейти в папку проекта
cd /home/alex/projects/GRADIENT_BASED_OPTIMIZER

# 2. Создать папку сборки
mkdir -p build && cd build

# 3. Собрать (выберите один вариант)
cmake .. && make -j4                              # Базовая сборка
# ИЛИ
cmake -DCMAKE_PREFIX_PATH=~/libtorch .. && make -j4  # Полная сборка

# 4. Вернуться в корень и протестировать
cd .. && ./build/gradient_based_optimizer --help
```

## ✅ Готово!

После успешной сборки у вас будет рабочая система встраивания ЦВЗ с опциональной поддержкой ИИ-классификатора.