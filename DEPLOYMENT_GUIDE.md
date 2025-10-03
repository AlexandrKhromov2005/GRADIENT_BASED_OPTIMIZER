# 📦 Руководство по развертыванию проекта на новом компьютере

Это руководство описывает процесс переноса и запуска проекта GRADIENT_BASED_OPTIMIZER на новом компьютере для проведения вычислений.

## 🎯 Быстрый старт

```bash
# 1. Клонировать/скопировать проект
git clone <repository-url> GRADIENT_BASED_OPTIMIZER
cd GRADIENT_BASED_OPTIMIZER

# 2. Установить зависимости
./setup_environment.sh --with-pytorch

# 3. Собрать и запустить
./deploy_and_run.sh --with-pytorch --test
```

## 📋 Детальная инструкция

### Шаг 1: Получение проекта

#### Вариант A: Через Git (рекомендуется)
```bash
git clone <repository-url> GRADIENT_BASED_OPTIMIZER
cd GRADIENT_BASED_OPTIMIZER
```

#### Вариант B: Копирование архива
```bash
# На исходном компьютере
tar -czf gradient_optimizer.tar.gz GRADIENT_BASED_OPTIMIZER/

# На новом компьютере
scp user@old-computer:gradient_optimizer.tar.gz .
tar -xzf gradient_optimizer.tar.gz
cd GRADIENT_BASED_OPTIMIZER
```

#### Вариант C: Через scp/rsync
```bash
# Синхронизация через сеть
rsync -avz --progress user@old-computer:/path/to/GRADIENT_BASED_OPTIMIZER/ ./GRADIENT_BASED_OPTIMIZER/
cd GRADIENT_BASED_OPTIMIZER
```

### Шаг 2: Установка зависимостей

Проект предоставляет автоматический скрипт установки всех необходимых зависимостей.

#### Базовая установка (только OpenCV, без классификатора)
```bash
./setup_environment.sh
```

Это установит:
- ✅ Компилятор C++ (GCC/Clang)
- ✅ CMake
- ✅ OpenCV 4.x
- ✅ Утилиты (wget, unzip, pkg-config)

**Функциональность:** Базовое встраивание водяных знаков без автоматического выбора схем.

#### Полная установка (с PyTorch классификатором)
```bash
./setup_environment.sh --with-pytorch
```

Дополнительно установит:
- ✅ PyTorch C++ библиотеку (~200MB)
- ✅ Интеграция с нейросетевым классификатором

**Функциональность:** Полная функциональность включая автоматический выбор оптимальной схемы встраивания для каждого блока.

#### Установка с CUDA (для GPU ускорения)
```bash
./setup_environment.sh --with-pytorch --pytorch-cuda
```

**Требования:** Установленные NVIDIA драйверы и CUDA Toolkit 11.8

#### Параметры setup_environment.sh

| Параметр | Описание |
|----------|----------|
| `--with-pytorch` | Установить PyTorch для классификатора |
| `--pytorch-path PATH` | Путь установки PyTorch (по умолчанию: ~/libtorch) |
| `--pytorch-cuda` | Установить версию с GPU поддержкой |
| `--help` | Показать справку |

### Шаг 3: Перенос необходимых файлов

#### Обязательные файлы (минимум)

```
GRADIENT_BASED_OPTIMIZER/
├── images/                          # Директория с изображениями
│   ├── watermark.png               # ⚠️ ОБЯЗАТЕЛЬНО
│   ├── lenna.png                   # Тестовые изображения
│   ├── airplane.png
│   └── ...
├── embedding_schemes.json          # Схемы встраивания
└── CMakeLists.txt                  # Файл сборки
```

#### Файлы для классификатора (опционально)

```
GRADIENT_BASED_OPTIMIZER/
├── final_model_torchscript.pt              # Основная модель (рекомендуется)
├── best_scheme_classifier_torchscript.pt   # Альтернативная модель
└── ensemble_model_1_torchscript.pt         # Модель ансамбля
```

**Важно:** Модели должны быть в формате TorchScript (.pt), не PyTorch (.pth).

#### Копирование изображений

Если у вас нет тестовых изображений:

```bash
mkdir -p images
# Скопируйте свои изображения или используйте стандартный набор
```

Стандартный набор изображений для тестирования:
- aerial.png
- airplane.png
- baboon.png
- boat.png
- bridge.png
- lenna.png
- pepper.png
- watermark.png (обязательно!)

### Шаг 4: Сборка и запуск

#### Автоматическая сборка и запуск

```bash
# Базовая сборка + быстрый тест
./deploy_and_run.sh --test

# С PyTorch + быстрый тест
./deploy_and_run.sh --with-pytorch --test

# С PyTorch + интерактивный выбор режима
./deploy_and_run.sh --with-pytorch
```

#### Параметры deploy_and_run.sh

**Опции сборки:**
| Параметр | Описание |
|----------|----------|
| `--with-pytorch` | Собрать с поддержкой PyTorch |
| `--pytorch-path PATH` | Путь к PyTorch |
| `--clean` | Очистить build директорию перед сборкой |
| `--skip-build` | Пропустить сборку, только запуск |

**Режимы эксперимента:**
| Параметр | Описание | Время |
|----------|----------|-------|
| `--test` | Быстрый тест (1 итерация) | ~1-2 мин |
| `--quick` | Быстрый эксперимент (5 изображений) | ~5-10 мин |
| `--dataset` | Генерация датасета scheme2 vs scheme3 | ~15-30 мин |
| `--classifier` | Интерактивный эксперимент с классификатором | варьируется |
| `--full` | Полный эксперимент | ~30-60 мин |

#### Ручная сборка (если скрипт не подходит)

**Базовая сборка:**
```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
cd ..
```

**С PyTorch:**
```bash
mkdir -p build && cd build
cmake -DCMAKE_PREFIX_PATH=~/libtorch ..
make -j$(nproc)
cd ..
```

**Запуск:**
```bash
./build/gradient_based_optimizer --help
./build/gradient_based_optimizer --test
```

## 🎮 Режимы работы системы

### 1. Быстрый тест
```bash
./deploy_and_run.sh --test
# или
./build/gradient_based_optimizer --test
```
- 1 итерация оптимизации на первом изображении
- Время: ~1-2 минуты
- Цель: проверка работоспособности

### 2. Генерация датасета
```bash
./deploy_and_run.sh --dataset
# или
./build/gradient_based_optimizer --dataset
```
- Сравнение scheme2 vs scheme3
- Встраивание и извлечение водяных знаков
- Результаты в `dataset/`

### 3. Датасет с классификатором
```bash
./deploy_and_run.sh --with-pytorch --classifier
# или
./build/gradient_based_optimizer --dataset-classifier
```
- Автоматический выбор схемы для каждого блока
- Использование нейросети
- Результаты в `dataset_classifier/`

### 4. Пример классификатора
```bash
./build/classifier_example
# или
./build/single_classifier_example
```
- Демонстрация работы классификатора
- Детальный вывод предсказаний

## 📊 Структура результатов

### После генерации датасета (`--dataset`)
```
dataset/
├── scheme2_correct/      # Правильно извлечённые (scheme2)
├── scheme2_incorrect/    # Неправильно извлечённые (scheme2)
├── scheme3_correct/      # Правильно извлечённые (scheme3)
└── scheme3_incorrect/    # Неправильно извлечённые (scheme3)
```

### После эксперимента с классификатором (`--dataset-classifier`)
```
dataset_classifier/
├── scheme2_selected/      # Блоки где выбрана scheme2
├── scheme3_selected/      # Блоки где выбрана scheme3
├── extraction_correct/    # Правильно извлечённые
└── extraction_incorrect/  # Неправильно извлечённые
```

## 🔧 Управление результатами

### Очистка результатов
```bash
./clean_experiment_results.sh  # Удалить все результаты экспериментов
./clean_dataset.sh             # Удалить только датасеты
./clean_results.sh             # Удалить промежуточные результаты
```

### Анализ результатов
```bash
./analyze_existing_results.sh  # Статистика по результатам
```

## ⚙️ Системные требования

### Минимальные (базовая сборка)
- **ОС:** Ubuntu 20.04+ / Debian 11+ / macOS 10.15+
- **CPU:** 2+ ядра
- **RAM:** 4 GB
- **Диск:** 1 GB свободного места
- **Зависимости:**
  - CMake 3.16+
  - GCC 9+ / Clang 10+
  - OpenCV 4.x

### Рекомендуемые (с классификатором)
- **ОС:** Ubuntu 22.04
- **CPU:** 4+ ядра
- **RAM:** 8 GB
- **Диск:** 2 GB свободного места
- **Дополнительно:** PyTorch C++ (~200MB)

### Оптимальные (с CUDA)
- **GPU:** NVIDIA с поддержкой CUDA 11.8+
- **VRAM:** 4 GB+
- **RAM:** 16 GB
- **CUDA:** 11.8 или 12.1

## 🐛 Устранение проблем

### Ошибка: CMake не найден
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install cmake

# macOS
brew install cmake
```

### Ошибка: OpenCV не найден
```bash
# Ubuntu/Debian
sudo apt install libopencv-dev

# macOS
brew install opencv
```

### Ошибка: PyTorch not found
```bash
# Переустановите PyTorch
./setup_environment.sh --with-pytorch

# Или вручную
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcpu.zip
unzip libtorch-*.zip -d ~
```

### Ошибка: Classifier mode requires PyTorch
Решение: Пересоберите проект с PyTorch
```bash
./deploy_and_run.sh --clean --with-pytorch
```

### Ошибка: libopencv_core.so not found
```bash
# Обновите кэш библиотек
sudo ldconfig

# Проверьте установку
pkg-config --modversion opencv4
```

### Ошибка: Cannot find watermark.png
```bash
# Убедитесь что файл существует
ls images/watermark.png

# Создайте тестовый watermark (32x32 белый квадрат)
convert -size 32x32 xc:white images/watermark.png
```

### Скрипт не запускается
```bash
# Сделайте скрипты исполняемыми
chmod +x setup_environment.sh
chmod +x deploy_and_run.sh
```

### Ошибки компиляции C++
```bash
# Проверьте версию компилятора (нужен C++17)
g++ --version  # Должен быть 9.0+

# Обновите компилятор
sudo apt install g++-11
```

## 📈 Типичные сценарии использования

### Сценарий 1: Перенос на вычислительный сервер (CPU only)
```bash
# На новом сервере
./setup_environment.sh --with-pytorch
./deploy_and_run.sh --with-pytorch --test

# Если всё работает - запустить полный эксперимент
./deploy_and_run.sh --with-pytorch --classifier
```

### Сценарий 2: Перенос на рабочую станцию с GPU
```bash
# Установка с CUDA
./setup_environment.sh --with-pytorch --pytorch-cuda

# Сборка и тест
./deploy_and_run.sh --with-pytorch --test

# Запуск на всех изображениях
./deploy_and_run.sh --skip-build --full
```

### Сценарий 3: Быстрая проверка на новой машине
```bash
# Минимальная установка
./setup_environment.sh

# Базовый тест
./deploy_and_run.sh --test

# Проверка что всё работает
./build/gradient_based_optimizer --help
```

### Сценарий 4: Batch обработка множества изображений
```bash
# Подготовка
cp -r /path/to/images/* images/
./deploy_and_run.sh --with-pytorch --skip-build --full

# Анализ результатов
./analyze_existing_results.sh
```

## 📚 Дополнительная документация

- [README.md](README.md) - Общая информация о проекте
- [CLAUDE.md](CLAUDE.md) - Полное описание архитектуры и API
- [BUILD_INSTRUCTIONS.md](BUILD_INSTRUCTIONS.md) - Детальная инструкция по сборке
- [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md) - Руководство по экспериментам
- [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) - Интеграция классификатора

## ✅ Чеклист для развертывания

- [ ] Проект скопирован на новый компьютер
- [ ] Запущен `setup_environment.sh` (с --with-pytorch если нужен классификатор)
- [ ] Скопированы изображения в `images/`
- [ ] Скопирован `watermark.png` в `images/`
- [ ] (Опционально) Скопированы модели классификатора (*.pt файлы)
- [ ] Запущен `deploy_and_run.sh --test` для проверки
- [ ] Система успешно выполнила тестовый запуск
- [ ] Запущен полный эксперимент

## 🎯 Контакты и поддержка

При возникновении проблем:
1. Проверьте раздел "Устранение проблем"
2. Изучите логи: `build/cmake_output.log`, `build/make_output.log`
3. Проверьте наличие всех необходимых файлов
4. Убедитесь что все зависимости установлены

---

**Успешного развертывания! 🚀**
