# 📜 Документация скриптов

## 🔧 setup_environment.sh

Автоматическая установка всех зависимостей проекта.

### Использование

```bash
./setup_environment.sh [OPTIONS]
```

### Опции

| Опция | Описание |
|-------|----------|
| `--with-pytorch` | Установить PyTorch для классификатора |
| `--pytorch-path PATH` | Путь установки PyTorch (по умолчанию: ~/libtorch) |
| `--pytorch-cuda` | Установить версию с CUDA GPU поддержкой |
| `--help` | Показать справку |

### Примеры

```bash
# Базовая установка
./setup_environment.sh

# С поддержкой классификатора (CPU)
./setup_environment.sh --with-pytorch

# С GPU поддержкой
./setup_environment.sh --with-pytorch --pytorch-cuda

# Установка в custom директорию
./setup_environment.sh --with-pytorch --pytorch-path /opt/libtorch
```

### Что устанавливается

**Базовая установка:**
- ✅ GCC/Clang компилятор
- ✅ CMake 3.16+
- ✅ OpenCV 4.x
- ✅ build-essential (make, pkg-config и т.д.)
- ✅ Утилиты (wget, unzip, git)

**С --with-pytorch:**
- ✅ Всё из базовой установки
- ✅ PyTorch C++ библиотека (~200MB)
- ✅ Автоматическая настройка переменных окружения

**С --pytorch-cuda:**
- ✅ Всё из --with-pytorch
- ✅ CUDA-оптимизированная версия PyTorch

### Системные требования

- Ubuntu 20.04+ / Debian 11+ / macOS 10.15+
- Права sudo
- Интернет соединение
- ~500MB свободного места (базовая)
- ~1GB свободного места (с PyTorch)

---

## 🚀 deploy_and_run.sh

Автоматическая сборка проекта и запуск экспериментов.

### Использование

```bash
./deploy_and_run.sh [OPTIONS]
```

### Опции сборки

| Опция | Описание |
|-------|----------|
| `--with-pytorch` | Собрать с поддержкой PyTorch |
| `--pytorch-path PATH` | Путь к PyTorch (по умолчанию: ~/libtorch) |
| `--clean` | Очистить build директорию перед сборкой |
| `--skip-build` | Пропустить сборку, только запуск |

### Режимы эксперимента

| Опция | Описание | Время выполнения |
|-------|----------|------------------|
| `--test` | Быстрый тест (1 итерация) | ~1-2 мин |
| `--quick` | Быстрый эксперимент (5 изображений) | ~5-10 мин |
| `--dataset` | Генерация датасета scheme2 vs scheme3 | ~15-30 мин |
| `--classifier` | Интерактивный эксперимент с классификатором | варьируется |
| `--full` | Полный эксперимент | ~30-60 мин |

### Примеры

```bash
# Быстрый тест базовой функциональности
./deploy_and_run.sh --test

# Сборка с PyTorch и быстрый тест
./deploy_and_run.sh --with-pytorch --test

# Интерактивный выбор режима эксперимента
./deploy_and_run.sh --with-pytorch

# Полный эксперимент с классификатором
./deploy_and_run.sh --with-pytorch --full

# Чистая пересборка
./deploy_and_run.sh --clean --with-pytorch

# Запуск без пересборки
./deploy_and_run.sh --skip-build --classifier

# Генерация датасета
./deploy_and_run.sh --dataset
```

### Этапы выполнения

1. **Проверка зависимостей**
   - CMake, GCC, OpenCV
   - PyTorch (если указан --with-pytorch)

2. **Сборка проекта**
   - Конфигурация CMake
   - Компиляция (используя все доступные ядра)
   - Проверка собранных файлов

3. **Проверка файлов**
   - Наличие изображений
   - Наличие watermark.png
   - Наличие моделей классификатора (если нужно)

4. **Запуск эксперимента**
   - Выбранный режим
   - Вывод результатов
   - Статистика

### Результаты

После выполнения эксперимента:

**Для --dataset:**
```
dataset/
├── scheme2_correct/
├── scheme2_incorrect/
├── scheme3_correct/
└── scheme3_incorrect/
```

**Для --classifier:**
```
dataset_classifier/
├── scheme2_selected/
├── scheme3_selected/
├── extraction_correct/
└── extraction_incorrect/
```

### Требования

- Успешная установка зависимостей (`setup_environment.sh`)
- Наличие изображений в `images/`
- Наличие `images/watermark.png`
- Модели классификатора (для режимов с --pytorch)

---

## 🧹 Скрипты очистки

### clean_experiment_results.sh
Удаляет все результаты экспериментов.

```bash
./clean_experiment_results.sh
```

Удаляет:
- `dataset/`
- `dataset_classifier/`
- Все `new_*.png` файлы
- Все `*_wm.png` файлы

### clean_dataset.sh
Удаляет только датасеты.

```bash
./clean_dataset.sh
```

Удаляет:
- `dataset/`
- `dataset_classifier/`

### clean_results.sh
Удаляет промежуточные результаты.

```bash
./clean_results.sh
```

Удаляет:
- `results/`
- Временные файлы

---

## 📊 Скрипты анализа

### analyze_existing_results.sh
Анализирует существующие результаты экспериментов.

```bash
./analyze_existing_results.sh
```

Выводит:
- Количество файлов в каждой категории
- Процент правильных/неправильных извлечений
- Распределение по схемам
- Общую точность системы

---

## 🎯 Специализированные скрипты

### run_quick_experiment.sh
Быстрый эксперимент на 5 изображениях с классификатором.

```bash
./run_quick_experiment.sh
```

### run_classifier_experiment.sh
Интерактивный эксперимент с классификатором и детальным меню.

```bash
./run_classifier_experiment.sh
```

Предлагает выбор:
1. Быстрый тест (1 итерация)
2. Полный эксперимент
3. Ограниченный набор (5 изображений)
4. Пользовательские настройки

### run_full_metrics_experiment.sh
Полный эксперимент с детальными метриками.

```bash
./run_full_metrics_experiment.sh
```

### build.sh
Простая автоматическая сборка проекта.

```bash
./build.sh
```

Автоматически определяет наличие PyTorch и собирает соответствующую версию.

---

## 🔄 Типичный рабочий процесс

### Первый запуск на новой машине

```bash
# 1. Установка окружения
./setup_environment.sh --with-pytorch

# 2. Быстрый тест
./deploy_and_run.sh --with-pytorch --test

# 3. Если успешно - полный эксперимент
./deploy_and_run.sh --skip-build --classifier
```

### Разработка и тестирование

```bash
# 1. Внести изменения в код
# ...

# 2. Пересобрать
./deploy_and_run.sh --clean --with-pytorch --test

# 3. Если работает - запустить на всех данных
./deploy_and_run.sh --skip-build --full
```

### Batch обработка

```bash
# 1. Подготовить изображения
cp /path/to/images/* images/

# 2. Запустить обработку
./deploy_and_run.sh --skip-build --dataset-classifier

# 3. Проанализировать результаты
./analyze_existing_results.sh

# 4. Очистить перед новым запуском
./clean_experiment_results.sh
```

---

## ⚠️ Важные замечания

### Переменные окружения

После установки PyTorch скрипт добавляет в `~/.bashrc`:

```bash
export LD_LIBRARY_PATH=~/libtorch/lib:$LD_LIBRARY_PATH
export CMAKE_PREFIX_PATH=~/libtorch:$CMAKE_PREFIX_PATH
```

Для применения: `source ~/.bashrc` или перезапустить терминал.

### Права на выполнение

Если скрипт не запускается:

```bash
chmod +x setup_environment.sh
chmod +x deploy_and_run.sh
chmod +x *.sh
```

### Логи

Скрипт `deploy_and_run.sh` сохраняет логи:
- `build/cmake_output.log` - лог конфигурации CMake
- `build/make_output.log` - лог компиляции

Полезно для диагностики проблем.

---

## 📚 Дополнительная информация

- [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) - Полное руководство по развертыванию
- [QUICK_START.md](QUICK_START.md) - Быстрый старт
- [BUILD_INSTRUCTIONS.md](BUILD_INSTRUCTIONS.md) - Детали сборки
- [CLAUDE.md](CLAUDE.md) - Архитектура проекта
