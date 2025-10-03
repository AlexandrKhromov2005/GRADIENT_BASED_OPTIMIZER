#!/bin/bash

# 🚀 Скрипт сборки и запуска экспериментов GRADIENT_BASED_OPTIMIZER
# Использование: ./deploy_and_run.sh [OPTIONS]

set -e  # Останавливать выполнение при ошибке

echo "🚀 =================================================="
echo "   СБОРКА И ЗАПУСК GRADIENT_BASED_OPTIMIZER"
echo "=================================================="
echo

# Параметры по умолчанию
WITH_PYTORCH=false
PYTORCH_PATH="$HOME/libtorch"
SKIP_BUILD=false
EXPERIMENT_MODE=""
CLEAN_BUILD=false
NUM_CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Парсинг аргументов
while [[ $# -gt 0 ]]; do
    case $1 in
        --with-pytorch)
            WITH_PYTORCH=true
            shift
            ;;
        --pytorch-path)
            PYTORCH_PATH="$2"
            WITH_PYTORCH=true
            shift 2
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --test)
            EXPERIMENT_MODE="test"
            shift
            ;;
        --quick)
            EXPERIMENT_MODE="quick"
            shift
            ;;
        --full)
            EXPERIMENT_MODE="full"
            shift
            ;;
        --classifier)
            EXPERIMENT_MODE="classifier"
            shift
            ;;
        --dataset)
            EXPERIMENT_MODE="dataset"
            shift
            ;;
        --help)
            echo "Использование: $0 [OPTIONS]"
            echo
            echo "Опции сборки:"
            echo "  --with-pytorch           Собрать с поддержкой PyTorch"
            echo "  --pytorch-path PATH      Путь к PyTorch (по умолчанию: ~/libtorch)"
            echo "  --clean                  Очистить build директорию перед сборкой"
            echo "  --skip-build             Пропустить сборку, только запуск"
            echo
            echo "Режимы эксперимента:"
            echo "  --test                   Быстрый тест (1 итерация)"
            echo "  --quick                  Быстрый эксперимент (5 изображений)"
            echo "  --full                   Полный эксперимент"
            echo "  --classifier             Эксперимент с классификатором (интерактивно)"
            echo "  --dataset                Генерация датасета scheme2 vs scheme3"
            echo
            echo "Примеры:"
            echo "  $0                              # Базовая сборка и быстрый тест"
            echo "  $0 --with-pytorch --test        # Сборка с PyTorch и быстрый тест"
            echo "  $0 --with-pytorch --classifier  # Полный эксперимент с классификатором"
            echo "  $0 --skip-build --full          # Пропустить сборку, запустить эксперимент"
            echo "  $0 --clean --with-pytorch       # Чистая пересборка с PyTorch"
            exit 0
            ;;
        *)
            echo "❌ Неизвестный параметр: $1"
            echo "Используйте --help для справки"
            exit 1
            ;;
    esac
done

# Проверка наличия проекта
if [[ ! -f "CMakeLists.txt" ]]; then
    echo "❌ Ошибка: CMakeLists.txt не найден"
    echo "   Запустите скрипт из корневой директории проекта"
    exit 1
fi

echo "📂 Рабочая директория: $(pwd)"
echo

# ========================================
# 1. ПРОВЕРКА ЗАВИСИМОСТЕЙ
# ========================================

if [[ "$SKIP_BUILD" != true ]]; then
    echo "🔍 Шаг 1/4: Проверка зависимостей"
    echo "----------------------------------------"

    # Проверка CMake
    if ! command -v cmake &> /dev/null; then
        echo "❌ CMake не найден"
        echo "   Установите: sudo apt install cmake (Linux) или brew install cmake (macOS)"
        exit 1
    fi
    CMAKE_VERSION=$(cmake --version | head -n1)
    echo "✅ $CMAKE_VERSION"

    # Проверка компилятора
    if ! command -v g++ &> /dev/null; then
        echo "❌ g++ не найден"
        echo "   Установите: sudo apt install build-essential"
        exit 1
    fi
    GCC_VERSION=$(g++ --version | head -n1)
    echo "✅ $GCC_VERSION"

    # Проверка OpenCV
    if pkg-config --exists opencv4 2>/dev/null; then
        OPENCV_VERSION=$(pkg-config --modversion opencv4)
        echo "✅ OpenCV $OPENCV_VERSION"
    else
        echo "⚠️  OpenCV не найден через pkg-config"
        echo "   Попытка продолжить сборку..."
    fi

    # Проверка PyTorch
    if [[ "$WITH_PYTORCH" == true ]]; then
        if [[ -d "$PYTORCH_PATH" ]]; then
            echo "✅ PyTorch найден: $PYTORCH_PATH"
            PYTORCH_SIZE=$(du -sh "$PYTORCH_PATH" | cut -f1)
            echo "   Размер: $PYTORCH_SIZE"
        else
            echo "❌ PyTorch не найден в $PYTORCH_PATH"
            echo "   Установите: ./setup_environment.sh --with-pytorch"
            exit 1
        fi
    else
        echo "⏭️  PyTorch: базовая сборка (без классификатора)"
    fi

    echo
fi

# ========================================
# 2. СБОРКА ПРОЕКТА
# ========================================

if [[ "$SKIP_BUILD" != true ]]; then
    echo "🔨 Шаг 2/4: Сборка проекта"
    echo "----------------------------------------"

    # Создание/очистка директории сборки
    if [[ "$CLEAN_BUILD" == true ]] && [[ -d "build" ]]; then
        echo "🧹 Очистка build директории..."
        rm -rf build/*
    fi

    mkdir -p build
    cd build

    # Конфигурация CMake
    echo "⚙️  Конфигурация CMake..."
    CMAKE_ARGS=""
    if [[ "$WITH_PYTORCH" == true ]]; then
        CMAKE_ARGS="-DCMAKE_PREFIX_PATH=$PYTORCH_PATH"
        echo "   С поддержкой PyTorch"
    else
        echo "   Базовая конфигурация"
    fi

    if cmake $CMAKE_ARGS .. 2>&1 | tee cmake_output.log; then
        echo "✅ Конфигурация успешна"

        # Проверка что нашлось
        if grep -q "PyTorch found" cmake_output.log 2>/dev/null; then
            echo "   🤖 PyTorch интеграция включена"
        fi
        if grep -q "OpenCV" cmake_output.log 2>/dev/null; then
            echo "   📷 OpenCV найден"
        fi
    else
        echo "❌ Ошибка конфигурации CMake"
        cd ..
        exit 1
    fi

    echo
    echo "🔧 Компиляция проекта (используя $NUM_CORES ядер)..."
    if make -j"$NUM_CORES" 2>&1 | tee make_output.log; then
        echo "✅ Компиляция успешна"
    else
        echo "❌ Ошибка компиляции"
        echo "   Проверьте логи: build/make_output.log"
        cd ..
        exit 1
    fi

    cd ..

    echo
    echo "📦 Собранные исполняемые файлы:"
    if [[ -f "build/gradient_based_optimizer" ]]; then
        ls -lh build/gradient_based_optimizer
    fi
    if [[ -f "build/classifier_example" ]]; then
        ls -lh build/classifier_example
        echo "   ✅ Классификатор поддерживается"
    fi
    if [[ -f "build/single_classifier_example" ]]; then
        ls -lh build/single_classifier_example
    fi

    echo
else
    echo "⏭️  Пропуск сборки (--skip-build)"
    echo

    # Проверка что проект уже собран
    if [[ ! -f "build/gradient_based_optimizer" ]]; then
        echo "❌ Исполняемый файл не найден: build/gradient_based_optimizer"
        echo "   Соберите проект сначала (без --skip-build)"
        exit 1
    fi
fi

# ========================================
# 3. ПРОВЕРКА ФАЙЛОВ ПРОЕКТА
# ========================================

echo "📁 Шаг 3/4: Проверка файлов проекта"
echo "----------------------------------------"

# Проверка изображений
if [[ ! -d "images" ]]; then
    echo "⚠️  Директория images/ не найдена, создаю..."
    mkdir -p images
fi

IMAGE_COUNT=$(find images -name "*.png" -o -name "*.jpg" 2>/dev/null | wc -l)
if [[ $IMAGE_COUNT -eq 0 ]]; then
    echo "⚠️  Изображения не найдены в images/"
    echo "   Добавьте тестовые изображения перед запуском"
else
    echo "✅ Найдено изображений: $IMAGE_COUNT"
fi

# Проверка watermark.png
if [[ ! -f "images/watermark.png" ]]; then
    echo "⚠️  watermark.png не найден в images/"
    echo "   Система может не работать корректно"
else
    echo "✅ Водяной знак найден: images/watermark.png"
fi

# Проверка моделей классификатора
if [[ "$WITH_PYTORCH" == true ]] || [[ -f "build/classifier_example" ]]; then
    echo
    echo "🤖 Проверка моделей классификатора:"

    MODEL_FOUND=false
    if [[ -f "final_model_torchscript.pt" ]]; then
        MODEL_SIZE=$(du -sh final_model_torchscript.pt | cut -f1)
        echo "   ✅ final_model_torchscript.pt ($MODEL_SIZE)"
        MODEL_FOUND=true
    else
        echo "   ⚠️  final_model_torchscript.pt не найден"
    fi

    if [[ -f "best_scheme_classifier_torchscript.pt" ]]; then
        MODEL_SIZE=$(du -sh best_scheme_classifier_torchscript.pt | cut -f1)
        echo "   ✅ best_scheme_classifier_torchscript.pt ($MODEL_SIZE)"
        MODEL_FOUND=true
    else
        echo "   ⚠️  best_scheme_classifier_torchscript.pt не найден"
    fi

    if [[ -f "ensemble_model_1_torchscript.pt" ]]; then
        MODEL_SIZE=$(du -sh ensemble_model_1_torchscript.pt | cut -f1)
        echo "   ✅ ensemble_model_1_torchscript.pt ($MODEL_SIZE)"
    else
        echo "   ⚠️  ensemble_model_1_torchscript.pt не найден"
    fi

    if [[ "$MODEL_FOUND" != true ]]; then
        echo "   ❌ Модели не найдены - классификатор не будет работать"
    fi
fi

# Проверка схем встраивания
if [[ ! -f "embedding_schemes.json" ]]; then
    echo "⚠️  embedding_schemes.json не найден"
else
    echo "✅ Схемы встраивания: embedding_schemes.json"
fi

echo

# ========================================
# 4. ЗАПУСК ЭКСПЕРИМЕНТА
# ========================================

echo "🚀 Шаг 4/4: Запуск эксперимента"
echo "----------------------------------------"

# Если режим не указан, спросить у пользователя
if [[ -z "$EXPERIMENT_MODE" ]]; then
    echo "🎯 Выберите режим эксперимента:"
    echo
    echo "1) Быстрый тест (--test) - 1 итерация, ~1-2 мин"
    echo "2) Генерация датасета (--dataset) - сравнение scheme2 vs scheme3"

    if [[ -f "build/classifier_example" ]]; then
        echo "3) Эксперимент с классификатором (интерактивный)"
        echo "4) Датасет с классификатором (--dataset-classifier)"
        echo "5) Пример классификатора (./classifier_example)"
        echo "6) Не запускать эксперимент"
        MAX_OPTION=6
    else
        echo "3) Не запускать эксперимент"
        MAX_OPTION=3
    fi

    echo
    read -p "Введите номер (1-$MAX_OPTION): " choice
    echo

    case $choice in
        1) EXPERIMENT_MODE="test" ;;
        2) EXPERIMENT_MODE="dataset" ;;
        3)
            if [[ -f "build/classifier_example" ]]; then
                EXPERIMENT_MODE="classifier"
            else
                echo "✅ Сборка завершена. Запуск экспериментов пропущен."
                exit 0
            fi
            ;;
        4)
            if [[ -f "build/classifier_example" ]]; then
                EXPERIMENT_MODE="dataset-classifier"
            else
                echo "❌ Неверный выбор"
                exit 1
            fi
            ;;
        5)
            if [[ -f "build/classifier_example" ]]; then
                EXPERIMENT_MODE="classifier-example"
            else
                echo "❌ Неверный выбор"
                exit 1
            fi
            ;;
        6)
            echo "✅ Сборка завершена. Запуск экспериментов пропущен."
            exit 0
            ;;
        *)
            echo "❌ Неверный выбор"
            exit 1
            ;;
    esac
fi

# Запуск выбранного эксперимента
echo "⏱️  Начало: $(date)"
echo

case $EXPERIMENT_MODE in
    test)
        echo "⚡ Запуск быстрого теста..."
        ./build/gradient_based_optimizer --test
        ;;

    dataset)
        echo "📊 Генерация датасета (scheme2 vs scheme3)..."
        ./build/gradient_based_optimizer --dataset
        ;;

    dataset-classifier)
        echo "🤖 Генерация датасета с классификатором..."
        ./build/gradient_based_optimizer --dataset-classifier
        ;;

    quick)
        echo "⚡ Быстрый эксперимент (5 изображений)..."
        if [[ -f "run_quick_experiment.sh" ]]; then
            chmod +x run_quick_experiment.sh
            ./run_quick_experiment.sh
        else
            echo "❌ run_quick_experiment.sh не найден"
            exit 1
        fi
        ;;

    classifier)
        echo "🤖 Запуск интерактивного эксперимента с классификатором..."
        if [[ -f "run_classifier_experiment.sh" ]]; then
            chmod +x run_classifier_experiment.sh
            ./run_classifier_experiment.sh
        else
            echo "❌ run_classifier_experiment.sh не найден"
            exit 1
        fi
        ;;

    classifier-example)
        echo "🧪 Запуск примера классификатора..."
        ./build/classifier_example
        ;;

    full)
        echo "🚀 Запуск полного эксперимента..."
        if [[ -f "run_full_metrics_experiment.sh" ]]; then
            chmod +x run_full_metrics_experiment.sh
            ./run_full_metrics_experiment.sh
        else
            ./build/gradient_based_optimizer
        fi
        ;;

    *)
        echo "❌ Неизвестный режим: $EXPERIMENT_MODE"
        exit 1
        ;;
esac

EXIT_CODE=$?

echo
echo "⏱️  Окончание: $(date)"
echo

# ========================================
# ИТОГОВЫЙ ОТЧЕТ
# ========================================

if [[ $EXIT_CODE -eq 0 ]]; then
    echo "✅ =================================================="
    echo "   ЭКСПЕРИМЕНТ ЗАВЕРШЕН УСПЕШНО"
    echo "=================================================="
    echo

    # Показать результаты если есть
    if [[ -d "dataset" ]]; then
        echo "📂 Результаты датасета:"
        find dataset -type d -maxdepth 1 2>/dev/null | while read dir; do
            if [[ "$dir" != "dataset" ]]; then
                COUNT=$(find "$dir" -name "*.png" 2>/dev/null | wc -l)
                echo "   $(basename "$dir"): $COUNT файлов"
            fi
        done
        echo
    fi

    if [[ -d "dataset_classifier" ]]; then
        echo "📂 Результаты с классификатором:"

        scheme2=$(find dataset_classifier/scheme2_selected -name "*.png" 2>/dev/null | wc -l)
        scheme3=$(find dataset_classifier/scheme3_selected -name "*.png" 2>/dev/null | wc -l)
        correct=$(find dataset_classifier/extraction_correct -name "*.png" 2>/dev/null | wc -l)
        incorrect=$(find dataset_classifier/extraction_incorrect -name "*.png" 2>/dev/null | wc -l)
        total=$((correct + incorrect))

        echo "   Scheme2 выбрано: $scheme2 блоков"
        echo "   Scheme3 выбрано: $scheme3 блоков"
        echo "   Извлечено правильно: $correct блоков"
        echo "   Извлечено неправильно: $incorrect блоков"

        if [[ $total -gt 0 ]]; then
            accuracy=$(echo "scale=2; $correct * 100 / $total" | bc -l 2>/dev/null || echo "N/A")
            echo "   🎯 Точность: $accuracy%"
        fi
        echo
    fi

    echo "📚 Полезные команды:"
    echo "   ./build/gradient_based_optimizer --help  # Справка"
    echo "   ./clean_experiment_results.sh            # Очистить результаты"
    echo "   ./analyze_existing_results.sh            # Анализ результатов"

else
    echo "❌ =================================================="
    echo "   ЭКСПЕРИМЕНТ ЗАВЕРШЕН С ОШИБКОЙ"
    echo "=================================================="
    echo
    echo "Код ошибки: $EXIT_CODE"
    echo
    echo "🔍 Проверьте:"
    echo "   1. Наличие изображений в images/"
    echo "   2. Наличие watermark.png"
    echo "   3. Корректность моделей классификатора (если используется)"
    echo "   4. Логи сборки: build/cmake_output.log, build/make_output.log"
fi

echo
echo "🏁 Скрипт завершен"
