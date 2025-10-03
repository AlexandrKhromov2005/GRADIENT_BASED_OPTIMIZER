#!/bin/bash

# 🚀 Скрипт автоматической установки всех зависимостей для GRADIENT_BASED_OPTIMIZER
# Использование: ./setup_environment.sh [--with-pytorch] [--pytorch-path PATH]

set -e  # Останавливать выполнение при ошибке

echo "🔧 =================================================="
echo "   УСТАНОВКА ОКРУЖЕНИЯ GRADIENT_BASED_OPTIMIZER"
echo "=================================================="
echo

# Параметры
INSTALL_PYTORCH=false
PYTORCH_PATH="$HOME/libtorch"
PYTORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcpu.zip"

# Парсинг аргументов
while [[ $# -gt 0 ]]; do
    case $1 in
        --with-pytorch)
            INSTALL_PYTORCH=true
            shift
            ;;
        --pytorch-path)
            PYTORCH_PATH="$2"
            shift 2
            ;;
        --pytorch-cuda)
            PYTORCH_URL="https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcu118.zip"
            INSTALL_PYTORCH=true
            shift
            ;;
        --help)
            echo "Использование: $0 [OPTIONS]"
            echo
            echo "Опции:"
            echo "  --with-pytorch       Установить PyTorch для классификатора"
            echo "  --pytorch-path PATH  Путь установки PyTorch (по умолчанию: ~/libtorch)"
            echo "  --pytorch-cuda       Установить версию с CUDA поддержкой"
            echo "  --help               Показать эту справку"
            echo
            echo "Примеры:"
            echo "  $0                           # Базовая установка (только OpenCV)"
            echo "  $0 --with-pytorch            # С поддержкой классификатора (CPU)"
            echo "  $0 --with-pytorch --pytorch-cuda  # С поддержкой CUDA"
            exit 0
            ;;
        *)
            echo "❌ Неизвестный параметр: $1"
            echo "Используйте --help для справки"
            exit 1
            ;;
    esac
done

# Проверка ОС
echo "🔍 Проверка операционной системы..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "✅ Обнаружена Linux система"
    OS_TYPE="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "✅ Обнаружена macOS"
    OS_TYPE="macos"
else
    echo "⚠️  Неизвестная ОС: $OSTYPE"
    echo "Скрипт оптимизирован для Linux/macOS"
    read -p "Продолжить? (y/N): " -n 1 -r
    echo
    [[ ! $REPLY =~ ^[Yy]$ ]] && exit 1
    OS_TYPE="unknown"
fi

# Проверка прав sudo
echo "🔐 Проверка прав администратора..."
if ! sudo -v; then
    echo "❌ Требуются права sudo для установки пакетов"
    exit 1
fi
echo "✅ Права подтверждены"
echo

# ========================================
# 1. УСТАНОВКА БАЗОВЫХ ЗАВИСИМОСТЕЙ
# ========================================

echo "📦 Шаг 1/4: Установка базовых зависимостей"
echo "----------------------------------------"

if [[ "$OS_TYPE" == "linux" ]]; then
    echo "Обновление списка пакетов..."
    sudo apt update

    echo "Установка компилятора и инструментов сборки..."
    sudo apt install -y \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        pkg-config \
        ca-certificates

    echo "✅ Базовые инструменты установлены"

elif [[ "$OS_TYPE" == "macos" ]]; then
    echo "Проверка Homebrew..."
    if ! command -v brew &> /dev/null; then
        echo "Установка Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi

    echo "Установка инструментов сборки..."
    brew install cmake wget
    echo "✅ Базовые инструменты установлены"
fi

echo

# ========================================
# 2. УСТАНОВКА OpenCV
# ========================================

echo "📦 Шаг 2/4: Установка OpenCV"
echo "----------------------------------------"

if [[ "$OS_TYPE" == "linux" ]]; then
    echo "Установка OpenCV и зависимостей..."
    sudo apt install -y \
        libopencv-dev \
        libopencv-core-dev \
        libopencv-highgui-dev \
        libopencv-imgproc-dev \
        libopencv-imgcodecs-dev

    # Проверка установки
    if pkg-config --exists opencv4; then
        OPENCV_VERSION=$(pkg-config --modversion opencv4)
        echo "✅ OpenCV $OPENCV_VERSION установлен"
    else
        echo "⚠️  OpenCV не найден через pkg-config"
        echo "   Но это может быть нормально для некоторых дистрибутивов"
    fi

elif [[ "$OS_TYPE" == "macos" ]]; then
    echo "Установка OpenCV через Homebrew..."
    brew install opencv
    echo "✅ OpenCV установлен"
fi

echo

# ========================================
# 3. УСТАНОВКА PyTorch (опционально)
# ========================================

echo "📦 Шаг 3/4: Установка PyTorch"
echo "----------------------------------------"

if [[ "$INSTALL_PYTORCH" == true ]]; then
    echo "🤖 Установка PyTorch C++ библиотеки..."
    echo "Путь установки: $PYTORCH_PATH"
    echo "URL: $PYTORCH_URL"
    echo

    # Проверка существующей установки
    if [[ -d "$PYTORCH_PATH" ]]; then
        echo "⚠️  PyTorch уже установлен в $PYTORCH_PATH"
        read -p "Переустановить? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Удаление старой версии..."
            rm -rf "$PYTORCH_PATH"
        else
            echo "✅ Используем существующую установку PyTorch"
            INSTALL_PYTORCH=false
        fi
    fi

    if [[ "$INSTALL_PYTORCH" == true ]]; then
        # Создание временной директории
        TEMP_DIR=$(mktemp -d)
        cd "$TEMP_DIR"

        echo "📥 Загрузка PyTorch..."
        echo "   (это может занять несколько минут, размер ~200MB)"
        if wget --progress=bar:force "$PYTORCH_URL" -O libtorch.zip; then
            echo "✅ Загрузка завершена"
        else
            echo "❌ Ошибка загрузки PyTorch"
            rm -rf "$TEMP_DIR"
            exit 1
        fi

        echo "📦 Распаковка PyTorch..."
        if unzip -q libtorch.zip; then
            echo "✅ Распаковка завершена"
        else
            echo "❌ Ошибка распаковки"
            rm -rf "$TEMP_DIR"
            exit 1
        fi

        echo "📁 Перемещение в $PYTORCH_PATH..."
        mkdir -p "$(dirname "$PYTORCH_PATH")"
        mv libtorch "$PYTORCH_PATH"

        # Очистка
        cd -
        rm -rf "$TEMP_DIR"

        echo "✅ PyTorch установлен успешно"
        echo "   Размер: $(du -sh "$PYTORCH_PATH" | cut -f1)"

        # Добавление в LD_LIBRARY_PATH
        if [[ "$OS_TYPE" == "linux" ]]; then
            echo
            echo "📝 Добавление PyTorch в переменные окружения..."

            # Добавление в .bashrc
            if ! grep -q "$PYTORCH_PATH/lib" ~/.bashrc; then
                echo "" >> ~/.bashrc
                echo "# PyTorch C++ Library" >> ~/.bashrc
                echo "export LD_LIBRARY_PATH=$PYTORCH_PATH/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
                echo "export CMAKE_PREFIX_PATH=$PYTORCH_PATH:\$CMAKE_PREFIX_PATH" >> ~/.bashrc
                echo "✅ Добавлено в ~/.bashrc"
            fi

            # Экспорт для текущей сессии
            export LD_LIBRARY_PATH="$PYTORCH_PATH/lib:$LD_LIBRARY_PATH"
            export CMAKE_PREFIX_PATH="$PYTORCH_PATH:$CMAKE_PREFIX_PATH"
        fi
    fi
else
    echo "⏭️  Пропуск установки PyTorch (базовая установка)"
    echo "   Для установки используйте флаг --with-pytorch"
fi

echo

# ========================================
# 4. ПРОВЕРКА УСТАНОВКИ
# ========================================

echo "✅ Шаг 4/4: Проверка установки"
echo "----------------------------------------"

echo "🔍 Проверка компилятора:"
if command -v g++ &> /dev/null; then
    GCC_VERSION=$(g++ --version | head -n1)
    echo "   ✅ $GCC_VERSION"
else
    echo "   ❌ g++ не найден"
fi

echo "🔍 Проверка CMake:"
if command -v cmake &> /dev/null; then
    CMAKE_VERSION=$(cmake --version | head -n1)
    echo "   ✅ $CMAKE_VERSION"
else
    echo "   ❌ CMake не найден"
fi

echo "🔍 Проверка OpenCV:"
if [[ "$OS_TYPE" == "linux" ]] && pkg-config --exists opencv4; then
    OPENCV_VERSION=$(pkg-config --modversion opencv4)
    echo "   ✅ OpenCV $OPENCV_VERSION"
elif [[ "$OS_TYPE" == "macos" ]] && brew list opencv &> /dev/null; then
    echo "   ✅ OpenCV установлен"
else
    echo "   ⚠️  OpenCV статус неизвестен"
fi

echo "🔍 Проверка PyTorch:"
if [[ -d "$PYTORCH_PATH" ]]; then
    echo "   ✅ PyTorch найден в $PYTORCH_PATH"
    PYTORCH_SIZE=$(du -sh "$PYTORCH_PATH" | cut -f1)
    echo "   📊 Размер: $PYTORCH_SIZE"
else
    echo "   ⏭️  PyTorch не установлен (базовая конфигурация)"
fi

echo

# ========================================
# ИТОГОВЫЙ ОТЧЕТ
# ========================================

echo "🎉 =================================================="
echo "   УСТАНОВКА ЗАВЕРШЕНА"
echo "=================================================="
echo
echo "📋 Установленные компоненты:"
echo "   ✅ Компилятор C++ (g++)"
echo "   ✅ Система сборки (CMake)"
echo "   ✅ Библиотека компьютерного зрения (OpenCV)"
if [[ -d "$PYTORCH_PATH" ]]; then
    echo "   ✅ PyTorch C++ для классификатора"
else
    echo "   ⏭️  PyTorch (не установлен - базовая конфигурация)"
fi
echo

echo "📝 Следующие шаги:"
echo
echo "1. Перейдите в директорию проекта:"
echo "   cd /путь/к/GRADIENT_BASED_OPTIMIZER"
echo
echo "2. Запустите скрипт сборки и запуска:"
echo "   ./deploy_and_run.sh"
echo
if [[ -d "$PYTORCH_PATH" ]]; then
    echo "3. Для сборки с PyTorch используйте:"
    echo "   ./deploy_and_run.sh --with-pytorch"
else
    echo "3. Для установки PyTorch позже используйте:"
    echo "   ./setup_environment.sh --with-pytorch"
fi
echo

if [[ "$OS_TYPE" == "linux" ]] && [[ -d "$PYTORCH_PATH" ]]; then
    echo "⚠️  ВАЖНО: Для применения изменений в .bashrc выполните:"
    echo "   source ~/.bashrc"
    echo "   или перезапустите терминал"
    echo
fi

echo "📚 Документация:"
echo "   README.md - Общая информация"
echo "   BUILD_INSTRUCTIONS.md - Детальная инструкция по сборке"
echo "   CLAUDE.md - Полное описание проекта"
echo
echo "✅ Установка окружения завершена успешно!"
