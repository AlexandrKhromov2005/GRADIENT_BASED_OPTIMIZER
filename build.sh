#!/bin/bash

# 🔧 Скрипт автоматической сборки проекта

echo "🚀 Автоматическая сборка GRADIENT_BASED_OPTIMIZER"
echo

# Проверяем наличие PyTorch
LIBTORCH_PATH="$HOME/libtorch"
if [ -d "$LIBTORCH_PATH" ]; then
    echo "✅ PyTorch найден в $LIBTORCH_PATH"
    CMAKE_ARGS="-DCMAKE_PREFIX_PATH=$LIBTORCH_PATH"
    BUILD_TYPE="ПОЛНАЯ (с классификатором)"
else
    echo "⚠️  PyTorch не найден, базовая сборка"
    CMAKE_ARGS=""
    BUILD_TYPE="БАЗОВАЯ (без классификатора)"
fi

echo "📦 Тип сборки: $BUILD_TYPE"
echo

# Создаем папку сборки
echo "📁 Создание папки build..."
mkdir -p build
cd build

# Очищаем предыдущую сборку
echo "🧹 Очистка предыдущей сборки..."
rm -rf *

# Конфигурируем проект
echo "⚙️  Конфигурация проекта..."
if cmake $CMAKE_ARGS ..; then
    echo "✅ Конфигурация успешна"
else
    echo "❌ Ошибка конфигурации"
    exit 1
fi

echo

# Собираем проект
echo "🔨 Сборка проекта..."
if make -j4; then
    echo
    echo "🎉 СБОРКА УСПЕШНА!"
    echo
    
    # Показываем что собрали
    echo "📦 Собранные файлы:"
    ls -la gradient_based_optimizer classifier_example 2>/dev/null || ls -la gradient_based_optimizer
    echo
    
    # Тестируем сборку
    echo "🧪 Тестирование сборки..."
    cd ..
    echo "Справка проекта:"
    ./build/gradient_based_optimizer --help
    
else
    echo
    echo "❌ ОШИБКА СБОРКИ"
    exit 1
fi