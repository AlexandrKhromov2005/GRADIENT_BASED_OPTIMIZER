#!/bin/bash

echo "🔍 =================================================="
echo "   ПРОВЕРКА ГОТОВНОСТИ К ПЕРЕНОСУ"
echo "=================================================="
echo

errors=0
warnings=0

# ========================================
# Проверка скриптов
# ========================================

echo "📜 Проверка скриптов развертывания:"
echo "----------------------------------------"

for script in setup_environment.sh deploy_and_run.sh; do
    if [ -f "$script" ] && [ -x "$script" ]; then
        size=$(du -h "$script" | cut -f1)
        echo "   ✅ $script ($size)"
    elif [ -f "$script" ]; then
        echo "   ⚠️  $script - не исполняемый (запустите: chmod +x $script)"
        warnings=$((warnings + 1))
    else
        echo "   ❌ $script - не найден"
        errors=$((errors + 1))
    fi
done

# Проверка остальных скриптов
for script in build.sh run_quick_experiment.sh run_classifier_experiment.sh; do
    if [ -f "$script" ]; then
        echo "   ✅ $script (опционально)"
    fi
done

echo

# ========================================
# Проверка документации
# ========================================

echo "📚 Проверка документации:"
echo "----------------------------------------"

required_docs="DEPLOYMENT_GUIDE.md QUICK_START.md"
optional_docs="SCRIPTS_README.md TRANSFER_CHECKLIST.md DEPLOYMENT_FILES.md CLAUDE.md README.md BUILD_INSTRUCTIONS.md"

for doc in $required_docs; do
    if [ -f "$doc" ]; then
        size=$(du -h "$doc" | cut -f1)
        lines=$(wc -l < "$doc")
        echo "   ✅ $doc ($size, $lines строк)"
    else
        echo "   ❌ $doc - не найден (рекомендуется)"
        warnings=$((warnings + 1))
    fi
done

for doc in $optional_docs; do
    if [ -f "$doc" ]; then
        echo "   ✅ $doc (опционально)"
    fi
done

echo

# ========================================
# Проверка исходного кода
# ========================================

echo "💻 Проверка исходного кода:"
echo "----------------------------------------"

if [ -d "GRADIENT_BASED_OPTIMIZER" ]; then
    echo "   ✅ Директория GRADIENT_BASED_OPTIMIZER/"

    if [ -d "GRADIENT_BASED_OPTIMIZER/src" ]; then
        cpp_count=$(find GRADIENT_BASED_OPTIMIZER/src -name "*.cpp" 2>/dev/null | wc -l)
        h_count=$(find GRADIENT_BASED_OPTIMIZER/src -name "*.h" 2>/dev/null | wc -l)
        echo "   ✅ Исходный код: $cpp_count файлов .cpp, $h_count файлов .h"
    else
        echo "   ❌ Директория src/ не найдена"
        errors=$((errors + 1))
    fi

    if [ -f "GRADIENT_BASED_OPTIMIZER/GRADIENT_BASED_OPTIMIZER.cpp" ]; then
        echo "   ✅ Главный файл GRADIENT_BASED_OPTIMIZER.cpp"
    else
        echo "   ❌ GRADIENT_BASED_OPTIMIZER.cpp не найден"
        errors=$((errors + 1))
    fi
else
    echo "   ❌ Директория GRADIENT_BASED_OPTIMIZER/ не найдена"
    errors=$((errors + 1))
fi

echo

# ========================================
# Проверка изображений
# ========================================

echo "🖼️  Проверка изображений:"
echo "----------------------------------------"

if [ -d "images" ]; then
    echo "   ✅ Директория images/"

    if [ -f "images/watermark.png" ]; then
        size=$(du -h "images/watermark.png" | cut -f1)
        dimensions=$(file "images/watermark.png" 2>/dev/null | grep -o '[0-9]\+ x [0-9]\+' || echo "неизвестно")
        echo "   ✅ watermark.png ($size, $dimensions) - КРИТИЧНЫЙ ФАЙЛ"
    else
        echo "   ❌ watermark.png не найден - КРИТИЧНО!"
        errors=$((errors + 1))
    fi

    image_count=$(find images -maxdepth 1 -name "*.png" -o -name "*.jpg" 2>/dev/null | wc -l)
    if [ $image_count -gt 1 ]; then
        echo "   ✅ Тестовые изображения: $image_count файлов"
        echo "      Примеры:"
        find images -maxdepth 1 -name "*.png" -o -name "*.jpg" 2>/dev/null | head -5 | while read img; do
            echo "      - $(basename "$img")"
        done
    else
        echo "   ⚠️  Мало тестовых изображений: $image_count"
        echo "      Добавьте изображения для тестирования"
        warnings=$((warnings + 1))
    fi
else
    echo "   ❌ Директория images/ не найдена"
    errors=$((errors + 1))
fi

echo

# ========================================
# Проверка конфигурации
# ========================================

echo "⚙️  Проверка конфигурации:"
echo "----------------------------------------"

if [ -f "CMakeLists.txt" ]; then
    size=$(du -h "CMakeLists.txt" | cut -f1)
    echo "   ✅ CMakeLists.txt ($size)"
else
    echo "   ❌ CMakeLists.txt не найден"
    errors=$((errors + 1))
fi

if [ -f "embedding_schemes.json" ]; then
    size=$(du -h "embedding_schemes.json" | cut -f1)

    # Проверка валидности JSON
    if command -v python3 &> /dev/null; then
        if python3 -c "import json; json.load(open('embedding_schemes.json'))" 2>/dev/null; then
            schemes=$(python3 -c "import json; print(len(json.load(open('embedding_schemes.json'))['schemes']))" 2>/dev/null || echo "?")
            echo "   ✅ embedding_schemes.json ($size, $schemes схем)"
        else
            echo "   ⚠️  embedding_schemes.json - невалидный JSON"
            warnings=$((warnings + 1))
        fi
    else
        echo "   ✅ embedding_schemes.json ($size)"
    fi
else
    echo "   ❌ embedding_schemes.json не найден"
    errors=$((errors + 1))
fi

echo

# ========================================
# Проверка моделей классификатора
# ========================================

echo "🤖 Проверка моделей классификатора (опционально):"
echo "----------------------------------------"

models_found=false

if [ -f "final_model_torchscript.pt" ]; then
    size=$(du -h "final_model_torchscript.pt" | cut -f1)
    echo "   ✅ final_model_torchscript.pt ($size)"
    models_found=true
else
    echo "   ⏭️  final_model_torchscript.pt не найден"
fi

if [ -f "best_scheme_classifier_torchscript.pt" ]; then
    size=$(du -h "best_scheme_classifier_torchscript.pt" | cut -f1)
    echo "   ✅ best_scheme_classifier_torchscript.pt ($size)"
    models_found=true
else
    echo "   ⏭️  best_scheme_classifier_torchscript.pt не найден"
fi

if [ -f "ensemble_model_1_torchscript.pt" ]; then
    size=$(du -h "ensemble_model_1_torchscript.pt" | cut -f1)
    echo "   ✅ ensemble_model_1_torchscript.pt ($size)"
    models_found=true
else
    echo "   ⏭️  ensemble_model_1_torchscript.pt не найден"
fi

if [ "$models_found" = false ]; then
    echo "   ℹ️  Модели не найдены - будет доступна только базовая функциональность"
    echo "      Классификатор требует установки PyTorch и моделей"
fi

echo

# ========================================
# Проверка build директории
# ========================================

echo "🔨 Проверка сборки:"
echo "----------------------------------------"

if [ -d "build" ]; then
    build_size=$(du -sh "build" 2>/dev/null | cut -f1)
    echo "   ⚠️  Директория build/ существует ($build_size)"
    echo "      Рекомендуется удалить перед переносом для уменьшения размера"

    if [ -f "build/gradient_based_optimizer" ]; then
        echo "      ✅ Проект уже собран на этой машине"
    fi
else
    echo "   ✅ Директория build/ отсутствует (хорошо для переноса)"
fi

echo

# ========================================
# Проверка результатов экспериментов
# ========================================

echo "📊 Проверка результатов экспериментов:"
echo "----------------------------------------"

results_found=false

if [ -d "dataset" ]; then
    dataset_size=$(du -sh "dataset" 2>/dev/null | cut -f1)
    echo "   ⚠️  Директория dataset/ существует ($dataset_size)"
    echo "      Рекомендуется удалить перед переносом или сохранить отдельно"
    results_found=true
fi

if [ -d "dataset_classifier" ]; then
    dataset_size=$(du -sh "dataset_classifier" 2>/dev/null | cut -f1)
    echo "   ⚠️  Директория dataset_classifier/ существует ($dataset_size)"
    echo "      Рекомендуется удалить перед переносом или сохранить отдельно"
    results_found=true
fi

if [ "$results_found" = false ]; then
    echo "   ✅ Результаты экспериментов отсутствуют (хорошо для переноса)"
fi

echo

# ========================================
# Оценка размера
# ========================================

echo "💾 Оценка размера проекта:"
echo "----------------------------------------"

total_size=$(du -sh . 2>/dev/null | cut -f1)
echo "   Общий размер: $total_size"

if [ -d "build" ] || [ -d "dataset" ] || [ -d "dataset_classifier" ]; then
    minimal_size=$(du -sh --exclude='build' --exclude='dataset*' . 2>/dev/null | cut -f1 || echo "неизвестно")
    echo "   Размер без build/dataset: $minimal_size"
fi

echo

# ========================================
# ИТОГОВЫЙ ОТЧЕТ
# ========================================

echo "📋 =================================================="
echo "   ИТОГОВЫЙ ОТЧЕТ"
echo "=================================================="
echo

if [ $errors -eq 0 ] && [ $warnings -eq 0 ]; then
    echo "🎉 ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ!"
    echo
    echo "✅ Проект полностью готов к переносу"
    echo
    echo "Следующие шаги:"
    echo "1. Создать архив: tar -czf gradient_optimizer.tar.gz --exclude='build' --exclude='dataset*' ."
    echo "2. Перенести на новый компьютер"
    echo "3. Распаковать и запустить: ./setup_environment.sh --with-pytorch"
    echo
    exit 0

elif [ $errors -eq 0 ]; then
    echo "⚠️  ПРОВЕРКИ ПРОЙДЕНЫ С ПРЕДУПРЕЖДЕНИЯМИ"
    echo
    echo "Предупреждений: $warnings"
    echo
    echo "✅ Проект готов к переносу, но есть рекомендации:"
    echo

    if [ -d "build" ]; then
        echo "   • Удалить build/ перед архивацией: rm -rf build"
    fi

    if [ -d "dataset" ] || [ -d "dataset_classifier" ]; then
        echo "   • Сохранить результаты отдельно и удалить: rm -rf dataset*"
    fi

    echo
    echo "Или создать архив исключая временные файлы:"
    echo "tar -czf gradient_optimizer.tar.gz --exclude='build' --exclude='dataset*' ."
    echo
    exit 0

else
    echo "❌ ОБНАРУЖЕНЫ КРИТИЧЕСКИЕ ПРОБЛЕМЫ"
    echo
    echo "Ошибок: $errors"
    echo "Предупреждений: $warnings"
    echo
    echo "Необходимо исправить следующие проблемы:"
    echo

    if [ ! -d "GRADIENT_BASED_OPTIMIZER/src" ]; then
        echo "   • Отсутствует исходный код проекта"
    fi

    if [ ! -f "images/watermark.png" ]; then
        echo "   • Отсутствует критичный файл watermark.png"
    fi

    if [ ! -f "CMakeLists.txt" ]; then
        echo "   • Отсутствует CMakeLists.txt"
    fi

    if [ ! -f "setup_environment.sh" ] || [ ! -f "deploy_and_run.sh" ]; then
        echo "   • Отсутствуют скрипты развертывания"
    fi

    echo
    echo "Исправьте проблемы перед переносом проекта"
    echo
    exit 1
fi
