#!/bin/bash

echo "🚀 Полный эксперимент с анализом метрик качества"
echo "================================================"

# Функция проверки зависимостей
check_dependencies() {
    echo "🔍 Проверка зависимостей..."
    
    if [ ! -f "final_model_torchscript.pt" ]; then
        echo "❌ Модель final_model_torchscript.pt не найдена"
        if [ -f "best_scheme_classifier_torchscript.pt" ]; then
            echo "📋 Копирую тестовую модель..."
            cp best_scheme_classifier_torchscript.pt final_model_torchscript.pt
        else
            echo "❌ Не найдена ни одна TorchScript модель"
            exit 1
        fi
    fi
    
    # Проверка сборки основного проекта
    if [ ! -f "build/gradient_based_optimizer" ]; then
        echo "📦 Сборка основного проекта..."
        mkdir -p build && cd build
        cmake .. && make gradient_based_optimizer -j$(nproc)
        cd ..
        
        if [ ! -f "build/gradient_based_optimizer" ]; then
            echo "❌ Ошибка сборки основного проекта"
            exit 1
        fi
    fi
    
    # Сборка анализатора качества
    echo "🔧 Сборка анализатора качества..."
    make -f Makefile.quality clean
    make -f Makefile.quality quality_analyzer
    
    if [ ! -f "quality_analyzer" ]; then
        echo "❌ Ошибка сборки анализатора качества"
        exit 1
    fi
    
    echo "✅ Все зависимости готовы"
}

# Запуск основного эксперимента с классификатором
run_classifier_experiment() {
    echo ""
    echo "🤖 Шаг 1: Запуск эксперимента с классификатором"
    echo "=============================================="
    
    # Очистка предыдущих результатов классификатора
    rm -rf dataset_classifier/
    
    # Выбор режима
    echo "Выберите режим:"
    echo "1) Быстрый тест (--test)"
    echo "2) Полный эксперимент"
    
    read -p "Введите номер (1-2): " mode_choice
    
    if [ "$mode_choice" = "1" ]; then
        echo "⚡ Запуск быстрого теста..."
        timeout 600 ./build/gradient_based_optimizer --dataset-classifier --test
    else
        echo "🚀 Запуск полного эксперимента..."
        echo "⚠️  Это может занять 15-30 минут"
        read -p "Продолжить? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "❌ Отмена"
            exit 0
        fi
        timeout 3600 ./build/gradient_based_optimizer --dataset-classifier
    fi
    
    local exit_code=$?
    if [ $exit_code -eq 124 ]; then
        echo "⏰ Эксперимент остановлен по таймауту"
    elif [ $exit_code -eq 0 ]; then
        echo "✅ Эксперимент с классификатором завершен"
    else
        echo "❌ Эксперимент завершен с ошибкой (код: $exit_code)"
    fi
}

# Запуск обычного эксперимента для получения обработанных изображений
run_image_processing() {
    echo ""
    echo "📸 Шаг 2: Обработка изображений для анализа качества"
    echo "=================================================="
    
    # Запускаем обычный режим для получения обработанных изображений
    echo "🔄 Обработка изображений с классификатором..."
    
    # Запускаем режим classifier для получения обработанных изображений
    timeout 1800 ./build/gradient_based_optimizer --classifier --test
    
    local exit_code=$?
    if [ $exit_code -eq 124 ]; then
        echo "⏰ Обработка изображений остановлена по таймауту"
    elif [ $exit_code -eq 0 ]; then
        echo "✅ Изображения обработаны"
    else
        echo "⚠️  Обработка завершена с кодом: $exit_code"
    fi
}

# Анализ качества изображений
analyze_quality() {
    echo ""
    echo "📊 Шаг 3: Анализ качества изображений"
    echo "====================================="
    
    echo "🔍 Запуск анализатора качества..."
    ./quality_analyzer
    
    if [ $? -eq 0 ]; then
        echo "✅ Анализ качества завершен"
    else
        echo "❌ Ошибка при анализе качества"
    fi
}

# Сводный отчет
generate_summary() {
    echo ""
    echo "📋 Шаг 4: Создание сводного отчета"
    echo "=================================="
    
    local report_file="FULL_METRICS_REPORT.md"
    
    echo "# Полный отчет эксперимента с final_model.pt" > $report_file
    echo "" >> $report_file
    echo "## Дата эксперимента: $(date)" >> $report_file
    echo "" >> $report_file
    
    # Результаты классификатора
    if [ -d "dataset_classifier" ]; then
        local scheme2=$(find dataset_classifier/scheme2_selected -name "*.png" 2>/dev/null | wc -l)
        local scheme3=$(find dataset_classifier/scheme3_selected -name "*.png" 2>/dev/null | wc -l)
        local correct=$(find dataset_classifier/extraction_correct -name "*.png" 2>/dev/null | wc -l)
        local incorrect=$(find dataset_classifier/extraction_incorrect -name "*.png" 2>/dev/null | wc -l)
        local total=$((correct + incorrect))
        
        echo "## 🤖 Результаты классификатора" >> $report_file
        echo "" >> $report_file
        echo "| Метрика | Значение |" >> $report_file
        echo "|---------|----------|" >> $report_file
        echo "| Scheme2 выбрано | $scheme2 блоков |" >> $report_file
        echo "| Scheme3 выбрано | $scheme3 блоков |" >> $report_file
        echo "| Правильно извлечено | $correct блоков |" >> $report_file
        echo "| Неправильно извлечено | $incorrect блоков |" >> $report_file
        
        if [ $total -gt 0 ]; then
            local accuracy=$(echo "scale=2; $correct * 100 / $total" | bc -l)
            echo "| **Точность** | **$accuracy%** |" >> $report_file
        fi
        echo "" >> $report_file
    fi
    
    # Метрики качества изображений
    if [ -f "image_quality_report.txt" ]; then
        echo "## 📊 Метрики качества изображений" >> $report_file
        echo "" >> $report_file
        echo "\`\`\`" >> $report_file
        cat image_quality_report.txt >> $report_file
        echo "\`\`\`" >> $report_file
        echo "" >> $report_file
    fi
    
    echo "## 📁 Структура результатов" >> $report_file
    echo "" >> $report_file
    echo "\`\`\`" >> $report_file
    echo "Результаты эксперимента:" >> $report_file
    if [ -d "dataset_classifier" ]; then
        echo "dataset_classifier/ - результаты классификации блоков" >> $report_file
    fi
    if [ -f "image_quality_report.txt" ]; then
        echo "image_quality_report.txt - детальный анализ качества" >> $report_file
    fi
    echo "$report_file - этот сводный отчет" >> $report_file
    echo "\`\`\`" >> $report_file
    
    echo "✅ Сводный отчет создан: $report_file"
}

# Показать результаты
show_results() {
    echo ""
    echo "🎯 ИТОГОВЫЕ РЕЗУЛЬТАТЫ"
    echo "====================="
    
    # Результаты классификатора
    if [ -d "dataset_classifier" ]; then
        local scheme2=$(find dataset_classifier/scheme2_selected -name "*.png" 2>/dev/null | wc -l)
        local scheme3=$(find dataset_classifier/scheme3_selected -name "*.png" 2>/dev/null | wc -l)
        local correct=$(find dataset_classifier/extraction_correct -name "*.png" 2>/dev/null | wc -l)
        local incorrect=$(find dataset_classifier/extraction_incorrect -name "*.png" 2>/dev/null | wc -l)
        local total=$((correct + incorrect))
        
        echo "🤖 Классификатор:"
        echo "  Scheme2: $scheme2, Scheme3: $scheme3"
        if [ $total -gt 0 ]; then
            local accuracy=$(echo "scale=2; $correct * 100 / $total" | bc -l)
            echo "  Точность: $accuracy% ($correct/$total)"
        fi
    fi
    
    # Показать краткие метрики качества
    if [ -f "image_quality_report.txt" ]; then
        echo ""
        echo "📊 Качество изображений:"
        grep -E "(PSNR|SSIM|BER).*среднее" image_quality_report.txt 2>/dev/null || \
        echo "  Смотрите подробности в image_quality_report.txt"
    fi
    
    echo ""
    echo "📋 Отчеты:"
    [ -f "FULL_METRICS_REPORT.md" ] && echo "  📄 FULL_METRICS_REPORT.md - сводный отчет"
    [ -f "image_quality_report.txt" ] && echo "  📄 image_quality_report.txt - анализ качества"
    [ -d "dataset_classifier" ] && echo "  📁 dataset_classifier/ - результаты классификации"
}

# Основная функция
main() {
    echo "⏱️  Начало: $(date)"
    
    check_dependencies
    run_classifier_experiment
    run_image_processing  
    analyze_quality
    generate_summary
    show_results
    
    echo ""
    echo "⏱️  Завершение: $(date)"
    echo "🎉 Полный эксперимент завершен!"
    echo ""
    echo "💡 Для очистки результатов: ./clean_experiment_results.sh"
}

# Запуск
main "$@"