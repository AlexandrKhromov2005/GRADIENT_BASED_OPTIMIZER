#!/bin/bash

echo "📊 Анализ существующих результатов эксперимента"
echo "=============================================="

# Проверка результатов классификатора
check_classifier_results() {
    if [ -d "dataset_classifier" ]; then
        echo "🤖 РЕЗУЛЬТАТЫ КЛАССИФИКАТОРА:"
        echo "============================"
        
        local scheme2=$(find dataset_classifier/scheme2_selected -name "*.png" 2>/dev/null | wc -l)
        local scheme3=$(find dataset_classifier/scheme3_selected -name "*.png" 2>/dev/null | wc -l)
        local correct=$(find dataset_classifier/extraction_correct -name "*.png" 2>/dev/null | wc -l)
        local incorrect=$(find dataset_classifier/extraction_incorrect -name "*.png" 2>/dev/null | wc -l)
        local total=$((correct + incorrect))
        local total_schemes=$((scheme2 + scheme3))
        
        echo "📊 Выбор схем встраивания:"
        echo "  Scheme2 (Alternative): $scheme2 блоков"
        echo "  Scheme3 (Variable Size): $scheme3 блоков"
        echo "  Всего обработано: $total_schemes блоков"
        
        if [ $total_schemes -gt 0 ]; then
            local scheme2_percent=$(echo "scale=1; $scheme2 * 100 / $total_schemes" | bc -l)
            local scheme3_percent=$(echo "scale=1; $scheme3 * 100 / $total_schemes" | bc -l)
            echo "  Распределение: Scheme2 $scheme2_percent%, Scheme3 $scheme3_percent%"
        fi
        
        echo ""
        echo "🔍 Точность извлечения водяных знаков:"
        echo "  Правильно извлечено: $correct блоков"
        echo "  Неправильно извлечено: $incorrect блоков"
        echo "  Всего: $total блоков"
        
        if [ $total -gt 0 ]; then
            local accuracy=$(echo "scale=2; $correct * 100 / $total" | bc -l)
            local error_rate=$(echo "scale=2; $incorrect * 100 / $total" | bc -l)
            echo "  🎯 Точность: $accuracy%"
            echo "  ❌ Частота ошибок: $error_rate%"
        fi
        
        echo ""
        return 0
    else
        echo "❌ Результаты классификатора не найдены (dataset_classifier/)"
        return 1
    fi
}

# Анализ качества изображений
analyze_image_quality() {
    echo "📸 АНАЛИЗ КАЧЕСТВА ИЗОБРАЖЕНИЙ:"
    echo "=============================="
    
    # Проверяем наличие обработанных изображений
    local processed_count=$(find images -name "new_*.png" 2>/dev/null | wc -l)
    
    if [ $processed_count -gt 0 ]; then
        echo "✅ Найдено $processed_count обработанных изображений"
        
        # Собираем анализатор если его нет
        if [ ! -f "quality_analyzer" ]; then
            echo "🔧 Сборка анализатора качества..."
            make -f Makefile.quality quality_analyzer >/dev/null 2>&1
        fi
        
        if [ -f "quality_analyzer" ]; then
            echo "🔍 Запуск анализа..."
            ./quality_analyzer
        else
            echo "❌ Не удалось собрать анализатор качества"
            echo "💡 Попробуйте: make -f Makefile.quality quality_analyzer"
        fi
    else
        echo "❌ Обработанные изображения не найдены"
        echo "💡 Запустите эксперимент: ./build/gradient_based_optimizer --classifier"
    fi
    
    echo ""
}

# Сводка файлов результатов
summarize_files() {
    echo "📁 ФАЙЛЫ РЕЗУЛЬТАТОВ:"
    echo "===================="
    
    local found_results=false
    
    if [ -d "dataset_classifier" ]; then
        local size=$(du -sh dataset_classifier 2>/dev/null | cut -f1)
        echo "📂 dataset_classifier/ - результаты классификации ($size)"
        found_results=true
    fi
    
    if [ -f "image_quality_report.txt" ]; then
        local size=$(ls -lah image_quality_report.txt | awk '{print $5}')
        echo "📄 image_quality_report.txt - анализ качества изображений ($size)"
        found_results=true
    fi
    
    if [ -f "EXPERIMENT_RESULTS.md" ]; then
        echo "📄 EXPERIMENT_RESULTS.md - детальный отчет эксперимента"
        found_results=true
    fi
    
    if [ -f "FINAL_EXPERIMENT_SUMMARY.md" ]; then
        echo "📄 FINAL_EXPERIMENT_SUMMARY.md - итоговый отчет"
        found_results=true
    fi
    
    if [ -f "FULL_METRICS_REPORT.md" ]; then
        echo "📄 FULL_METRICS_REPORT.md - полный отчет с метриками"
        found_results=true
    fi
    
    if [ ! "$found_results" = true ]; then
        echo "❌ Файлы результатов не найдены"
    fi
    
    echo ""
}

# Рекомендации
show_recommendations() {
    echo "💡 РЕКОМЕНДАЦИИ:"
    echo "==============="
    
    if [ ! -d "dataset_classifier" ]; then
        echo "🚀 Запустите эксперимент с классификатором:"
        echo "   ./run_classifier_experiment.sh"
        echo ""
    fi
    
    local processed_count=$(find images -name "new_*.png" 2>/dev/null | wc -l)
    if [ $processed_count -eq 0 ]; then
        echo "📸 Для анализа качества изображений запустите:"
        echo "   ./build/gradient_based_optimizer --classifier --test"
        echo ""
    fi
    
    if [ ! -f "image_quality_report.txt" ] && [ $processed_count -gt 0 ]; then
        echo "📊 Для получения метрик PSNR, SSIM, BER запустите:"
        echo "   make -f Makefile.quality quality_analyzer && ./quality_analyzer"
        echo ""
    fi
    
    echo "🧹 Для очистки результатов:"
    echo "   ./clean_experiment_results.sh"
    echo ""
    
    echo "📋 Для полного эксперимента с метриками:"
    echo "   ./run_full_metrics_experiment.sh"
}

# Основная функция
main() {
    echo "🕐 Анализ от: $(date)"
    echo ""
    
    local has_classifier_results=false
    
    if check_classifier_results; then
        has_classifier_results=true
    fi
    
    analyze_image_quality
    summarize_files
    
    if [ "$has_classifier_results" = true ] || [ -f "image_quality_report.txt" ]; then
        echo "✅ Результаты найдены и проанализированы!"
    else
        echo "⚠️  Результаты экспериментов не найдены"
    fi
    
    echo ""
    show_recommendations
}

# Запуск
main "$@"