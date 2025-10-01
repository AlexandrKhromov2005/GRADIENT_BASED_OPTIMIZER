#!/bin/bash

echo "🚀 Запуск эксперимента с классификатором final_model.pt"
echo "=================================================="

# Проверка зависимостей
check_dependencies() {
    echo "🔍 Проверка зависимостей..."
    
    # Проверка PyTorch
    if [ ! -d "/tmp/libtorch" ]; then
        echo "❌ PyTorch не найден в /tmp/libtorch"
        echo "📥 Скачиваю PyTorch..."
        cd /tmp
        wget -q https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcpu.zip -O libtorch.zip
        unzip -q libtorch.zip
        echo "✅ PyTorch установлен"
        cd - > /dev/null
    else
        echo "✅ PyTorch найден"
    fi
    
    # Проверка модели
    if [ ! -f "final_model_torchscript.pt" ]; then
        echo "❌ Модель final_model_torchscript.pt не найдена"
        if [ -f "best_scheme_classifier_torchscript.pt" ]; then
            echo "📋 Копирую тестовую модель..."
            cp best_scheme_classifier_torchscript.pt final_model_torchscript.pt
            echo "✅ Тестовая модель подготовлена"
        else
            echo "❌ Не найдена ни одна TorchScript модель"
            exit 1
        fi
    else
        echo "✅ Модель final_model_torchscript.pt найдена"
    fi
    
    # Проверка изображений
    if [ ! -d "images" ] || [ $(ls images/*.png 2>/dev/null | wc -l) -eq 0 ]; then
        echo "❌ Изображения не найдены в папке images/"
        exit 1
    else
        local img_count=$(ls images/*.png | wc -l)
        echo "✅ Найдено $img_count изображений"
    fi
    
    # Проверка сборки
    if [ ! -f "build/gradient_based_optimizer" ]; then
        echo "📦 Сборка проекта..."
        mkdir -p build
        cd build
        cmake ..
        make gradient_based_optimizer -j$(nproc)
        cd ..
        
        if [ ! -f "build/gradient_based_optimizer" ]; then
            echo "❌ Ошибка сборки"
            exit 1
        fi
        echo "✅ Проект собран"
    else
        echo "✅ Исполняемый файл найден"
    fi
}

# Показать информацию о системе
show_system_info() {
    echo ""
    echo "💻 Информация о системе:"
    echo "  OS: $(uname -s) $(uname -r)"
    echo "  Архитектура: $(uname -m)"
    echo "  Процессоры: $(nproc)"
    echo "  Модель: $(ls -lah final_model_torchscript.pt | awk '{print $5}')"
    echo "  Изображений: $(ls images/*.png | wc -l)"
    echo ""
}

# Выбор режима запуска
choose_mode() {
    echo "🎯 Выберите режим эксперимента:"
    echo "1) Быстрый тест (--test) - 1 итерация, ~2-5 минут"
    echo "2) Полный эксперимент - все итерации, ~15-30 минут"
    echo "3) Ограниченный набор (5 изображений) - ~5-10 минут"
    echo "4) Только определенные изображения"
    echo "5) Отмена"
    
    read -p "Введите номер (1-5): " choice
    
    case $choice in
        1) return 1 ;;  # test mode
        2) return 2 ;;  # full mode
        3) return 3 ;;  # limited mode
        4) return 4 ;;  # custom mode
        5) echo "❌ Отмена"; exit 0 ;;
        *) echo "❌ Неверный выбор"; exit 1 ;;
    esac
}

# Запуск быстрого теста
run_test() {
    echo "⚡ Запуск быстрого теста..."
    echo "Время выполнения: ~2-5 минут"
    echo ""
    
    timeout 600 ./build/gradient_based_optimizer --dataset-classifier --test
    local exit_code=$?
    
    if [ $exit_code -eq 124 ]; then
        echo "⏰ Тест остановлен по таймауту (10 минут)"
    elif [ $exit_code -eq 0 ]; then
        echo "✅ Тест завершен успешно"
    else
        echo "❌ Тест завершен с ошибкой (код: $exit_code)"
    fi
}

# Запуск полного эксперимента
run_full() {
    echo "🚀 Запуск полного эксперимента..."
    echo "⚠️  Время выполнения: ~15-30 минут"
    echo "💡 Для остановки нажмите Ctrl+C"
    echo ""
    
    read -p "Продолжить? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Отмена"
        exit 0
    fi
    
    timeout 3600 ./build/gradient_based_optimizer --dataset-classifier
    local exit_code=$?
    
    if [ $exit_code -eq 124 ]; then
        echo "⏰ Эксперимент остановлен по таймауту (1 час)"
    elif [ $exit_code -eq 0 ]; then
        echo "✅ Эксперимент завершен успешно"
    else
        echo "❌ Эксперимент завершен с ошибкой (код: $exit_code)"
    fi
}

# Запуск с ограниченным набором
run_limited() {
    echo "📊 Запуск с ограниченным набором (5 изображений)..."
    
    # Backup and limit images
    if [ ! -d "images_backup_full" ]; then
        cp -r images images_backup_full
    fi
    
    mkdir -p images_limited
    ls images/*.png | head -5 | xargs -I {} cp {} images_limited/
    cp images/watermark.png images_limited/ 2>/dev/null || true
    
    mv images images_all
    mv images_limited images
    
    timeout 900 ./build/gradient_based_optimizer --dataset-classifier
    local exit_code=$?
    
    # Restore images
    mv images images_limited
    mv images_all images
    
    if [ $exit_code -eq 124 ]; then
        echo "⏰ Эксперимент остановлен по таймауту (15 минут)"
    elif [ $exit_code -eq 0 ]; then
        echo "✅ Ограниченный эксперимент завершен успешно"
    else
        echo "❌ Ограниченный эксперимент завершен с ошибкой"
    fi
}

# Показать результаты
show_results() {
    if [ -d "dataset_classifier" ]; then
        echo ""
        echo "📊 РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА:"
        echo "=========================="
        
        local scheme2=$(find dataset_classifier/scheme2_selected -name "*.png" 2>/dev/null | wc -l)
        local scheme3=$(find dataset_classifier/scheme3_selected -name "*.png" 2>/dev/null | wc -l)
        local correct=$(find dataset_classifier/extraction_correct -name "*.png" 2>/dev/null | wc -l)
        local incorrect=$(find dataset_classifier/extraction_incorrect -name "*.png" 2>/dev/null | wc -l)
        local total=$((correct + incorrect))
        local total_schemes=$((scheme2 + scheme3))
        
        echo "🎯 Выбор схем:"
        echo "  Scheme2: $scheme2 блоков"
        echo "  Scheme3: $scheme3 блоков"
        echo "  Всего: $total_schemes блоков"
        
        if [ $total_schemes -gt 0 ]; then
            local scheme2_percent=$(echo "scale=1; $scheme2 * 100 / $total_schemes" | bc -l)
            local scheme3_percent=$(echo "scale=1; $scheme3 * 100 / $total_schemes" | bc -l)
            echo "  Распределение: Scheme2 $scheme2_percent%, Scheme3 $scheme3_percent%"
        fi
        
        echo ""
        echo "🔍 Точность извлечения:"
        echo "  Правильно: $correct блоков"
        echo "  Неправильно: $incorrect блоков"
        echo "  Всего: $total блоков"
        
        if [ $total -gt 0 ]; then
            local accuracy=$(echo "scale=2; $correct * 100 / $total" | bc -l)
            echo "  🎯 Точность: $accuracy%"
        fi
        
        echo ""
        echo "📁 Результаты сохранены в: dataset_classifier/"
        echo "📋 Подробный отчет: EXPERIMENT_RESULTS.md"
    else
        echo "❌ Результаты не найдены"
    fi
}

# Основной скрипт
main() {
    check_dependencies
    show_system_info
    
    choose_mode
    mode=$?
    
    echo ""
    echo "⏱️  Начало эксперимента: $(date)"
    echo ""
    
    case $mode in
        1) run_test ;;
        2) run_full ;;
        3) run_limited ;;
        4) 
            echo "📝 Пользовательский режим пока не реализован"
            echo "💡 Используйте: ./build/gradient_based_optimizer --dataset-classifier [--test]"
            exit 0
            ;;
    esac
    
    echo ""
    echo "⏱️  Окончание: $(date)"
    
    show_results
    
    echo ""
    echo "🏁 Эксперимент завершен!"
    echo "💡 Для очистки результатов используйте: ./clean_experiment_results.sh"
}

# Запуск
main "$@"