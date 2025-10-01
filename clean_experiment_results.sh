#!/bin/bash

echo "🧹 Очистка результатов экспериментов..."

# Функция для безопасного удаления с подтверждением
safe_remove() {
    local dir="$1"
    local description="$2"
    
    if [ -d "$dir" ]; then
        local count=$(find "$dir" -type f | wc -l)
        echo "📁 Найдена папка $dir ($count файлов) - $description"
        
        read -p "Удалить? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$dir"
            echo "✅ Удалено: $dir"
        else
            echo "⏭️  Пропущено: $dir"
        fi
    else
        echo "ℹ️  Папка $dir не существует"
    fi
}

# Функция для принудительного удаления без подтверждения
force_remove() {
    local dir="$1"
    local description="$2"
    
    if [ -d "$dir" ]; then
        local count=$(find "$dir" -type f | wc -l)
        echo "🗑️  Удаляю $dir ($count файлов) - $description"
        rm -rf "$dir"
        echo "✅ Удалено: $dir"
    else
        echo "ℹ️  Папка $dir не существует"
    fi
}

echo ""
echo "Выберите режим очистки:"
echo "1) Интерактивная очистка (с подтверждением)"
echo "2) Полная очистка (без подтверждения)"
echo "3) Только результаты с классификатором"
echo "4) Только базовые результаты"
echo "5) Отмена"

read -p "Введите номер (1-5): " choice

case $choice in
    1)
        echo "🔍 Интерактивная очистка..."
        safe_remove "dataset_classifier" "результаты с классификатором final_model.pt"
        safe_remove "dataset" "результаты базового алгоритма"
        safe_remove "images_backup_temp" "временная резервная копия изображений"
        safe_remove "images_quick" "быстрый тест изображений"
        safe_remove "images_full" "полный набор изображений"
        ;;
    2)
        echo "⚡ Полная очистка всех результатов..."
        force_remove "dataset_classifier" "результаты с классификатором final_model.pt"
        force_remove "dataset" "результаты базового алгоритма"
        force_remove "images_backup_temp" "временная резервная копия изображений"
        force_remove "images_quick" "быстрый тест изображений"
        force_remove "images_full" "полный набор изображений"
        
        # Удаление временных файлов
        rm -f *.log *.tmp 2>/dev/null
        echo "✅ Временные файлы очищены"
        ;;
    3)
        echo "🤖 Очистка только результатов классификатора..."
        force_remove "dataset_classifier" "результаты с классификатором final_model.pt"
        ;;
    4)
        echo "📊 Очистка только базовых результатов..."
        force_remove "dataset" "результаты базового алгоритма"
        ;;
    5)
        echo "❌ Отмена"
        exit 0
        ;;
    *)
        echo "❌ Неверный выбор"
        exit 1
        ;;
esac

echo ""
echo "🏁 Очистка завершена!"

# Показать оставшиеся результаты
echo ""
echo "📂 Оставшиеся результаты:"
for dir in dataset dataset_classifier images_backup_temp images_quick images_full; do
    if [ -d "$dir" ]; then
        count=$(find "$dir" -type f | wc -l)
        size=$(du -sh "$dir" 2>/dev/null | cut -f1)
        echo "  📁 $dir: $count файлов ($size)"
    fi
done