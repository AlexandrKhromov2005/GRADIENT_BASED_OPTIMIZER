#!/bin/bash

# 🚀 Скрипт запуска экспериментов с отправкой результатов в Telegram
# Использование: ./run_experiment_with_telegram.sh [OPTIONS]

set -e

echo "🚀 =================================================="
echo "   ЭКСПЕРИМЕНТ С ОТПРАВКОЙ В TELEGRAM"
echo "=================================================="
echo

# ========================================
# ПАРАМЕТРЫ
# ========================================

EXPERIMENT_MODE="full"
SEND_PROGRESS=true
COMPRESS_RESULTS=true
ARCHIVE_NAME="experiment_results_$(date +%Y%m%d_%H%M%S).tar.gz"

# Парсинг аргументов
while [[ $# -gt 0 ]]; do
    case $1 in
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
        --dataset)
            EXPERIMENT_MODE="dataset"
            shift
            ;;
        --dataset-classifier)
            EXPERIMENT_MODE="dataset-classifier"
            shift
            ;;
        --no-progress)
            SEND_PROGRESS=false
            shift
            ;;
        --no-compress)
            COMPRESS_RESULTS=false
            shift
            ;;
        --archive-name)
            ARCHIVE_NAME="$2"
            shift 2
            ;;
        --help)
            echo "Использование: $0 [OPTIONS]"
            echo
            echo "Режимы эксперимента:"
            echo "  --test                Быстрый тест (1 итерация)"
            echo "  --quick               Быстрый эксперимент (5 изображений)"
            echo "  --full                Полный эксперимент (по умолчанию)"
            echo "  --dataset             Генерация датасета scheme2 vs scheme3"
            echo "  --dataset-classifier  Датасет с классификатором"
            echo
            echo "Опции:"
            echo "  --no-progress         Не отправлять промежуточные сообщения"
            echo "  --no-compress         Не сжимать результаты в архив"
            echo "  --archive-name NAME   Имя архива (по умолчанию: experiment_results_YYYYMMDD_HHMMSS.tar.gz)"
            echo "  --help                Показать эту справку"
            echo
            echo "Примеры:"
            echo "  $0 --test                    # Быстрый тест с отправкой в Telegram"
            echo "  $0 --full                    # Полный эксперимент"
            echo "  $0 --dataset-classifier      # Датасет с классификатором"
            echo "  $0 --quick --no-progress     # Без промежуточных сообщений"
            exit 0
            ;;
        *)
            echo "❌ Неизвестный параметр: $1"
            echo "Используйте --help для справки"
            exit 1
            ;;
    esac
done

# ========================================
# ЗАГРУЗКА КОНФИГУРАЦИИ TELEGRAM
# ========================================

echo "📡 Загрузка конфигурации Telegram..."

if [ ! -f "telegram_config.sh" ]; then
    echo "❌ Файл telegram_config.sh не найден!"
    echo
    echo "Создайте файл telegram_config.sh со следующим содержимым:"
    echo
    cat << 'EOF'
#!/bin/bash
TELEGRAM_BOT_TOKEN="your_bot_token_here"
TELEGRAM_CHAT_ID="your_chat_id_here"
export TELEGRAM_BOT_TOKEN
export TELEGRAM_CHAT_ID
EOF
    echo
    echo "Затем настройте токен и chat ID"
    exit 1
fi

source telegram_config.sh || exit 1
echo "✅ Конфигурация загружена"
echo

# ========================================
# ФУНКЦИИ ДЛЯ РАБОТЫ С TELEGRAM
# ========================================

send_telegram_message() {
    local message="$1"
    local api_url="https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage"

    curl -s -X POST "$api_url" \
        -d "chat_id=${TELEGRAM_CHAT_ID}" \
        -d "text=${message}" \
        -d "parse_mode=HTML" > /dev/null 2>&1
}

send_telegram_file() {
    local file_path="$1"
    local caption="$2"
    local api_url="https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendDocument"

    if [ ! -f "$file_path" ]; then
        echo "⚠️  Файл не найден: $file_path"
        return 1
    fi

    local file_size=$(du -h "$file_path" | cut -f1)
    echo "   Отправка: $(basename "$file_path") ($file_size)"

    response=$(curl -s -X POST "$api_url" \
        -F "chat_id=${TELEGRAM_CHAT_ID}" \
        -F "document=@${file_path}" \
        -F "caption=${caption}")

    if echo "$response" | grep -q '"ok":true'; then
        echo "   ✅ Отправлено"
        return 0
    else
        echo "   ❌ Ошибка отправки"
        echo "$response" | head -3
        return 1
    fi
}

# ========================================
# ПРОВЕРКА ЗАВИСИМОСТЕЙ
# ========================================

echo "🔍 Проверка зависимостей..."

# Проверка curl
if ! command -v curl &> /dev/null; then
    echo "❌ curl не установлен (нужен для Telegram)"
    exit 1
fi

# Проверка исполняемого файла
if [ ! -f "build/gradient_based_optimizer" ]; then
    echo "❌ Проект не собран!"
    echo "   Запустите: ./deploy_and_run.sh --with-pytorch"
    exit 1
fi

# Проверка изображений
image_count=$(find images -name "*.png" -o -name "*.jpg" 2>/dev/null | wc -l)
if [ $image_count -eq 0 ]; then
    echo "❌ Изображения не найдены в images/"
    exit 1
fi

echo "✅ Зависимости в порядке"
echo "   Изображений: $image_count"
echo

# ========================================
# ОТПРАВКА СТАРТОВОГО СООБЩЕНИЯ
# ========================================

START_TIME=$(date +%s)
START_DATE=$(date '+%Y-%m-%d %H:%M:%S')

start_message="🚀 <b>ЭКСПЕРИМЕНТ ЗАПУЩЕН</b>

<b>Режим:</b> $EXPERIMENT_MODE
<b>Начало:</b> $START_DATE
<b>Хост:</b> $(hostname)
<b>Изображений:</b> $image_count

⏳ Обработка началась..."

echo "📨 Отправка стартового сообщения..."
send_telegram_message "$start_message"
echo "✅ Стартовое сообщение отправлено"
echo

# ========================================
# ЗАПУСК ЭКСПЕРИМЕНТА
# ========================================

echo "🔬 Запуск эксперимента (режим: $EXPERIMENT_MODE)..."
echo "⏱️  Начало: $START_DATE"
echo

# Определить команду запуска
case $EXPERIMENT_MODE in
    test)
        COMMAND="./build/gradient_based_optimizer --test"
        EXPECTED_TIME="1-2 минуты"
        ;;
    quick)
        COMMAND="./run_quick_experiment.sh"
        EXPECTED_TIME="5-10 минут"
        ;;
    dataset)
        COMMAND="./build/gradient_based_optimizer --dataset"
        EXPECTED_TIME="15-30 минут"
        ;;
    dataset-classifier)
        COMMAND="./build/gradient_based_optimizer --dataset-classifier"
        EXPECTED_TIME="20-40 минут"
        ;;
    full)
        COMMAND="./build/gradient_based_optimizer"
        EXPECTED_TIME="30-60 минут"
        ;;
    *)
        echo "❌ Неизвестный режим: $EXPERIMENT_MODE"
        exit 1
        ;;
esac

echo "📋 Команда: $COMMAND"
echo "⏱️  Ожидаемое время: $EXPECTED_TIME"
echo

# Отправить сообщение о начале обработки
if [ "$SEND_PROGRESS" = true ]; then
    send_telegram_message "⚙️ <b>Обработка началась</b>

Команда: <code>$COMMAND</code>
Ожидаемое время: $EXPECTED_TIME

💡 Дождитесь завершения..."
fi

# Создать лог файл
LOG_FILE="experiment_log_$(date +%Y%m%d_%H%M%S).txt"

# Запустить эксперимент
echo "🔄 Выполнение эксперимента..."
echo "   (лог сохраняется в $LOG_FILE)"
echo

if $COMMAND 2>&1 | tee "$LOG_FILE"; then
    EXPERIMENT_STATUS="success"
    echo
    echo "✅ Эксперимент завершён успешно"
else
    EXPERIMENT_STATUS="failed"
    echo
    echo "❌ Эксперимент завершился с ошибкой"
fi

END_TIME=$(date +%s)
END_DATE=$(date '+%Y-%m-%d %H:%M:%S')
DURATION=$((END_TIME - START_TIME))
DURATION_MIN=$((DURATION / 60))
DURATION_SEC=$((DURATION % 60))

echo "⏱️  Окончание: $END_DATE"
echo "⏱️  Длительность: ${DURATION_MIN}м ${DURATION_SEC}с"
echo

# ========================================
# АНАЛИЗ РЕЗУЛЬТАТОВ
# ========================================

echo "📊 Анализ результатов..."

# Подсчёт результатов
results_summary=""

if [ -d "dataset" ]; then
    scheme2_correct=$(find dataset/scheme2_correct -name "*.png" 2>/dev/null | wc -l)
    scheme2_incorrect=$(find dataset/scheme2_incorrect -name "*.png" 2>/dev/null | wc -l)
    scheme3_correct=$(find dataset/scheme3_correct -name "*.png" 2>/dev/null | wc -l)
    scheme3_incorrect=$(find dataset/scheme3_incorrect -name "*.png" 2>/dev/null | wc -l)

    total_scheme2=$((scheme2_correct + scheme2_incorrect))
    total_scheme3=$((scheme3_correct + scheme3_incorrect))
    total=$((total_scheme2 + total_scheme3))

    if [ $total -gt 0 ]; then
        accuracy=$(echo "scale=1; ($scheme2_correct + $scheme3_correct) * 100 / $total" | bc -l 2>/dev/null || echo "N/A")

        results_summary="📂 <b>Датасет (dataset/):</b>
• Scheme2: $total_scheme2 блоков ($scheme2_correct правильных)
• Scheme3: $total_scheme3 блоков ($scheme3_correct правильных)
• Общая точность: $accuracy%"
    fi
fi

if [ -d "dataset_classifier" ]; then
    scheme2_selected=$(find dataset_classifier/scheme2_selected -name "*.png" 2>/dev/null | wc -l)
    scheme3_selected=$(find dataset_classifier/scheme3_selected -name "*.png" 2>/dev/null | wc -l)
    extraction_correct=$(find dataset_classifier/extraction_correct -name "*.png" 2>/dev/null | wc -l)
    extraction_incorrect=$(find dataset_classifier/extraction_incorrect -name "*.png" 2>/dev/null | wc -l)

    total_classifier=$((extraction_correct + extraction_incorrect))

    if [ $total_classifier -gt 0 ]; then
        accuracy_classifier=$(echo "scale=1; $extraction_correct * 100 / $total_classifier" | bc -l 2>/dev/null || echo "N/A")

        results_summary="${results_summary}

📂 <b>Датасет с классификатором:</b>
• Scheme2 выбрано: $scheme2_selected блоков
• Scheme3 выбрано: $scheme3_selected блоков
• Правильно извлечено: $extraction_correct блоков
• Точность: $accuracy_classifier%"
    fi
fi

if [ -z "$results_summary" ]; then
    results_summary="ℹ️ Результаты не найдены или не применимо к режиму $EXPERIMENT_MODE"
fi

echo "$results_summary"
echo

# ========================================
# СОЗДАНИЕ АРХИВА С РЕЗУЛЬТАТАМИ
# ========================================

if [ "$COMPRESS_RESULTS" = true ]; then
    echo "📦 Создание архива с результатами..."

    # Определить что архивировать
    ARCHIVE_CONTENT=""

    [ -d "dataset" ] && ARCHIVE_CONTENT="$ARCHIVE_CONTENT dataset/"
    [ -d "dataset_classifier" ] && ARCHIVE_CONTENT="$ARCHIVE_CONTENT dataset_classifier/"
    [ -f "$LOG_FILE" ] && ARCHIVE_CONTENT="$ARCHIVE_CONTENT $LOG_FILE"

    # Добавить новые изображения если есть
    new_images=$(find images -name "new_*.png" 2>/dev/null)
    if [ -n "$new_images" ]; then
        mkdir -p temp_results/images
        find images -name "new_*.png" -exec cp {} temp_results/images/ \;
        ARCHIVE_CONTENT="$ARCHIVE_CONTENT temp_results/"
    fi

    if [ -n "$ARCHIVE_CONTENT" ]; then
        echo "   Архивирование: $ARCHIVE_CONTENT"

        if tar -czf "$ARCHIVE_NAME" $ARCHIVE_CONTENT 2>&1; then
            ARCHIVE_SIZE=$(du -h "$ARCHIVE_NAME" | cut -f1)
            echo "   ✅ Архив создан: $ARCHIVE_NAME ($ARCHIVE_SIZE)"

            # Очистка временных файлов
            [ -d "temp_results" ] && rm -rf temp_results
        else
            echo "   ❌ Ошибка создания архива"
            ARCHIVE_NAME=""
        fi
    else
        echo "   ⚠️  Нечего архивировать"
        ARCHIVE_NAME=""
    fi
else
    echo "⏭️  Создание архива пропущено (--no-compress)"
    ARCHIVE_NAME=""
fi

echo

# ========================================
# ОТПРАВКА РЕЗУЛЬТАТОВ В TELEGRAM
# ========================================

echo "📤 Отправка результатов в Telegram..."
echo

# Итоговое сообщение
if [ "$EXPERIMENT_STATUS" = "success" ]; then
    status_emoji="✅"
    status_text="УСПЕШНО ЗАВЕРШЁН"
else
    status_emoji="❌"
    status_text="ЗАВЕРШЁН С ОШИБКОЙ"
fi

final_message="$status_emoji <b>ЭКСПЕРИМЕНТ $status_text</b>

<b>Режим:</b> $EXPERIMENT_MODE
<b>Начало:</b> $START_DATE
<b>Окончание:</b> $END_DATE
<b>Длительность:</b> ${DURATION_MIN}м ${DURATION_SEC}с

$results_summary

<b>Хост:</b> $(hostname)"

echo "📨 Отправка итогового сообщения..."
send_telegram_message "$final_message"
echo "✅ Итоговое сообщение отправлено"
echo

# Отправка архива
if [ -n "$ARCHIVE_NAME" ] && [ -f "$ARCHIVE_NAME" ]; then
    echo "📦 Отправка архива с результатами..."

    archive_caption="📦 Результаты эксперимента

Режим: $EXPERIMENT_MODE
Время: ${DURATION_MIN}м ${DURATION_SEC}с
Размер: $(du -h "$ARCHIVE_NAME" | cut -f1)"

    if send_telegram_file "$ARCHIVE_NAME" "$archive_caption"; then
        echo "✅ Архив отправлен"
    else
        echo "⚠️  Не удалось отправить архив"
        echo "   Возможно файл слишком большой (лимит Telegram: 50MB)"

        # Попробовать разбить на части если файл большой
        archive_size_mb=$(du -m "$ARCHIVE_NAME" | cut -f1)
        if [ $archive_size_mb -gt 50 ]; then
            send_telegram_message "⚠️ <b>Архив слишком большой</b>

Размер: ${archive_size_mb}MB (лимит 50MB)

Результаты сохранены локально:
<code>$ARCHIVE_NAME</code>

Скачайте файл вручную с сервера."
        fi
    fi
else
    echo "⏭️  Архив не создан, отправка пропущена"
fi

# Отправка лога
if [ -f "$LOG_FILE" ]; then
    log_size=$(du -h "$LOG_FILE" | cut -f1)

    # Отправить лог только если он не слишком большой
    log_size_kb=$(du -k "$LOG_FILE" | cut -f1)
    if [ $log_size_kb -lt 5000 ]; then  # Меньше 5MB
        echo
        echo "📄 Отправка лога..."
        if send_telegram_file "$LOG_FILE" "📄 Лог эксперимента ($log_size)"; then
            echo "✅ Лог отправлен"
        fi
    else
        echo "   ⏭️  Лог слишком большой ($log_size), пропускаем"
    fi
fi

echo

# ========================================
# ЗАВЕРШЕНИЕ
# ========================================

echo "🎉 =================================================="
echo "   ЭКСПЕРИМЕНТ ЗАВЕРШЁН"
echo "=================================================="
echo
echo "📊 Статистика:"
echo "   Режим: $EXPERIMENT_MODE"
echo "   Статус: $status_text"
echo "   Длительность: ${DURATION_MIN}м ${DURATION_SEC}с"
if [ -n "$ARCHIVE_NAME" ] && [ -f "$ARCHIVE_NAME" ]; then
    echo "   Архив: $ARCHIVE_NAME ($(du -h "$ARCHIVE_NAME" | cut -f1))"
fi
echo "   Лог: $LOG_FILE"
echo
echo "✅ Результаты отправлены в Telegram!"
echo

# Финальное уведомление
send_telegram_message "🏁 <b>Обработка завершена</b>

Все результаты отправлены.
Проверьте файлы выше. ✅"

# Очистка (опционально)
if [ "$EXPERIMENT_STATUS" = "success" ]; then
    echo "💡 Для очистки результатов используйте:"
    echo "   rm -rf dataset dataset_classifier $ARCHIVE_NAME $LOG_FILE"
    echo
fi

exit 0
