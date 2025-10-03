#!/bin/bash

# 🧪 Скрипт для проверки отправки сообщений в Telegram
# Использование: ./test_telegram.sh

set -e

echo "🧪 =================================================="
echo "   ПРОВЕРКА ОТПРАВКИ В TELEGRAM"
echo "=================================================="
echo

# Загрузка конфигурации
if [ ! -f "telegram_config.sh" ]; then
    echo "❌ Файл telegram_config.sh не найден!"
    echo "   Создайте его из шаблона и настройте"
    exit 1
fi

source telegram_config.sh || exit 1

# Проверка что curl установлен
if ! command -v curl &> /dev/null; then
    echo "❌ curl не установлен"
    echo "   Установите: sudo apt install curl"
    exit 1
fi

echo "✅ Конфигурация загружена"
echo "   Bot Token: ${TELEGRAM_BOT_TOKEN:0:10}...${TELEGRAM_BOT_TOKEN: -5}"
echo "   Chat ID: $TELEGRAM_CHAT_ID"
echo

# ========================================
# Функция для отправки текстового сообщения
# ========================================

send_telegram_message() {
    local message="$1"
    local api_url="https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage"

    response=$(curl -s -X POST "$api_url" \
        -d "chat_id=${TELEGRAM_CHAT_ID}" \
        -d "text=${message}" \
        -d "parse_mode=HTML")

    if echo "$response" | grep -q '"ok":true'; then
        return 0
    else
        echo "$response"
        return 1
    fi
}

# ========================================
# Функция для отправки файла
# ========================================

send_telegram_file() {
    local file_path="$1"
    local caption="$2"
    local api_url="https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendDocument"

    if [ ! -f "$file_path" ]; then
        echo "❌ Файл не найден: $file_path"
        return 1
    fi

    response=$(curl -s -X POST "$api_url" \
        -F "chat_id=${TELEGRAM_CHAT_ID}" \
        -F "document=@${file_path}" \
        -F "caption=${caption}")

    if echo "$response" | grep -q '"ok":true'; then
        return 0
    else
        echo "$response"
        return 1
    fi
}

# ========================================
# ТЕСТ 1: Проверка подключения к API
# ========================================

echo "📡 ТЕСТ 1: Проверка подключения к Telegram API"
echo "----------------------------------------"

api_check_url="https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/getMe"
response=$(curl -s "$api_check_url")

if echo "$response" | grep -q '"ok":true'; then
    bot_username=$(echo "$response" | grep -o '"username":"[^"]*"' | cut -d'"' -f4)
    bot_name=$(echo "$response" | grep -o '"first_name":"[^"]*"' | cut -d'"' -f4)
    echo "✅ API доступен"
    echo "   Бот: @$bot_username ($bot_name)"
else
    echo "❌ Ошибка подключения к API"
    echo "$response"
    exit 1
fi

echo

# ========================================
# ТЕСТ 2: Отправка текстового сообщения
# ========================================

echo "📝 ТЕСТ 2: Отправка текстового сообщения"
echo "----------------------------------------"

test_message="🧪 <b>Тестовое сообщение</b>

Это автоматическое сообщение для проверки работы Telegram бота.

<i>Время:</i> $(date '+%Y-%m-%d %H:%M:%S')
<i>Хост:</i> $(hostname)

✅ Если вы видите это сообщение, отправка работает!"

if send_telegram_message "$test_message"; then
    echo "✅ Текстовое сообщение отправлено успешно"
else
    echo "❌ Ошибка отправки текстового сообщения"
    exit 1
fi

echo

# ========================================
# ТЕСТ 3: Создание и отправка тестового файла
# ========================================

echo "📄 ТЕСТ 3: Отправка файла"
echo "----------------------------------------"

# Создать тестовый файл
test_file="/tmp/telegram_test_$(date +%s).txt"
cat > "$test_file" << EOF
Тестовый файл для проверки отправки в Telegram

Дата создания: $(date)
Хост: $(hostname)
Пользователь: $(whoami)

Этот файл был создан автоматически для проверки
работы скрипта отправки файлов в Telegram.

Если вы получили этот файл, значит всё работает корректно!
EOF

echo "   Создан тестовый файл: $test_file"
echo "   Размер: $(du -h "$test_file" | cut -f1)"

if send_telegram_file "$test_file" "📄 Тестовый файл для проверки отправки"; then
    echo "✅ Файл отправлен успешно"
else
    echo "❌ Ошибка отправки файла"
    rm -f "$test_file"
    exit 1
fi

# Удалить тестовый файл
rm -f "$test_file"
echo "   Тестовый файл удалён"

echo

# ========================================
# ТЕСТ 4: Отправка нескольких сообщений
# ========================================

echo "📨 ТЕСТ 4: Отправка нескольких сообщений"
echo "----------------------------------------"

for i in {1..3}; do
    if send_telegram_message "🔢 Сообщение $i из 3"; then
        echo "   ✅ Сообщение $i отправлено"
    else
        echo "   ❌ Ошибка отправки сообщения $i"
        exit 1
    fi
    sleep 1  # Пауза между сообщениями
done

echo

# ========================================
# ТЕСТ 5: Отправка сообщения с форматированием
# ========================================

echo "🎨 ТЕСТ 5: Отправка форматированного сообщения"
echo "----------------------------------------"

formatted_message="📊 <b>Тестовая статистика</b>

<b>Система:</b>
• ОС: $(uname -s) $(uname -r)
• CPU: $(nproc) ядер
• Время работы: $(uptime -p 2>/dev/null || echo 'N/A')

<b>Проект:</b>
• Директория: $(pwd)
• Размер: $(du -sh . 2>/dev/null | cut -f1)

<code>Тест форматирования завершён</code>

✅ <i>Все функции работают</i>"

if send_telegram_message "$formatted_message"; then
    echo "✅ Форматированное сообщение отправлено"
else
    echo "❌ Ошибка отправки форматированного сообщения"
    exit 1
fi

echo

# ========================================
# ИТОГОВЫЙ ОТЧЁТ
# ========================================

echo "🎉 =================================================="
echo "   ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!"
echo "=================================================="
echo
echo "✅ Проверено:"
echo "   • Подключение к Telegram API"
echo "   • Отправка текстовых сообщений"
echo "   • Отправка файлов"
echo "   • Множественная отправка"
echo "   • HTML форматирование"
echo
echo "🚀 Telegram бот настроен и работает корректно!"
echo
echo "📝 Следующие шаги:"
echo "   1. Используйте run_experiment_with_telegram.sh для запуска экспериментов"
echo "   2. После завершения результаты будут отправлены в Telegram"
echo
echo "💡 Полезные команды:"
echo "   ./run_experiment_with_telegram.sh --test    # Быстрый тест"
echo "   ./run_experiment_with_telegram.sh --full    # Полный эксперимент"
echo

# Финальное сообщение
send_telegram_message "✅ <b>Проверка завершена</b>

Все тесты отправки в Telegram пройдены успешно!

Бот готов к работе. 🚀"

echo "✅ Финальное сообщение отправлено"
echo
