#!/bin/bash

# 🔐 Конфигурация для Telegram Bot API
# 
# ИНСТРУКЦИЯ ПО НАСТРОЙКЕ:
# 
# 1. Создайте Telegram бота:
#    - Найдите @BotFather в Telegram
#    - Отправьте команду /newbot
#    - Следуйте инструкциям
#    - Сохраните токен который выдаст BotFather
# 
# 2. Узнайте ваш Chat ID:
#    - Найдите @userinfobot в Telegram
#    - Отправьте ему любое сообщение
#    - Он пришлёт ваш ID
# 
# 3. Заполните данные ниже и сохраните как telegram_config.sh:
#    cp telegram_config.example.sh telegram_config.sh
#    nano telegram_config.sh
# 
# 4. Проверьте настройку:
#    ./test_telegram.sh

# Telegram Bot Token (получить у @BotFather)
# Формат: 1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_BOT_TOKEN="ваш_токен_здесь"

# Telegram Chat ID (получить у @userinfobot)  
# Формат: 123456789 (для личного чата) или -123456789 (для группы)
TELEGRAM_CHAT_ID="ваш_chat_id_здесь"

# Экспорт переменных
export TELEGRAM_BOT_TOKEN
export TELEGRAM_CHAT_ID

# Проверка что переменные заданы
if [ "$TELEGRAM_BOT_TOKEN" = "ваш_токен_здесь" ] || [ "$TELEGRAM_CHAT_ID" = "ваш_chat_id_здесь" ]; then
    echo "❌ ОШИБКА: Telegram конфигурация не настроена!"
    echo
    echo "Отредактируйте этот файл и укажите:"
    echo "  TELEGRAM_BOT_TOKEN - токен вашего бота от @BotFather"
    echo "  TELEGRAM_CHAT_ID - ваш chat ID от @userinfobot"
    echo
    echo "Подробная инструкция: TELEGRAM_QUICKSTART.md"
    return 1
fi
