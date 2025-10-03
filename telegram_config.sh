#!/bin/bash

# 🔐 Конфигурация для Telegram Bot API
# Настройте эти переменные перед использованием

# Telegram Bot Token (получить у @BotFather)
TELEGRAM_BOT_TOKEN=""

# Telegram Chat ID (получить у @userinfobot или @get_id_bot)
TELEGRAM_CHAT_ID=""

# Проверка что переменные заданы
if [ -z "$TELEGRAM_BOT_TOKEN" ] || [ -z "$TELEGRAM_CHAT_ID" ]; then
    echo "⚠️  ВНИМАНИЕ: Telegram конфигурация не настроена!"
    echo
    echo "Отредактируйте файл telegram_config.sh и задайте:"
    echo "  TELEGRAM_BOT_TOKEN - токен вашего бота"
    echo "  TELEGRAM_CHAT_ID - ваш chat ID"
    echo
    echo "Как получить:"
    echo "1. Создайте бота у @BotFather и получите токен"
    echo "2. Узнайте ваш chat_id у @userinfobot"
    echo
    return 1
fi

export TELEGRAM_BOT_TOKEN
export TELEGRAM_CHAT_ID
