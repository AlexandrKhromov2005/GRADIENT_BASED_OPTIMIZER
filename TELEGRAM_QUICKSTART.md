# ⚡ Telegram - Быстрый старт

## 3 шага до автоматических уведомлений

### Шаг 1: Создать бота

1. Найдите [@BotFather](https://t.me/BotFather) в Telegram
2. Отправьте `/newbot`
3. Придумайте имя и username
4. Сохраните **токен** (например: `1234567890:ABCdef...`)

### Шаг 2: Узнать Chat ID

1. Найдите [@userinfobot](https://t.me/userinfobot) в Telegram
2. Отправьте любое сообщение
3. Сохраните ваш **Chat ID** (например: `123456789`)

### Шаг 3: Настроить и запустить

```bash
# Отредактировать конфигурацию
nano telegram_config.sh
# Вписать TELEGRAM_BOT_TOKEN и TELEGRAM_CHAT_ID

# Проверить что работает
./test_telegram.sh

# Запустить генерацию квадрантного датасета
./run_experiment_with_telegram.sh
```

## ✅ Готово!

Теперь результаты квадрантного датасета будут приходить в Telegram автоматически!

**Полная документация:** [TELEGRAM_GUIDE.md](TELEGRAM_GUIDE.md)

---

## Быстрые команды

```bash
# Быстрый тест (1-2 мин)
./run_experiment_with_telegram.sh --test

# Генерация квадрантного датасета (основной режим, по умолчанию)
./run_experiment_with_telegram.sh

# Без промежуточных сообщений
./run_experiment_with_telegram.sh --no-progress
```

## Запуск в фоне

```bash
# Запустить и отключиться от SSH - результаты придут в Telegram
screen -S quadrant_dataset
./run_experiment_with_telegram.sh
# Нажать Ctrl+A, затем D
```

## Что получите в Telegram

1. 🚀 Уведомление о старте
2. 📊 Статистику: количество квадрантов по каждому типу (Q1, Q2, Q3, Q4)
3. 📦 Архив с квадрантным датасетом (`dataset_quadrant/`)
4. 📄 Лог выполнения

## Помощь

Проблемы? Читайте [TELEGRAM_GUIDE.md](TELEGRAM_GUIDE.md) раздел "Устранение проблем"
