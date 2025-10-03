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

# Запустить эксперимент
./run_experiment_with_telegram.sh --test
```

## ✅ Готово!

Теперь результаты будут приходить в Telegram автоматически!

**Полная документация:** [TELEGRAM_GUIDE.md](TELEGRAM_GUIDE.md)

---

## Быстрые команды

```bash
# Быстрый тест (1-2 мин)
./run_experiment_with_telegram.sh --test

# Быстрый эксперимент (5-10 мин)
./run_experiment_with_telegram.sh --quick

# Полный эксперимент (30-60 мин)
./run_experiment_with_telegram.sh --full

# С классификатором
./run_experiment_with_telegram.sh --dataset-classifier
```

## Запуск в фоне

```bash
# Запустить и отключиться от SSH - результаты придут в Telegram
screen -S experiment
./run_experiment_with_telegram.sh --full
# Нажать Ctrl+A, затем D
```

## Что получите в Telegram

1. 🚀 Уведомление о старте
2. 📊 Статистику по завершению
3. 📦 Архив с результатами
4. 📄 Лог выполнения

## Помощь

Проблемы? Читайте [TELEGRAM_GUIDE.md](TELEGRAM_GUIDE.md) раздел "Устранение проблем"
