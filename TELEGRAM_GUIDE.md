# 📱 Руководство по интеграции с Telegram

Это руководство описывает как настроить автоматическую отправку результатов экспериментов в Telegram.

## 🎯 Что было создано

### Скрипты

1. **telegram_config.sh** - Конфигурация с токеном бота и chat ID
2. **test_telegram.sh** - Проверка отправки сообщений в Telegram
3. **run_experiment_with_telegram.sh** - Запуск экспериментов с отправкой результатов

### Функциональность

- ✅ Отправка уведомлений о начале/окончании эксперимента
- ✅ Отправка статистики и результатов
- ✅ Автоматическая архивация и отправка датасетов
- ✅ Отправка логов выполнения
- ✅ Поддержка всех режимов экспериментов
- ✅ HTML форматирование сообщений

## 📋 Настройка Telegram бота

### Шаг 1: Создание бота

1. Откройте Telegram и найдите [@BotFather](https://t.me/BotFather)
2. Отправьте команду `/newbot`
3. Придумайте имя для бота (например: "My Experiment Bot")
4. Придумайте username (например: "my_experiment_bot")
5. Сохраните токен который вам выдаст BotFather

**Пример токена:** `1234567890:ABCdefGHIjklMNOpqrsTUVwxyz`

### Шаг 2: Получение Chat ID

#### Вариант A: Через @userinfobot (рекомендуется)

1. Найдите бота [@userinfobot](https://t.me/userinfobot)
2. Отправьте ему любое сообщение
3. Он ответит с вашим ID

#### Вариант B: Через @get_id_bot

1. Найдите бота [@get_id_bot](https://t.me/get_id_bot)
2. Отправьте команду `/start`
3. Получите ваш chat ID

#### Вариант C: Вручную через API

1. Отправьте сообщение вашему новому боту
2. Откройте в браузере:
   ```
   https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates
   ```
3. Найдите `"chat":{"id":123456789}`

**Пример Chat ID:** `123456789` или `-123456789` (для групп)

### Шаг 3: Настройка конфигурации

Отредактируйте файл `telegram_config.sh`:

```bash
nano telegram_config.sh
```

Впишите ваши данные:

```bash
#!/bin/bash

# Telegram Bot Token (получить у @BotFather)
TELEGRAM_BOT_TOKEN="1234567890:ABCdefGHIjklMNOpqrsTUVwxyz"

# Telegram Chat ID (получить у @userinfobot)
TELEGRAM_CHAT_ID="123456789"

export TELEGRAM_BOT_TOKEN
export TELEGRAM_CHAT_ID
```

Сохраните файл (Ctrl+O, Enter, Ctrl+X).

## 🧪 Проверка настройки

После настройки конфигурации запустите тест:

```bash
./test_telegram.sh
```

### Что проверяет тест

1. ✅ Подключение к Telegram API
2. ✅ Отправка текстового сообщения
3. ✅ Отправка файла
4. ✅ Множественная отправка
5. ✅ HTML форматирование

### Ожидаемый результат

Если всё настроено правильно, вы увидите:

```
🎉 ==================================================
   ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!
==================================================

✅ Проверено:
   • Подключение к Telegram API
   • Отправка текстовых сообщений
   • Отправка файлов
   • Множественная отправка
   • HTML форматирование

🚀 Telegram бот настроен и работает корректно!
```

И получите несколько тестовых сообщений в Telegram, включая тестовый файл.

## 🚀 Запуск экспериментов

### Быстрый тест

```bash
./run_experiment_with_telegram.sh --test
```

Время: ~1-2 минуты
Отправит: стартовое сообщение, итоги, архив с результатами

### Быстрый эксперимент

```bash
./run_experiment_with_telegram.sh --quick
```

Время: ~5-10 минут
Обработает 5 изображений

### Полный эксперимент

```bash
./run_experiment_with_telegram.sh --full
```

Время: ~30-60 минут
Обработает все изображения

### Генерация датасета

```bash
./run_experiment_with_telegram.sh --dataset
```

Сравнение scheme2 vs scheme3

### Датасет с классификатором

```bash
./run_experiment_with_telegram.sh --dataset-classifier
```

Автоматический выбор схем через нейросеть

## 📊 Что отправляется в Telegram

### 1. Стартовое сообщение

```
🚀 ЭКСПЕРИМЕНТ ЗАПУЩЕН

Режим: full
Начало: 2025-10-03 22:00:00
Хост: server-01
Изображений: 47

⏳ Обработка началась...
```

### 2. Сообщение о прогрессе (опционально)

```
⚙️ Обработка началась

Команда: ./build/gradient_based_optimizer
Ожидаемое время: 30-60 минут

💡 Дождитесь завершения...
```

### 3. Итоговое сообщение

```
✅ ЭКСПЕРИМЕНТ УСПЕШНО ЗАВЕРШЁН

Режим: dataset-classifier
Начало: 2025-10-03 22:00:00
Окончание: 2025-10-03 22:25:30
Длительность: 25м 30с

📂 Датасет с классификатором:
• Scheme2 выбрано: 512 блоков
• Scheme3 выбрано: 512 блоков
• Правильно извлечено: 980 блоков
• Точность: 95.7%

Хост: server-01
```

### 4. Архив с результатами

Файл: `experiment_results_20251003_220000.tar.gz`

Содержит:
- `dataset/` или `dataset_classifier/` - результаты
- `experiment_log_*.txt` - лог выполнения
- `temp_results/images/` - новые изображения (если есть)

### 5. Лог эксперимента (если < 5MB)

Файл: `experiment_log_20251003_220000.txt`

Полный вывод команды эксперимента

## ⚙️ Параметры скрипта

### Режимы эксперимента

| Параметр | Описание | Время |
|----------|----------|-------|
| `--test` | Быстрый тест (1 итерация) | 1-2 мин |
| `--quick` | 5 изображений | 5-10 мин |
| `--full` | Полный эксперимент | 30-60 мин |
| `--dataset` | Датасет scheme2 vs scheme3 | 15-30 мин |
| `--dataset-classifier` | С классификатором | 20-40 мин |

### Опции

| Параметр | Описание |
|----------|----------|
| `--no-progress` | Не отправлять промежуточные сообщения |
| `--no-compress` | Не сжимать результаты в архив |
| `--archive-name NAME` | Задать имя архива |
| `--help` | Показать справку |

### Примеры

```bash
# Базовый запуск
./run_experiment_with_telegram.sh --full

# Без промежуточных сообщений
./run_experiment_with_telegram.sh --dataset --no-progress

# Без архивации (только статистика)
./run_experiment_with_telegram.sh --test --no-compress

# Свое имя архива
./run_experiment_with_telegram.sh --quick --archive-name my_results.tar.gz
```

## 🔄 Запуск в фоновом режиме

Для длительных экспериментов можно запустить в фоне:

```bash
# С nohup
nohup ./run_experiment_with_telegram.sh --full > /dev/null 2>&1 &

# Со screen
screen -S experiment
./run_experiment_with_telegram.sh --full
# Нажмите Ctrl+A, затем D для отсоединения

# С tmux
tmux new -s experiment
./run_experiment_with_telegram.sh --full
# Нажмите Ctrl+B, затем D для отсоединения
```

Результаты всё равно придут в Telegram!

## 🔒 Безопасность

### Защита токена

**ВАЖНО:** Не публикуйте `telegram_config.sh` в git!

Добавьте в `.gitignore`:

```bash
echo "telegram_config.sh" >> .gitignore
```

### Ограничение доступа к файлу

```bash
chmod 600 telegram_config.sh
```

Теперь только владелец может читать файл.

### Использование переменных окружения

Альтернативный способ без файла конфигурации:

```bash
export TELEGRAM_BOT_TOKEN="your_token"
export TELEGRAM_CHAT_ID="your_chat_id"
./run_experiment_with_telegram.sh --test
```

## 🐛 Устранение проблем

### Ошибка: "telegram_config.sh не найден"

**Решение:** Создайте файл конфигурации по инструкции выше.

### Ошибка: "curl не установлен"

**Решение:**
```bash
# Ubuntu/Debian
sudo apt install curl

# macOS
brew install curl
```

### Тест не проходит, ошибка API

**Возможные причины:**

1. **Неверный токен** - проверьте токен у @BotFather
2. **Неверный Chat ID** - проверьте ID у @userinfobot
3. **Бот заблокирован** - разблокируйте бота в Telegram
4. **Нет интернета** - проверьте подключение

**Проверка токена:**
```bash
curl "https://api.telegram.org/bot<YOUR_TOKEN>/getMe"
```

Должен вернуть информацию о боте.

### Файлы не отправляются

**Причины:**

1. **Файл слишком большой** - Telegram лимит 50MB
   - Используйте `--no-compress` и отправьте вручную
   - Разбейте датасет на части

2. **Файл не существует** - проверьте что эксперимент завершился

3. **Права доступа** - проверьте что файл читаемый

### Сообщения не приходят в группу

Если отправляете в группу:

1. Добавьте бота в группу
2. Сделайте его администратором
3. Chat ID группы будет отрицательным (например: `-123456789`)

## 📊 Мониторинг экспериментов

### Получение статуса удаленного эксперимента

Все уведомления приходят автоматически. Вы можете:

1. Видеть когда эксперимент начался
2. Получать статистику в реальном времени
3. Видеть когда эксперимент завершился
4. Сразу скачать результаты

### Логирование

Все логи сохраняются в файлы:
- `experiment_log_YYYYMMDD_HHMMSS.txt` - полный вывод
- Отправляются в Telegram если < 5MB

### Архивация результатов

Автоматически создаются архивы:
- `experiment_results_YYYYMMDD_HHMMSS.tar.gz`
- Содержат датасеты, логи, новые изображения
- Отправляются в Telegram если < 50MB

## 🎯 Типичные сценарии

### Сценарий 1: Удаленный запуск через SSH

```bash
# Подключиться к серверу
ssh user@server

# Перейти в проект
cd GRADIENT_BASED_OPTIMIZER

# Запустить в screen
screen -S exp1
./run_experiment_with_telegram.sh --full

# Отсоединиться (Ctrl+A, D)
# Отключиться от SSH
exit

# Результаты придут в Telegram!
```

### Сценарий 2: Несколько экспериментов подряд

```bash
# Создать скрипт для последовательных экспериментов
cat > run_batch.sh << 'EOF'
#!/bin/bash
./run_experiment_with_telegram.sh --dataset
sleep 60
./run_experiment_with_telegram.sh --dataset-classifier
sleep 60
./run_experiment_with_telegram.sh --full
EOF

chmod +x run_batch.sh
nohup ./run_batch.sh &
```

### Сценарий 3: Запуск по расписанию (cron)

```bash
# Открыть crontab
crontab -e

# Добавить задачу (запуск каждую ночь в 2:00)
0 2 * * * cd /path/to/GRADIENT_BASED_OPTIMIZER && ./run_experiment_with_telegram.sh --full
```

## 📚 Дополнительные возможности

### Отправка в несколько чатов

Отредактируйте функцию `send_telegram_message` в скрипте:

```bash
send_telegram_message() {
    local message="$1"
    local api_url="https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage"

    # Чат 1
    curl -s -X POST "$api_url" -d "chat_id=${TELEGRAM_CHAT_ID}" ...

    # Чат 2
    curl -s -X POST "$api_url" -d "chat_id=${TELEGRAM_CHAT_ID_2}" ...
}
```

### Отправка графиков и изображений

Если генерируются графики:

```bash
send_telegram_photo() {
    local file_path="$1"
    local caption="$2"
    local api_url="https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendPhoto"

    curl -s -X POST "$api_url" \
        -F "chat_id=${TELEGRAM_CHAT_ID}" \
        -F "photo=@${file_path}" \
        -F "caption=${caption}"
}
```

### Уведомления об ошибках

Скрипт автоматически отправляет сообщение если эксперимент завершился с ошибкой:

```
❌ ЭКСПЕРИМЕНТ ЗАВЕРШЁН С ОШИБКОЙ
...
```

## ✅ Чеклист настройки

- [ ] Создан Telegram бот через @BotFather
- [ ] Получен токен бота
- [ ] Получен Chat ID (через @userinfobot)
- [ ] Отредактирован `telegram_config.sh` с токеном и ID
- [ ] Запущен `./test_telegram.sh` - все тесты прошли
- [ ] Получены тестовые сообщения в Telegram
- [ ] Запущен тестовый эксперимент `./run_experiment_with_telegram.sh --test`
- [ ] Получены уведомления и результаты в Telegram
- [ ] Добавлен `telegram_config.sh` в `.gitignore`

## 🏁 Готово!

Теперь вы можете запускать эксперименты и получать результаты прямо в Telegram!

```bash
./run_experiment_with_telegram.sh --full
```

**Удачных экспериментов! 🚀**
