# 📱 Интеграция с Telegram - Обзор

## ✅ Что было создано

Для автоматической отправки результатов экспериментов в Telegram созданы следующие файлы:

### Скрипты (4 файла)

1. **telegram_config.sh** - Конфигурация с токеном и Chat ID (нужно настроить)
2. **telegram_config.example.sh** - Пример конфигурации с инструкциями
3. **test_telegram.sh** (8.3KB) - Проверка отправки сообщений
4. **run_experiment_with_telegram.sh** (17KB) - Запуск экспериментов с уведомлениями

### Документация (3 файла)

5. **TELEGRAM_QUICKSTART.md** - Быстрый старт (3 шага)
6. **TELEGRAM_GUIDE.md** - Полное руководство
7. **TELEGRAM_README.md** - Этот файл (обзор)

## 🚀 Быстрый старт

### 1. Настроить конфигурацию

```bash
# Скопировать пример
cp telegram_config.example.sh telegram_config.sh

# Отредактировать и вписать свои данные
nano telegram_config.sh
```

Вам нужны:
- **Токен бота** - получить у [@BotFather](https://t.me/BotFather)
- **Chat ID** - получить у [@userinfobot](https://t.me/userinfobot)

### 2. Проверить что работает

```bash
./test_telegram.sh
```

Должны прийти тестовые сообщения в Telegram.

### 3. Запустить эксперимент

```bash
./run_experiment_with_telegram.sh --test
```

Результаты придут в Telegram автоматически!

## 📚 Документация

### Для быстрого старта
→ **[TELEGRAM_QUICKSTART.md](TELEGRAM_QUICKSTART.md)** - 3 шага до автоматических уведомлений

### Для детальной настройки
→ **[TELEGRAM_GUIDE.md](TELEGRAM_GUIDE.md)** - Полное руководство (50+ разделов)

## 🎯 Основные возможности

### ✅ Что умеет система

- Отправка уведомлений о начале/окончании эксперимента
- Отправка детальной статистики и результатов
- Автоматическая архивация датасетов
- Отправка архивов с результатами (<50MB)
- Отправка логов выполнения (<5MB)
- Поддержка всех режимов экспериментов
- HTML форматирование сообщений
- Работа в фоновом режиме

### 📊 Режимы экспериментов

| Команда | Время | Описание |
|---------|-------|----------|
| `--test` | 1-2 мин | Быстрый тест |
| `--quick` | 5-10 мин | 5 изображений |
| `--full` | 30-60 мин | Все изображения |
| `--dataset` | 15-30 мин | Scheme2 vs Scheme3 |
| `--dataset-classifier` | 20-40 мин | С классификатором |

## 💡 Примеры использования

### Запуск локально

```bash
# Быстрый тест
./run_experiment_with_telegram.sh --test

# Полный эксперимент
./run_experiment_with_telegram.sh --full
```

### Запуск на удаленном сервере

```bash
# Подключиться к серверу
ssh user@server
cd GRADIENT_BASED_OPTIMIZER

# Запустить в screen
screen -S experiment
./run_experiment_with_telegram.sh --dataset-classifier

# Отсоединиться (Ctrl+A, затем D)
# Можно выйти из SSH - результаты придут в Telegram!
exit
```

### Запуск в фоне (nohup)

```bash
nohup ./run_experiment_with_telegram.sh --full > /dev/null 2>&1 &
```

Уведомления всё равно придут в Telegram!

## 📨 Что приходит в Telegram

### 1. Стартовое уведомление
```
🚀 ЭКСПЕРИМЕНТ ЗАПУЩЕН

Режим: dataset-classifier
Начало: 2025-10-03 22:00:00
Хост: server-01
Изображений: 47

⏳ Обработка началась...
```

### 2. Итоговое сообщение
```
✅ ЭКСПЕРИМЕНТ УСПЕШНО ЗАВЕРШЁН

Режим: dataset-classifier
Длительность: 25м 30с

📂 Датасет с классификатором:
• Scheme2 выбрано: 512 блоков
• Scheme3 выбрано: 512 блоков
• Правильно извлечено: 980 блоков
• Точность: 95.7%
```

### 3. Архив с результатами
- Файл: `experiment_results_YYYYMMDD_HHMMSS.tar.gz`
- Содержит: датасеты, логи, изображения
- Автоматически отправляется если <50MB

### 4. Лог эксперимента
- Полный вывод команды
- Отправляется если <5MB

## 🔧 Параметры скрипта

### Основные опции

```bash
# Режимы
--test                  # Быстрый тест (1 итерация)
--quick                 # 5 изображений
--full                  # Полный эксперимент
--dataset               # Датасет scheme2 vs scheme3
--dataset-classifier    # С классификатором

# Дополнительно
--no-progress           # Без промежуточных сообщений
--no-compress           # Не создавать архив
--archive-name NAME     # Свое имя архива
--help                  # Справка
```

### Примеры

```bash
# Базовый запуск
./run_experiment_with_telegram.sh --full

# Без промежуточных сообщений
./run_experiment_with_telegram.sh --dataset --no-progress

# Без архивации (только статистика)
./run_experiment_with_telegram.sh --quick --no-compress

# Свое имя архива
./run_experiment_with_telegram.sh --test --archive-name test_results.tar.gz
```

## 🧪 Тестирование

### Проверка базовой отправки

```bash
./test_telegram.sh
```

Выполняет 5 тестов:
1. ✅ Подключение к API
2. ✅ Текстовое сообщение
3. ✅ Отправка файла
4. ✅ Множественная отправка
5. ✅ HTML форматирование

### Быстрый тест эксперимента

```bash
./run_experiment_with_telegram.sh --test
```

Время: ~1-2 минуты
Проверяет весь цикл: старт → выполнение → отправка результатов

## 🔒 Безопасность

### Защита токена

**ВАЖНО:** Не публикуйте `telegram_config.sh` в git!

```bash
# Добавить в .gitignore
echo "telegram_config.sh" >> .gitignore

# Ограничить права доступа
chmod 600 telegram_config.sh
```

### Рекомендации

- ✅ Используйте отдельного бота для каждого сервера
- ✅ Периодически меняйте токен через @BotFather
- ✅ Не храните токены в публичных репозиториях
- ✅ Используйте переменные окружения для CI/CD

## 🐛 Устранение проблем

### Тест не проходит

**Проверьте:**
1. Токен правильно скопирован (без пробелов)
2. Chat ID корректный (без лишних символов)
3. Бот не заблокирован в Telegram
4. Есть интернет соединение
5. curl установлен: `which curl`

**Ручная проверка токена:**
```bash
curl "https://api.telegram.org/bot<YOUR_TOKEN>/getMe"
```

### Файлы не отправляются

**Причины:**
- Файл >50MB (лимит Telegram)
- Файл не существует (эксперимент не завершился)
- Нет прав на чтение файла

**Решение для больших файлов:**
```bash
# Запустить без архивации
./run_experiment_with_telegram.sh --full --no-compress

# Скачать результаты вручную
scp user@server:~/GRADIENT_BASED_OPTIMIZER/dataset_classifier.tar.gz .
```

### Сообщения не приходят

**Проверьте:**
1. Бот не заблокирован - напишите ему `/start`
2. Chat ID правильный - проверьте у @userinfobot
3. Для групп Chat ID отрицательный (начинается с `-`)
4. Бот добавлен в группу (если отправляете в группу)

## 📊 Структура результатов

После выполнения эксперимента создаются:

```
GRADIENT_BASED_OPTIMIZER/
├── dataset/                          # Результаты --dataset
│   ├── scheme2_correct/
│   ├── scheme2_incorrect/
│   ├── scheme3_correct/
│   └── scheme3_incorrect/
├── dataset_classifier/               # Результаты --dataset-classifier
│   ├── scheme2_selected/
│   ├── scheme3_selected/
│   ├── extraction_correct/
│   └── extraction_incorrect/
├── experiment_log_*.txt              # Логи
└── experiment_results_*.tar.gz       # Архив (отправляется в Telegram)
```

## 🔄 Интеграция с существующими скриптами

Telegram интеграция совместима со всеми существующими скриптами:

- ✅ `build.sh` - сборка проекта
- ✅ `deploy_and_run.sh` - развертывание
- ✅ `run_classifier_experiment.sh` - эксперименты
- ✅ `clean_experiment_results.sh` - очистка

Можно использовать вместе:

```bash
# Собрать проект
./deploy_and_run.sh --with-pytorch

# Запустить с Telegram уведомлениями
./run_experiment_with_telegram.sh --full
```

## 📈 Мониторинг длительных экспериментов

### Удаленный запуск

```bash
# На сервере
screen -S exp1
./run_experiment_with_telegram.sh --full
# Ctrl+A, затем D для отсоединения

# Можно закрыть SSH
# Результаты придут в Telegram!
```

### Множественные эксперименты

```bash
# Создать батч-скрипт
cat > run_batch.sh << 'SCRIPT'
#!/bin/bash
./run_experiment_with_telegram.sh --dataset
sleep 300
./run_experiment_with_telegram.sh --dataset-classifier
sleep 300
./run_experiment_with_telegram.sh --full
SCRIPT

chmod +x run_batch.sh
nohup ./run_batch.sh &
```

Каждый эксперимент отправит свои результаты отдельно.

## ✅ Чеклист настройки

- [ ] Создан Telegram бот через @BotFather
- [ ] Получен токен бота
- [ ] Получен Chat ID через @userinfobot
- [ ] Создан `telegram_config.sh` из примера
- [ ] Заполнены TELEGRAM_BOT_TOKEN и TELEGRAM_CHAT_ID
- [ ] Запущен `./test_telegram.sh` - все 5 тестов прошли
- [ ] Получены тестовые сообщения в Telegram
- [ ] Запущен `./run_experiment_with_telegram.sh --test`
- [ ] Получены уведомления о старте/завершении
- [ ] Получен архив с результатами
- [ ] Добавлен `telegram_config.sh` в .gitignore

## 📞 Поддержка

### Документация

- **Быстрый старт:** [TELEGRAM_QUICKSTART.md](TELEGRAM_QUICKSTART.md)
- **Полное руководство:** [TELEGRAM_GUIDE.md](TELEGRAM_GUIDE.md)
- **Эта страница:** TELEGRAM_README.md

### Полезные ссылки

- [@BotFather](https://t.me/BotFather) - создание ботов
- [@userinfobot](https://t.me/userinfobot) - получить Chat ID
- [Telegram Bot API](https://core.telegram.org/bots/api) - документация API

## 🏁 Готово к работе!

Теперь все эксперименты будут автоматически отправлять результаты в Telegram!

```bash
./run_experiment_with_telegram.sh --full
```

**Удачных экспериментов! 🚀**
