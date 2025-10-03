# 🚀 Руководство по развертыванию проекта

## ✅ Что было создано

Для переноса проекта на новый компьютер созданы следующие файлы:

### Основные скрипты
- **setup_environment.sh** - Автоматическая установка всех зависимостей
- **deploy_and_run.sh** - Сборка и запуск экспериментов
- **check_deployment_ready.sh** - Проверка готовности к переносу

### Документация
- **DEPLOYMENT_GUIDE.md** - Полное руководство по развертыванию (439 строк)
- **QUICK_START.md** - Быстрый старт (3 простых шага)
- **SCRIPTS_README.md** - Детальная документация скриптов
- **TRANSFER_CHECKLIST.md** - Чеклист для контроля процесса
- **DEPLOYMENT_FILES.md** - Список всех файлов развертывания

## 🎯 Быстрый старт

### На исходном компьютере

```bash
# 1. Проверить готовность
./check_deployment_ready.sh

# 2. Создать архив
tar -czf gradient_optimizer.tar.gz \
  --exclude='build' \
  --exclude='dataset*' \
  --exclude='*.o' \
  --exclude='*.log' \
  .

# 3. Перенести на новый компьютер (scp, rsync, или физически)
```

### На новом компьютере

```bash
# 1. Распаковать
tar -xzf gradient_optimizer.tar.gz
cd GRADIENT_BASED_OPTIMIZER

# 2. Установить зависимости
./setup_environment.sh --with-pytorch

# 3. Собрать и протестировать
./deploy_and_run.sh --with-pytorch --test

# 4. Запустить полный эксперимент
./deploy_and_run.sh --skip-build --full
```

## 📚 Какую документацию читать?

### Для быстрого старта
→ **QUICK_START.md** - 3 простых команды

### Для детального развертывания
→ **DEPLOYMENT_GUIDE.md** - Полное руководство с примерами

### Для изучения скриптов
→ **SCRIPTS_README.md** - Документация каждого скрипта

### Для контроля процесса
→ **TRANSFER_CHECKLIST.md** - Чеклист с галочками

### Для справки о файлах
→ **DEPLOYMENT_FILES.md** - Описание всех созданных файлов

## 🔍 Проверка перед переносом

```bash
./check_deployment_ready.sh
```

Этот скрипт проверит:
- ✅ Наличие всех необходимых файлов
- ✅ Исходный код и конфигурацию
- ✅ Изображения и watermark.png
- ✅ Модели классификатора (если есть)
- ✅ Оценит размер проекта

## 📦 Создание архива

### Рекомендуемый вариант (без временных файлов)
```bash
tar -czf gradient_optimizer.tar.gz \
  --exclude='build' \
  --exclude='dataset*' \
  --exclude='*.o' \
  --exclude='*.log' \
  .
```

### Минимальный архив (только необходимое)
```bash
tar -czf gradient_optimizer_minimal.tar.gz \
  GRADIENT_BASED_OPTIMIZER/ \
  images/ \
  embedding_schemes.json \
  CMakeLists.txt \
  setup_environment.sh \
  deploy_and_run.sh \
  *.pt \
  *.md
```

## 🎮 Режимы работы

### Базовая установка (без классификатора)
```bash
./setup_environment.sh
./deploy_and_run.sh --test
```
Время: ~5-10 минут
Функциональность: базовое встраивание водяных знаков

### Полная установка (с классификатором)
```bash
./setup_environment.sh --with-pytorch
./deploy_and_run.sh --with-pytorch --test
```
Время: ~15-20 минут (загрузка PyTorch)
Функциональность: автоматический выбор схем

## 🐛 Типичные проблемы

### Скрипт не запускается
```bash
chmod +x *.sh
```

### CMake не найден
```bash
# Ubuntu
sudo apt install cmake

# macOS
brew install cmake
```

### OpenCV не найден
```bash
# Ubuntu
sudo apt install libopencv-dev

# macOS
brew install opencv
```

### Watermark.png отсутствует
Это критичный файл! Скопируйте его из исходного проекта.

## 📊 Ожидаемые размеры

- Исходный код: ~5MB
- С изображениями: ~50MB
- С моделями PyTorch: ~90MB
- С результатами: ~500MB-1GB (не переносить)

## ✨ Особенности скриптов

### setup_environment.sh
- ✅ Автоматическое определение ОС
- ✅ Проверка зависимостей
- ✅ Установка PyTorch (опционально)
- ✅ Настройка переменных окружения
- ✅ Поддержка CUDA версии

### deploy_and_run.sh
- ✅ Проверка всех зависимостей
- ✅ Автоматическая сборка
- ✅ 5+ режимов экспериментов
- ✅ Детальный вывод прогресса
- ✅ Анализ результатов

### check_deployment_ready.sh
- ✅ Проверка 50+ параметров
- ✅ Оценка размера проекта
- ✅ Детальный отчет
- ✅ Рекомендации по оптимизации

## 🎯 Следующие шаги

После успешного развертывания:

1. **Запустить тестовый эксперимент**
   ```bash
   ./deploy_and_run.sh --skip-build --test
   ```

2. **Проверить результаты**
   ```bash
   ls -la images/new_*.png
   ```

3. **Запустить полный эксперимент**
   ```bash
   ./deploy_and_run.sh --skip-build --full
   ```

4. **Анализировать результаты**
   ```bash
   ./analyze_existing_results.sh
   ```

## 📞 Дополнительная помощь

Если что-то не работает:

1. Проверьте **DEPLOYMENT_GUIDE.md** раздел "Устранение проблем"
2. Используйте **TRANSFER_CHECKLIST.md** для пошаговой проверки
3. Изучите логи: `build/cmake_output.log`, `build/make_output.log`
4. Запустите `./check_deployment_ready.sh` для диагностики

## 🏁 Готово!

Все файлы созданы и протестированы. Проект готов к переносу на новый компьютер.

**Успешного развертывания! 🚀**
