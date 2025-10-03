# ⚡ Быстрый старт

## На исходном компьютере

Подготовьте проект к переносу:

```bash
cd /path/to/GRADIENT_BASED_OPTIMIZER

# Создать архив проекта
tar -czf gradient_optimizer_transfer.tar.gz \
  --exclude='build' \
  --exclude='dataset*' \
  --exclude='*.o' \
  --exclude='*.log' \
  .
```

## На новом компьютере

### Шаг 1: Получить проект

```bash
# Распаковать архив
tar -xzf gradient_optimizer_transfer.tar.gz
cd GRADIENT_BASED_OPTIMIZER
```

### Шаг 2: Установить зависимости

```bash
# Базовая установка (без классификатора)
./setup_environment.sh

# ИЛИ полная установка (с классификатором)
./setup_environment.sh --with-pytorch
```

### Шаг 3: Собрать и запустить

```bash
# Быстрый тест
./deploy_and_run.sh --test

# ИЛИ с классификатором
./deploy_and_run.sh --with-pytorch --test
```

## Готово! 🎉

Если тест прошёл успешно, запустите полный эксперимент:

```bash
# Интерактивный выбор режима
./deploy_and_run.sh --with-pytorch

# Или напрямую полный эксперимент
./deploy_and_run.sh --with-pytorch --full
```

---

**Подробнее:** [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
