# 🔧 Makefile для сборки проекта GRADIENT_BASED_OPTIMIZER

# Определяем наличие PyTorch
LIBTORCH_DIR = $(HOME)/libtorch
CMAKE_PREFIX = $(if $(wildcard $(LIBTORCH_DIR)),-DCMAKE_PREFIX_PATH=$(LIBTORCH_DIR),)

.PHONY: all build clean test help basic full

# Основная цель - автоматическая сборка
all: build

# Автоматическая сборка (определяет наличие PyTorch)
build:
	@echo "🚀 Автоматическая сборка..."
	@mkdir -p build
	@cd build && rm -rf * && cmake $(CMAKE_PREFIX) .. && make -j4
	@echo "✅ Сборка завершена!"
	@echo "📦 Тестирование:"
	@./build/gradient_based_optimizer --help | head -10

# Базовая сборка (без PyTorch)
basic:
	@echo "🔧 Базовая сборка (без классификатора)..."
	@mkdir -p build
	@cd build && rm -rf * && cmake .. && make -j4
	@echo "✅ Базовая сборка завершена!"

# Полная сборка (с PyTorch)
full:
	@echo "🤖 Полная сборка (с классификатором)..."
	@mkdir -p build
	@cd build && rm -rf * && cmake -DCMAKE_PREFIX_PATH=$(LIBTORCH_DIR) .. && make -j4
	@echo "✅ Полная сборка завершена!"

# Очистка
clean:
	@echo "🧹 Очистка..."
	@rm -rf build
	@echo "✅ Очистка завершена!"

# Тестирование
test: build
	@echo "🧪 Тестирование проекта..."
	@./build/gradient_based_optimizer --help
	@if [ -f "./build/classifier_example" ]; then \
		echo "🤖 Тестирование классификатора..."; \
		echo "Классификатор доступен!"; \
	else \
		echo "⚠️  Классификатор недоступен (базовая сборка)"; \
	fi

# Справка
help:
	@echo "🔧 Команды сборки:"
	@echo "  make          - Автоматическая сборка"
	@echo "  make build    - Автоматическая сборка" 
	@echo "  make basic    - Базовая сборка (без классификатора)"
	@echo "  make full     - Полная сборка (с классификатором)"
	@echo "  make clean    - Очистка"
	@echo "  make test     - Сборка + тестирование"
	@echo "  make help     - Эта справка"
	@echo
	@echo "🎯 Использование после сборки:"
	@echo "  ./build/gradient_based_optimizer --help"
	@echo "  ./build/gradient_based_optimizer --example    (если есть классификатор)"
	@echo "  ./build/classifier_example                    (если есть классификатор)"