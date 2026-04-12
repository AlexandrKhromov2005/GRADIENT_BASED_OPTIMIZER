# Gradient-Based Optimizer для встраивания ЦВЗ в DCT-домене

*[English version](README.md)*

Репозиторий содержит референсную реализацию на C++ градиентного метаэвристического
оптимизатора (GBO) для встраивания цифровых водяных знаков (ЦВЗ) в DCT-домен
полутоновых изображений с настраиваемой устойчивостью к JPEG-сжатию и другим
атакам. Оптимизатор работает над 8×8 DCT-блоками, поддерживает несколько
стратегий выбора коэффициентов («схем встраивания») и опциональную
интеграцию с PyTorch-классификаторами, которые автоматически подбирают
оптимальную схему для блока или квадранта изображения.

Код — это артефакт, сопровождающий статью; он задуман как
**воспроизводимый** и **пригодный для повторного использования** в сторонних
проектах.

---

## 1. Состав репозитория

```text
.
├── CMakeLists.txt                     # сборка с настраиваемыми целями
├── build.sh                           # обёртка cmake + make
├── embedding_schemes.json             # JSON-описания наборов DCT-коэффициентов
├── images/                            # эталонные изображения и ЦВЗ
└── GRADIENT_BASED_OPTIMIZER/
    ├── gbo_app.cpp                    # точка входа CLI-приложения
    ├── GRADIENT_BASED_OPTIMIZER.cpp   # точка входа тестового стенда
    └── src/
        ├── gbo_api.{h,cpp}            # ПУБЛИЧНЫЙ API БИБЛИОТЕКИ (namespace gbo)
        ├── gbo.{h,cpp}                # ядро градиентного оптимизатора
        ├── population.{h,cpp}         # популяция, целевая функция (с учётом атак)
        ├── launch.{h,cpp}             # оркестрация экспериментов (только bench)
        ├── embedding_schemes.{h,cpp}  # менеджер схем / загрузка JSON
        ├── attacks.{h,cpp}            # JPEG, контраст, шум, кроп и т.д.
        ├── image_metrics.*            # PSNR / SSIM / NCC / BER
        ├── image_processing_custom.*  # разбиение на 8×8 / DCT / сборка
        ├── jpeg/                      # поблочное JPEG-сжатие, таблицы квантования
        └── ...                        # классификаторы (опционально, PyTorch)
```

---

## 2. Зависимости

Обязательные:

- **C++17** (g++ ≥ 9, clang ≥ 10)
- **CMake** ≥ 3.16
- **OpenCV** ≥ 4.0 (core, imgproc, imgcodecs)

Опциональные (только для режимов с классификатором):

- **PyTorch C++ (libtorch)** — CPU или CUDA, протестировано с libtorch 2.x
- веса моделей в формате TorchScript (`.pt`)

```bash
sudo apt-get install build-essential cmake libopencv-dev
```

### 2.1 Установка libtorch (опционально)

```bash
cd /tmp
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cpu.zip
```

---

## 3. Сборка

Система сборки предоставляет четыре CMake-опции для выбора целей.
Все по умолчанию `OFF`, кроме `GBO_BUILD_APP`.

| CMake-опция         | По умолч. | Результат                                    |
|---------------------|-----------|----------------------------------------------|
| `GBO_BUILD_APP`     | **ON**    | `gbo_app` — CLI-инструмент                   |
| `GBO_BUILD_BENCH`   | OFF       | `gbo_bench` — тестовый стенд                 |
| `GBO_BUILD_SHARED`  | OFF       | `libgbo.so` / `.dylib` — динамическая библ.  |
| `GBO_BUILD_STATIC`  | OFF       | `libgbo.a` — статическая библиотека          |

### 3.1 Только приложение (по умолчанию)

```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

### 3.2 Все цели сразу

```bash
mkdir -p build && cd build
cmake -DGBO_BUILD_APP=ON    \
      -DGBO_BUILD_BENCH=ON  \
      -DGBO_BUILD_SHARED=ON \
      -DGBO_BUILD_STATIC=ON ..
make -j$(nproc)
```

### 3.3 С поддержкой классификатора (PyTorch)

```bash
cmake -DCMAKE_PREFIX_PATH=$HOME/libtorch \
      -DGBO_BUILD_BENCH=ON ..
```

### 3.4 Установка

```bash
cmake --install build --prefix /usr/local
```

Устанавливает:

- `include/gbo/gbo_api.h` — публичный заголовок
- `lib/libgbo.{so,a}` — библиотеки (если собраны)
- `bin/gbo_app`, `bin/gbo_bench` — исполняемые файлы (если собраны)
- `share/gbo/embedding_schemes.json` — описания схем

---

## 4. CLI-приложение (`gbo_app`)

**Важно:** запускайте из корня репозитория.

```bash
# Встроить ЦВЗ
./build/gbo_app embed images/lenna.png images/watermark.png output.png --scheme scheme1

# Извлечь ЦВЗ
./build/gbo_app extract output.png extracted_wm.png

# Вычислить метрики
./build/gbo_app metrics images/lenna.png output.png

# Симулировать атаку
./build/gbo_app attack output.png attacked.png --type jpeg --param 70

# Список схем
./build/gbo_app schemes
```

---

## 4a. Тестовый стенд (`gbo_bench`)

Полный pipeline из статьи: встраивание и извлечение ЦВЗ на 8 эталонных
изображениях, симуляция атак, отчёт min/avg/max метрик.

Собирается с `-DGBO_BUILD_BENCH=ON`:

```bash
./build/gbo_bench              # 10 итераций на изображение
./build/gbo_bench --test       # 1 итерация (smoke-тест)
./build/gbo_bench --scheme scheme2 --test
```

Также поддерживает генерацию датасетов (`--dataset`), режимы с
классификаторами и known-attack. `./build/gbo_bench --help` — полный список.

---

## 5. Схемы встраивания

*Схема* — детерминированный набор DCT-коэффициентов внутри 8×8 блока,
разбитый на `REG0`, `REG1` и `ZONE0`. Описываются в
[`embedding_schemes.json`](embedding_schemes.json) — **пересборка не нужна**.

| ID схемы          | Размер ZONE0 | Описание                                   |
|-------------------|--------------|---------------------------------------------|
| `scheme1`         | 22           | исходная, симметричная антидиагональ        |
| `scheme2`         | 22           | альтернативная, смещённые средние частоты   |
| `scheme3`         | 25           | 12+13, добавлены `[2,2]`, `[1,3]`, `[3,1]` |
| `extended_scheme` | 25           | 12+13, добавлены `[2,3]`, `[1,4]`, `[3,2]` |
| `standard_scheme` | 22           | непрерывная полоса 11+11                    |

### 5.1 Добавление своей схемы

Допишите объект в `embedding_schemes.json`:

```json
"my_scheme": {
  "name": "My Scheme",
  "description": "...",
  "REG0":  [[7,0],[6,0], "..."],
  "REG1":  [[5,2],[4,2], "..."],
  "ZONE0": [[7,0],[6,0], "..."]
}
```

Запуск: `--scheme my_scheme`. Пересборка не требуется.

---

## 6. Алгоритм

1. Загрузка изображения и перевод в grayscale.
2. Разбиение на 8×8 блоки, прямое DCT.
3. Для каждого бита 1024-битного ЦВЗ:
   1. Блок с индексом `i mod WM_SIZE`.
   2. (Опц.) классификатор выбирает схему.
   3. Популяция `POP_SIZE = 30` векторов в `[-TH, +TH]`.
   4. GBO за `ITERATIONS = 40` поколений оптимизирует целевую функцию.
   5. Лучшее возмущение → DCT → обратное DCT.
4. Сборка результирующего изображения.
5. Опционально: атаки + метрики (PSNR / SSIM / NCC / BER).

Константы в [`config.h`](GRADIENT_BASED_OPTIMIZER/src/config.h):

```cpp
#define POP_SIZE   30      // размер популяции
#define ITERATIONS 40      // итерации оптимизатора
#define TH         10.0    // граница возмущений
#define WM_SIZE    1024    // длина ЦВЗ в битах
```

---

## 7. Квадрантное встраивание (опц.)

Изображение 1024×1024 делится на 4 квадранта, в каждый встраивается копия
ЦВЗ, оптимизированная под свой тип атаки:

| Квадрант           | Атака                      |
|--------------------|----------------------------|
| N1 (верх-лево)     | `AttackType::NONE`         |
| N2 (верх-право)    | `AttackType::JPEG70`       |
| N3 (низ-лево)      | `AttackType::CONTRAST`     |
| N4 (низ-право)     | `AttackType::SALT_PEPPER`  |

При извлечении классификатор выбирает лучший квадрант.
Требует libtorch + `best_model_ultrahighres.pt`.

---

## 8. API библиотеки (`libgbo`)

Соберите shared- или static-библиотеку и линкуйтесь.
Публичный заголовок: [`gbo_api.h`](GRADIENT_BASED_OPTIMIZER/src/gbo_api.h).

### 8.1 Пример

```cpp
#include <gbo/gbo_api.h>
#include <opencv2/opencv.hpp>

int main() {
    gbo::init("embedding_schemes.json");
    gbo::setScheme("scheme1");

    cv::Mat cover = cv::imread("cover.png", cv::IMREAD_GRAYSCALE);
    cv::Mat wm    = cv::imread("watermark.png", cv::IMREAD_GRAYSCALE);
    cv::Mat watermarked = gbo::embedWatermark(cover, wm);
    cv::imwrite("watermarked.png", watermarked);

    cv::Mat attacked  = gbo::attackJPEG(watermarked, 70);
    cv::Mat extracted = gbo::extractWatermark(attacked);

    std::cout << "PSNR: " << gbo::computePSNR(cover, watermarked) << " dB\n";
    std::cout << "BER:  " << gbo::computeBER(wm, extracted) << "\n";
}
```

Компиляция и линковка:

```bash
g++ -std=c++17 my_app.cpp -lgbo -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -o my_app
```

### 8.2 Справочник API

Все функции в пространстве имён `gbo`.

**Инициализация:**

| Функция | Описание |
|---------|----------|
| `bool init(path)` | Загрузить схемы и таблицы квантования. Вызвать один раз. |
| `bool setScheme(id)` | Выбрать активную схему. |
| `vector<string> availableSchemes()` | Список доступных схем. |

**Встраивание/извлечение:**

| Функция | Описание |
|---------|----------|
| `cv::Mat embedWatermark(image, watermark)` | Встроить ЦВЗ. Возвращает изображение. |
| `cv::Mat extractWatermark(image)` | Извлечь ЦВЗ. |

**Метрики:**

| Функция | Описание |
|---------|----------|
| `double computeMSE(a, b)` | Среднеквадратичная ошибка. |
| `double computePSNR(a, b)` | PSNR (дБ). |
| `double computeSSIM(a, b)` | Структурное сходство. |
| `double computeNCC(a, b)` | Нормированная корреляция. |
| `double computeBER(wm1, wm2)` | Вероятность битовой ошибки. |

**Симуляция атак:**

| Функция | Описание |
|---------|----------|
| `attackJPEG(image, quality)` | JPEG-сжатие. |
| `attackBrightnessIncrease(image, value)` | Увеличение яркости. |
| `attackBrightnessDecrease(image, value)` | Уменьшение яркости. |
| `attackContrastIncrease(image, alpha)` | Увеличение контраста. |
| `attackContrastDecrease(image, alpha)` | Уменьшение контраста. |
| `attackSaltPepper(image, prob)` | Шум «соль-перец». |
| `attackMedianFilter(image, ksize)` | Медианная фильтрация. |
| `attackGaussianFilter(image, ksize)` | Гауссова фильтрация. |

### 8.3 Линковка из CMake-проекта

```cmake
find_package(OpenCV REQUIRED)
add_executable(my_app main.cpp)
target_link_libraries(my_app /usr/local/lib/libgbo.so ${OpenCV_LIBS})
target_include_directories(my_app PRIVATE /usr/local/include)
```
