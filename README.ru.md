# Gradient-Based Optimizer для встраивания ЦВЗ в область ДКП

*[English version](README.md)*

Реализация на C++ алгоритма встраивания цифровых водяных знаков Мельман и Евсютина
(Computers and Electrical Engineering 117 (2024) 109271). Один бит ЦВЗ хранится в каждом
блоке 8×8 как соотношение двух сумм модулей среднечастотных коэффициентов ДКП; изменение
коэффициентов подбирает градиентный оптимизатор GBO (Ahmadianfar и др., Information
Sciences 540 (2020)).

В репозитории есть и модификация алгоритма: целевая функция оптимизатора моделирует атаку
(JPEG или изменение контраста), разные части изображения оптимизируются под разные атаки,
а при извлечении используются копии, подготовленные под распознанную атаку (раздел 8).

## 1. Состав репозитория

```text
.
├── CMakeLists.txt                     # сборка, см. раздел 3
├── build.sh                           # сборка по умолчанию; подставляет путь к libtorch, если он найден
├── embedding_schemes.json             # схемы встраивания, читаются при запуске (раздел 6)
├── images/                            # 8 тестовых изображений 1024×1024, ЦВЗ 32×32
├── tools/perf_bench.cpp               # gbo_perf: время, хэш изображения, самопроверка
├── perf_results/                      # исходные данные для чисел из раздела 10
├── fuzz/                              # цели libFuzzer
├── tests/                             # модульные тесты (ctest)
└── GRADIENT_BASED_OPTIMIZER/
    ├── gbo_app.cpp                    # точка входа CLI
    ├── GRADIENT_BASED_OPTIMIZER.cpp   # точка входа тестового стенда
    └── src/
        ├── gbo_api.{h,cpp}            # публичный API библиотеки (namespace gbo)
        ├── gbo.{h,cpp}                # оптимизатор
        ├── population.{h,cpp}         # популяция и целевая функция
        ├── embedding_core.{h,cpp}     # встраивание / извлечение по изображению, потоки, seed
        ├── block_kernels.{h,cpp}      # ДКП 8×8, JPEG для блока, контраст, округление
        ├── random_utils.{h,cpp}       # потоки случайных чисел
        ├── embedding_schemes.{h,cpp}  # менеджер схем
        ├── scheme_json.{h,cpp}        # разбор embedding_schemes.json
        ├── attacks.{h,cpp}            # атаки на изображение
        ├── image_metrics.*            # MSE / PSNR / SSIM / NCC / BER
        ├── image_processing_custom.*  # разбиение на блоки и сборка, ЦВЗ <-> биты
        ├── launch.{h,cpp}             # эксперименты тестового стенда
        └── ...                        # интеграция классификаторов (нужен libtorch)
```

## 2. Зависимости

- компилятор C++17 (g++ ≥ 9 или clang ≥ 10)
- CMake ≥ 3.16
- OpenCV ≥ 4.0 (core, imgproc, imgcodecs)

```bash
sudo apt-get install build-essential cmake libopencv-dev
```

Необязательные:

- libtorch 2.x и модели в формате TorchScript (`.pt`) для режимов стенда с классификатором.
  Файлов моделей в репозитории нет.
- clang с libFuzzer для фаззинг-целей.

libtorch ищется в `CMAKE_PREFIX_PATH` и в `/tmp/libtorch`:

```bash
cd /tmp
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cpu.zip
```

## 3. Сборка

| Опция CMake         | По умолчанию | Результат                                   |
|---------------------|--------------|---------------------------------------------|
| `GBO_BUILD_APP`     | ON           | `gbo_app` — утилита командной строки        |
| `GBO_BUILD_BENCH`   | OFF          | `gbo_bench` — тестовый стенд                |
| `GBO_BUILD_SHARED`  | OFF          | `libgbo.so` — динамическая библиотека       |
| `GBO_BUILD_STATIC`  | OFF          | `libgbo.a` — статическая библиотека         |
| `GBO_BUILD_PERF`    | OFF          | `gbo_perf` — замер скорости и качества      |
| `GBO_BUILD_FUZZ`    | OFF          | `fuzz_*` — цели libFuzzer (нужен clang)     |
| `GBO_BUILD_TESTS`   | OFF          | `scheme_json_test`, запуск через `ctest`    |

```bash
cmake -B build                       # только gbo_app
cmake --build build -j

cmake -B build -DGBO_BUILD_BENCH=ON -DGBO_BUILD_SHARED=ON \
      -DGBO_BUILD_STATIC=ON -DGBO_BUILD_PERF=ON
cmake --build build -j
```

Для сборки с libtorch добавьте `-DCMAKE_PREFIX_PATH=/path/to/libtorch`. Если библиотека
найдена, определяется `TORCH_AVAILABLE`, в сборку попадают исходники классификаторов, а
вместе со стендом собираются `classifier_example` и `single_classifier_example`.

`cmake --install build --prefix /usr/local` устанавливает `include/gbo/gbo_api.h`, собранные
библиотеки и исполняемые файлы и `share/gbo/embedding_schemes.json`.

## 4. Утилита командной строки (`gbo_app`)

Запускать из корня репозитория: утилита открывает `embedding_schemes.json` в текущем
каталоге.

```bash
./build/gbo_app embed images/lenna.png images/watermark_32x32.png marked.png --scheme scheme1
./build/gbo_app attack marked.png attacked.png --type jpeg --param 70
./build/gbo_app extract attacked.png extracted_wm.png --scheme scheme1
./build/gbo_app metrics images/lenna.png marked.png \
    --wm-orig images/watermark_32x32.png --wm-extr extracted_wm.png
./build/gbo_app schemes
```

По умолчанию `--scheme` равен `scheme1`; неизвестный идентификатор считается ошибкой. У
`embed` есть также `--threads N` и `--seed S`. Типы атак: `jpeg`, `brightness+`,
`brightness-`, `contrast+`, `contrast-`, `salt-pepper`, `median`, `gaussian`.

ЦВЗ — чёрно-белое изображение 32×32 пикселя (чёрный = 1). Из изображения большего размера
берутся только первые 1024 пикселя по строкам.

## 5. Тестовый стенд (`gbo_bench`)

Встраивает и извлекает ЦВЗ на 8 тестовых изображениях, применяет атаки и записывает
min/avg/max для MSE, PSNR, SSIM, NCC и BER.

```bash
./build/gbo_bench                          # 10 прогонов на изображение
./build/gbo_bench --test                   # 1 прогон на изображение
./build/gbo_bench --scheme scheme2 --test
./build/gbo_bench --known-attack-1024 --test
```

`--known-attack-1024` — модифицированный алгоритм, в котором тип атаки задан, а не
предсказан; изображения читаются из каталога `test_images_1024/`, которого в репозитории
нет (файлы из `images/` подходят по размеру). Остальные режимы генерируют датасеты для
классификаторов (`--dataset`, `--attack-dataset`) или работают с классификатором; последним
и режиму `--quadrant-dataset` нужен libtorch. Список режимов выводит
`./build/gbo_bench --help`.

## 6. Схемы встраивания

Схема — набор позиций коэффициентов ДКП в блоке 8×8: `REG0` и `REG1` задают две суммы,
соотношение которых кодирует бит, `ZONE0` перечисляет коэффициенты, которые может менять
оптимизатор.

| Схема             | `REG0` + `REG1` | `ZONE0` |
|-------------------|-----------------|---------|
| `scheme1`         | 11 + 11         | 22      |
| `scheme2`         | 11 + 11         | 22      |
| `scheme3`         | 12 + 13         | 25      |
| `extended_scheme` | 12 + 13         | 25      |
| `standard_scheme` | 11 + 11         | 22      |

Схемы читаются из [`embedding_schemes.json`](embedding_schemes.json) при запуске, поэтому
новая схема добавляется правкой этого файла:

```json
"my_scheme": {
  "name": "My scheme",
  "description": "optional",
  "REG0":  [[6, 1], [5, 2], [4, 3]],
  "REG1":  [[6, 0], [5, 1], [4, 2]],
  "ZONE0": [[6, 1], [5, 2], [4, 3], [6, 0], [5, 1], [4, 2]]
}
```

Позиции записываются как `[строка, столбец]`, целые от 0 до 7. `REG0`, `REG1` и `ZONE0`
обязательны, не пусты и не содержат повторов; `REG0` и `REG1` не должны пересекаться, а
каждая позиция из `ZONE0` должна входить в `REG0` или `REG1`. Идентификаторы, имена и
описания записываются в UTF-8 без управляющих символов. Порядок `ZONE0` важен для воспроизводимости: элемент `i` вектора оптимизатора меняет
коэффициент `i` из списка. Остальные поля игнорируются. Файл, который нарушает эти правила
или не является корректным JSON, отклоняется с указанием строки и столбца ошибки, а ранее
загруженные схемы остаются на месте. Встраивание и извлечение должны использовать одну и ту
же схему.

## 7. Алгоритм

Встраивание:

1. Изображение переводится в оттенки серого и разбивается на блоки 8×8; блок `i` несёт бит
   ЦВЗ с номером `i mod 1024`, так что в изображении 512×512 помещаются 4 копии ЦВЗ.
2. Для каждого блока GBO ищет вектор изменений модулей коэффициентов из `ZONE0`, каждое в
   пределах `[-TH, TH]`. Популяция из `POP_SIZE` векторов развивается `ITERATIONS` итераций.
3. Минимизируется целевая функция `S1/S0 - 0.01·PSNR` для бита 0 и `S0/S1 - 0.01·PSNR` для
   бита 1. `S0` и `S1` — суммы модулей коэффициентов ДКП по `REG0` и `REG1` изменённого
   блока после округления до 8 бит и, если задан тип атаки, после этой атаки; PSNR считается
   относительно исходного блока.
4. Лучший вектор применяется, блок записывается обратно. Кайма уже 8 пикселей не меняется.

Извлечение: блок даёт 1, если `S0 < S1`, иначе 0; каждый бит ЦВЗ определяется большинством
голосов его копий.

Константы заданы в [`config.h`](GRADIENT_BASED_OPTIMIZER/src/config.h):

```cpp
#define POP_SIZE   30      // размер популяции
#define ITERATIONS 40      // число итераций оптимизатора
#define TH         10.0    // граница изменения коэффициента
#define WM_SIZE    1024    // длина ЦВЗ в битах
```

## 8. Модифицированный алгоритм (квадранты)

Изображение 1024×1024 делится сеткой 4×4 на квадранты 256×256. В квадранте 1024 блока, и он
несёт одну полную копию ЦВЗ. Квадрант в строке `r` и столбце `c` сетки оптимизируется с
атакой из ячейки `(r mod 2, c mod 2)` этой таблицы, смоделированной в целевой функции, так
что на каждый тип атаки приходится 4 копии:

|                   | чётный столбец | нечётный столбец |
|-------------------|----------------|------------------|
| **чётная строка**   | без атаки      | JPEG 70          |
| **нечётная строка** | контраст ×1.1  | JPEG 80          |

При извлечении тип атаки либо известен (`--known-attack-1024`), либо предсказывается
классификатором (`--attack-classifier`, нужны libtorch и `model_torchscript.pt`). ЦВЗ
получается голосованием по 4 квадрантам, подготовленным под этот тип.

Режим доступен в `gbo_bench` и в `gbo_perf --mode quad`. API библиотеки и `gbo_app`
реализуют только базовый алгоритм.

## 9. API библиотеки (`libgbo`)

Сборка с `-DGBO_BUILD_SHARED=ON` или `-DGBO_BUILD_STATIC=ON`. Заголовок -
[`gbo_api.h`](GRADIENT_BASED_OPTIMIZER/src/gbo_api.h), устанавливается как `gbo/gbo_api.h`.

```cpp
#include <gbo/gbo_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    if (!gbo::init("embedding_schemes.json")) return 1;
    gbo::setScheme("scheme1");

    cv::Mat cover = cv::imread("cover.png", cv::IMREAD_GRAYSCALE);
    cv::Mat wm    = cv::imread("watermark_32x32.png", cv::IMREAD_GRAYSCALE);
    cv::Mat marked = gbo::embedWatermark(cover, wm);

    cv::Mat attacked  = gbo::attackJPEG(marked, 70);
    cv::Mat extracted = gbo::extractWatermark(attacked);

    std::cout << "PSNR " << gbo::computePSNR(cover, marked) << " dB, "
              << "BER " << gbo::computeBER(wm, extracted) << "\n";
}
```

```bash
g++ -std=c++17 my_app.cpp -lgbo $(pkg-config --cflags --libs opencv4) -o my_app
```

```cmake
find_package(OpenCV REQUIRED)
add_executable(my_app main.cpp)
target_include_directories(my_app PRIVATE /usr/local/include)
target_link_libraries(my_app /usr/local/lib/libgbo.so ${OpenCV_LIBS})
```

| Функция | Описание |
|---------|----------|
| `bool init(path)` | Загружает схемы из JSON-файла и готовит таблицы; `false`, если файла нет или он некорректен. Вызывается первой. |
| `bool setScheme(id)` | Выбирает схему; `false`, если такой нет. Нельзя вызывать во время встраивания или извлечения. |
| `vector<string> availableSchemes()` | Идентификаторы схем. |
| `void setThreads(n)` | Число потоков встраивания; 0 (по умолчанию) = все аппаратные потоки или `GBO_THREADS`. |
| `void setSeed(seed)`, `void clearSeed()` | Воспроизводимое встраивание / возврат к новому случайному seed на каждый вызов. |
| `cv::Mat embedWatermark(image, watermark)` | Возвращает изображение с ЦВЗ, `CV_8UC1`, того же размера. |
| `cv::Mat extractWatermark(image)` | Возвращает ЦВЗ 32×32, `CV_8UC1`. Равенство голосов разрешается случайно. |
| `computeMSE`, `computePSNR`, `computeSSIM`, `computeNCC` `(a, b)` | Метрики качества изображения. |
| `double computeBER(wm1, wm2)` | Доля ошибочных битов между двумя ЦВЗ. |
| `attackJPEG(image, quality)` | JPEG-сжатие. |
| `attackBrightnessIncrease`, `attackBrightnessDecrease` `(image, value)` | Сдвиг яркости. |
| `attackContrastIncrease`, `attackContrastDecrease` `(image, alpha)` | Изменение контраста. |
| `attackSaltPepper(image, prob)` | Шум «соль и перец». |
| `attackMedianFilter`, `attackGaussianFilter` `(image, ksize)` | Фильтрация. |

`embedWatermark` и `extractWatermark` принимают изображения с 8 битами на канал и 1, 3 или 4
каналами (цветное переводится в оттенки серого) и бросают `std::invalid_argument` для
пустого изображения и любого другого типа. `embedWatermark` бросает его и тогда, когда ЦВЗ
не `CV_8UC1` или содержит меньше 1024 пикселей.

## 10. Производительность

При встраивании оптимизатор вызывает целевую функцию `POP_SIZE x (ITERATIONS + 1) = 1230`
раз на каждый блок 8x8, поэтому именно она — горячий путь. Она реализована без
создания `cv::Mat`, планов DCT и JPEG-кодека на каждый вызов:

- **DCT / IDCT 8x8** — факторизация Лёффлера — Лигтенберга — Мошица в двойной точности
  (совпадает с `cv::dct` / `cv::idct` до ~1e-12).
- **JPEG-атака внутри целевой функции** — целочисленная эмуляция baseline-конвейера
  libjpeg для одного блока 8x8 (масштабирование таблицы квантования по качеству, прямое
  `islow`-DCT, квантование, деквантование, обратное `islow`-DCT). Результат побитово
  совпадает с `cv::imencode` + `cv::imdecode`; это проверяется при старте, и при
  несовпадении используется кодек.
- **Контрастная атака** — таблица на 256 значений, построенная тем же вызовом `convertTo`.
- **Блоки независимы** и встраиваются на всех аппаратных потоках. У каждого блока свой
  поток случайных чисел, выведенный из `(seed, индекс блока)`: результат зависит только
  от seed и не зависит от числа потоков.
- **Извлечение** читает блоки на месте и обращается к `cv::dct` только когда `S0` и `S1`
  равны в точной арифметике, поэтому каждый извлечённый бит тот же, что и раньше.

Алгоритм не менялся: размер популяции, число итераций, целевая функция и каждое
обращение к генератору случайных чисел остались прежними. При фиксированном seed и едином
потоке случайных чисел оптимизированный код даёт **побитово идентичные** изображения
(проверено по хэшу на полноразмерных изображениях, базовый и квадрантный режим). С
поблочными потоками результаты статистически неотличимы (8 изображений x 10 seed для
базового алгоритма и x 5 для квадрантного, все
метрики качества и устойчивости в пределах разброса между запусками, см. `perf_results/`).

Замеры на Intel Core i5-11300H (4 ядра / 8 потоков), `lenna`, `scheme1`:

| Операция                                          | Было     | Стало, 1 поток   | Стало, 8 потоков |
|---------------------------------------------------|----------|------------------|------------------|
| Базовый алгоритм, встраивание 512x512 (4096 бл.)  | 24.7 с   | 4.7 с            | **1.4 с** (x17)  |
| Квадрантный алгоритм, 1024x1024 (16384 блока)     | 158.8 с  | 22.3 с           | **6.2 с** (x25)  |
| Извлечение одного ЦВЗ (4096 блоков)               | 5.2 мс   | **0.6 мс** (x8)  | -                |

В квадрантном конвейере время извлечения определяется в основном классификатором типа атаки
(ResNet-50 на 1024x1024 с TTA-отражением, около 2 с на изображение на этом CPU); само
извлечение битов — те же 0.6 мс.

### 10.1 Потоки и воспроизводимость

```cpp
gbo::setThreads(4);   // по умолчанию 0 = все аппаратные потоки (или переменная GBO_THREADS)
gbo::setSeed(42);     // воспроизводимое встраивание; gbo::clearSeed() возвращает случайный seed
```

```bash
./build/gbo_app embed cover.png wm.png out.png --threads 4 --seed 42
```

### 10.2 Бенчмарк и самопроверка (`gbo_perf`)

```bash
cmake -DGBO_BUILD_PERF=ON .. && make gbo_perf
./build/gbo_perf --mode base --image images/lenna.png --crop 512 --seed 1 --repeat 5
./build/gbo_perf --mode quad --image images/lenna.png --threads 1
./build/gbo_perf --selftest 3 --image images/baboon.png
```

Печатает время встраивания/извлечения, хэш изображения с ЦВЗ, PSNR/SSIM и BER после
набора атак. Одинаковый seed + одинаковый хэш = одинаковые вычисления; так проверялся
каждый шаг оптимизации. `--selftest` сравнивает быстрые ядра с эталоном OpenCV
(JPEG, контраст, округление и извлечённые биты должны совпадать точно).

### 10.3 Фаззинг

Шесть целей libFuzzer (ASan + UBSan): извлечение, встраивание, оптимизатор на одном блоке,
ядра 8x8 (дифференциально, против OpenCV), API метрик и атак и разбор
`embedding_schemes.json`. Что проверяет каждая цель и
как запускать — в `fuzz/README.md`.

```bash
cmake -B build_fuzz -DCMAKE_CXX_COMPILER=clang++ -DGBO_BUILD_FUZZ=ON -DGBO_BUILD_APP=OFF
cmake --build build_fuzz -j
```
