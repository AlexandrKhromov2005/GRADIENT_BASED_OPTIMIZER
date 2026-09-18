# Gradient-Based Optimizer for DCT-Domain Digital Watermarking

*[Русская версия](README.ru.md)*

C++ implementation of the image watermarking scheme of Melman and Evsutin
(Computers and Electrical Engineering 117 (2024) 109271). One watermark bit is stored in
every 8×8 block as the relation between two sums of absolute mid-frequency DCT
coefficients; the change applied to the coefficients is found by the gradient-based
optimizer (GBO) of Ahmadianfar et al. (Information Sciences 540 (2020)).

The repository also contains a modification of the scheme: the optimizer's objective
function simulates an attack (JPEG or contrast change), different parts of the image are
optimized for different attacks, and at extraction the copies prepared for the detected
attack are used (section 8).

## 1. Contents

```text
.
├── CMakeLists.txt                     # build, see section 3
├── build.sh                           # default build; adds libtorch to the path if found
├── embedding_schemes.json             # embedding schemes, read at start-up (section 6)
├── images/                            # 8 test images 1024×1024, watermark 32×32
├── tools/perf_bench.cpp               # gbo_perf: timing, image hash, self-test
├── perf_results/                      # raw output behind the numbers in section 10
├── fuzz/                              # libFuzzer targets
├── tests/                             # unit tests (ctest)
└── GRADIENT_BASED_OPTIMIZER/
    ├── gbo_app.cpp                    # CLI entry point
    ├── GRADIENT_BASED_OPTIMIZER.cpp   # test bench entry point
    └── src/
        ├── gbo_api.{h,cpp}            # public library API (namespace gbo)
        ├── gbo.{h,cpp}                # the optimizer
        ├── population.{h,cpp}         # population and objective function
        ├── embedding_core.{h,cpp}     # whole-image embedding / extraction, threads, seeds
        ├── block_kernels.{h,cpp}      # 8×8 DCT, JPEG round trip, contrast, rounding
        ├── random_utils.{h,cpp}       # random number streams
        ├── embedding_schemes.{h,cpp}  # scheme manager
        ├── scheme_json.{h,cpp}        # parser of embedding_schemes.json
        ├── attacks.{h,cpp}            # attacks on whole images
        ├── image_metrics.*            # MSE / PSNR / SSIM / NCC / BER
        ├── image_processing_custom.*  # block split / merge, watermark <-> bits
        ├── launch.{h,cpp}             # experiments of the test bench
        └── ...                        # classifier integration (needs libtorch)
```

## 2. Requirements

- C++17 compiler (g++ ≥ 9 or clang ≥ 10)
- CMake ≥ 3.16
- OpenCV ≥ 4.0 (core, imgproc, imgcodecs)

```bash
sudo apt-get install build-essential cmake libopencv-dev
```

Optional:

- libtorch 2.x and TorchScript model files (`.pt`) for the classifier modes of the test
  bench. The model files are not part of the repository.
- clang with libFuzzer for the fuzz targets.

libtorch is looked up in `CMAKE_PREFIX_PATH` and in `/tmp/libtorch`:

```bash
cd /tmp
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cpu.zip
```

## 3. Building

| CMake option        | Default | Product                                     |
|---------------------|---------|---------------------------------------------|
| `GBO_BUILD_APP`     | ON      | `gbo_app` - command-line tool               |
| `GBO_BUILD_BENCH`   | OFF     | `gbo_bench` - test bench                    |
| `GBO_BUILD_SHARED`  | OFF     | `libgbo.so` - shared library                |
| `GBO_BUILD_STATIC`  | OFF     | `libgbo.a` - static library                 |
| `GBO_BUILD_PERF`    | OFF     | `gbo_perf` - speed / quality benchmark      |
| `GBO_BUILD_FUZZ`    | OFF     | `fuzz_*` - libFuzzer targets (needs clang)  |
| `GBO_BUILD_TESTS`   | OFF     | `scheme_json_test`, run with `ctest`        |

```bash
cmake -B build                       # gbo_app only
cmake --build build -j

cmake -B build -DGBO_BUILD_BENCH=ON -DGBO_BUILD_SHARED=ON \
      -DGBO_BUILD_STATIC=ON -DGBO_BUILD_PERF=ON
cmake --build build -j
```

With libtorch add `-DCMAKE_PREFIX_PATH=/path/to/libtorch`. When it is found,
`TORCH_AVAILABLE` is defined, the classifier sources are compiled in and the bench
additionally builds `classifier_example` and `single_classifier_example`.

`cmake --install build --prefix /usr/local` installs `include/gbo/gbo_api.h`, the libraries
and executables that were built, and `share/gbo/embedding_schemes.json`.

## 4. Command-line tool (`gbo_app`)

Run it from the repository root: it opens `embedding_schemes.json` in the current
directory.

```bash
./build/gbo_app embed images/lenna.png images/watermark_32x32.png marked.png --scheme scheme1
./build/gbo_app attack marked.png attacked.png --type jpeg --param 70
./build/gbo_app extract attacked.png extracted_wm.png --scheme scheme1
./build/gbo_app metrics images/lenna.png marked.png \
    --wm-orig images/watermark_32x32.png --wm-extr extracted_wm.png
./build/gbo_app schemes
```

`--scheme` defaults to `scheme1`; an unknown id is an error. `embed` also takes
`--threads N` and `--seed S`. Attack types: `jpeg`, `brightness+`,
`brightness-`, `contrast+`, `contrast-`, `salt-pepper`, `median`, `gaussian`.

The watermark is a black-and-white image of 32×32 pixels (black = 1). From a larger image
only the first 1024 pixels in row order are used.

## 5. Test bench (`gbo_bench`)

Embeds and extracts the watermark on the 8 test images, applies the attacks and writes
min/avg/max of MSE, PSNR, SSIM, NCC and BER.

```bash
./build/gbo_bench                          # 10 runs per image
./build/gbo_bench --test                   # 1 run per image
./build/gbo_bench --scheme scheme2 --test
./build/gbo_bench --known-attack-1024 --test
```

`--known-attack-1024` is the modified algorithm with the attack type given instead of
predicted; it reads the images from `test_images_1024/`, which is not part of the
repository (the files in `images/` have the right size). The other modes generate
classifier datasets (`--dataset`, `--attack-dataset`) or run with a classifier; the latter
and `--quadrant-dataset` need libtorch. `./build/gbo_bench --help` lists them.

## 6. Embedding schemes

A scheme is a set of DCT coefficient positions in the 8×8 block: `REG0` and `REG1` give the
two sums that encode the bit, `ZONE0` lists the coefficients the optimizer may change.

| Scheme id         | `REG0` + `REG1` | `ZONE0` |
|-------------------|-----------------|---------|
| `scheme1`         | 11 + 11         | 22      |
| `scheme2`         | 11 + 11         | 22      |
| `scheme3`         | 12 + 13         | 25      |
| `extended_scheme` | 12 + 13         | 25      |
| `standard_scheme` | 11 + 11         | 22      |

The schemes are read from [`embedding_schemes.json`](embedding_schemes.json) at start-up,
so a scheme is added by editing that file:

```json
"my_scheme": {
  "name": "My scheme",
  "description": "optional",
  "REG0":  [[6, 1], [5, 2], [4, 3]],
  "REG1":  [[6, 0], [5, 1], [4, 2]],
  "ZONE0": [[6, 1], [5, 2], [4, 3], [6, 0], [5, 1], [4, 2]]
}
```

Positions are `[row, col]` with integers 0..7. `REG0`, `REG1` and `ZONE0` are required, not
empty and without repeated positions; `REG0` and `REG1` must not overlap, and every `ZONE0`
position has to be in `REG0` or `REG1`. Ids, names and descriptions are UTF-8 without
control characters. The order of
`ZONE0` matters for reproducibility: element `i` of the optimizer's vector changes
coefficient `i` of the list. Other members are ignored. A file that breaks these rules or
is not valid JSON is rejected with the line and column of the problem, and the schemes
loaded before stay in place. Embedding and extraction have to use the same scheme.

## 7. Algorithm

Embedding:

1. The image is converted to grayscale and split into 8×8 blocks; block `i` carries
   watermark bit `i mod 1024`, so a 512×512 image holds 4 copies of the watermark.
2. For every block GBO searches for a vector of changes to the absolute values of the
   `ZONE0` coefficients, each within `[-TH, TH]`. The population has `POP_SIZE` vectors and
   is evolved for `ITERATIONS` iterations.
3. The objective function, which is minimized, is `S1/S0 - 0.01·PSNR` for bit 0 and
   `S0/S1 - 0.01·PSNR` for bit 1. `S0` and `S1` are the sums of absolute DCT coefficients
   over `REG0` and `REG1` of the modified block after rounding to 8 bits and, if an attack
   type is set, after that attack; PSNR is taken against the original block.
4. The best vector is applied and the block is written back. A border narrower than 8
   pixels is left unchanged.

Extraction: a block gives 1 if `S0 < S1`, otherwise 0; every watermark bit is the majority
vote over its copies.

Constants are in [`config.h`](GRADIENT_BASED_OPTIMIZER/src/config.h):

```cpp
#define POP_SIZE   30      // population size
#define ITERATIONS 40      // optimizer iterations
#define TH         10.0    // bound of a coefficient change
#define WM_SIZE    1024    // watermark length in bits
```

## 8. Modified algorithm (quadrants)

A 1024×1024 image is divided into a 4×4 grid of 256×256 quadrants. Each quadrant has 1024
blocks and carries one full copy of the watermark. The quadrant in grid row `r`, column
`c` is optimized with the attack at position `(r mod 2, c mod 2)` of this table simulated
in the objective function, so every attack type has 4 copies:

|              | even column   | odd column |
|--------------|---------------|------------|
| **even row** | no attack     | JPEG 70    |
| **odd row**  | contrast ×1.1 | JPEG 80    |

At extraction the attack type is either known (`--known-attack-1024`) or predicted by a
classifier (`--attack-classifier`, needs libtorch and `model_torchscript.pt`). The
watermark is then the majority vote over the 4 quadrants prepared for that type.

This mode is available in `gbo_bench` and `gbo_perf --mode quad`. The library API and
`gbo_app` implement the base algorithm only.

## 9. Library API (`libgbo`)

Build with `-DGBO_BUILD_SHARED=ON` or `-DGBO_BUILD_STATIC=ON`. The header is
[`gbo_api.h`](GRADIENT_BASED_OPTIMIZER/src/gbo_api.h), installed as `gbo/gbo_api.h`.

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

| Function | Description |
|----------|-------------|
| `bool init(path)` | Loads the schemes from the JSON file and sets up the tables; `false` if the file is missing or invalid. Call first. |
| `bool setScheme(id)` | Selects the scheme; `false` if there is no such id. Not while an embedding or extraction is running. |
| `vector<string> availableSchemes()` | Scheme ids. |
| `void setThreads(n)` | Worker threads for embedding; 0 (default) = all hardware threads or `GBO_THREADS`. |
| `void setSeed(seed)`, `void clearSeed()` | Reproducible embedding / back to a fresh random seed per call. |
| `cv::Mat embedWatermark(image, watermark)` | Returns the watermarked `CV_8UC1` image of the same size. |
| `cv::Mat extractWatermark(image)` | Returns the 32×32 `CV_8UC1` watermark. A tie between copies is resolved at random. |
| `computeMSE`, `computePSNR`, `computeSSIM`, `computeNCC` `(a, b)` | Image quality metrics. |
| `double computeBER(wm1, wm2)` | Bit error rate between two watermarks. |
| `attackJPEG(image, quality)` | JPEG compression. |
| `attackBrightnessIncrease`, `attackBrightnessDecrease` `(image, value)` | Brightness shift. |
| `attackContrastIncrease`, `attackContrastDecrease` `(image, alpha)` | Contrast change. |
| `attackSaltPepper(image, prob)` | Salt-and-pepper noise. |
| `attackMedianFilter`, `attackGaussianFilter` `(image, ksize)` | Filtering. |

`embedWatermark` and `extractWatermark` accept images with 8 bits per channel and 1, 3 or
4 channels (colour is converted to grayscale) and throw `std::invalid_argument` for an
empty image or any other type. `embedWatermark` also throws if the watermark is not
`CV_8UC1` or has fewer than 1024 pixels.

## 10. Performance

Embedding runs the optimizer `POP_SIZE x (ITERATIONS + 1) = 1230` times per 8x8 block, so
the objective function is the hot path. It is implemented without per-call `cv::Mat`
objects, DCT plans or a JPEG codec:

- **8x8 DCT / IDCT** - Loeffler-Ligtenberg-Moschytz factorization in double precision
  (equals `cv::dct` / `cv::idct` to ~1e-12).
- **JPEG attack inside the objective function** - an integer emulation of libjpeg's
  baseline pipeline for a single 8x8 block (quality-scaled luminance table, `islow`
  forward DCT, quantization, dequantization, `islow` inverse DCT). It is bit-exact with
  `cv::imencode` + `cv::imdecode`; this is probed at start-up and the codec is used as a
  fallback if the probe ever fails.
- **Contrast attack** - a 256-entry table produced by the same `convertTo` call.
- **Blocks are independent**, so they are embedded on all hardware threads. Every block
  draws from its own random stream derived from `(seed, block index)`: the result depends
  on the seed only, never on the number of threads.
- **Extraction** reads blocks in place and only falls back to `cv::dct` when `S0` and `S1`
  are equal in exact arithmetic, so every extracted bit is the same as before.

Population size, number of iterations, the objective function and every random draw are
unchanged. With a fixed seed and a single
random stream the optimized code produces **bit-identical** watermarked images
(verified by image hash on full-size images, base and quadrant mode). With per-block
streams the results are statistically indistinguishable (8 images x 10 seeds for the base
algorithm and x 5 for the quadrant one, all quality
and robustness metrics within run-to-run noise, see `perf_results/`).

Measured on Intel Core i5-11300H (4 cores / 8 threads), `lenna`, `scheme1`:

| Operation                                   | Before   | After, 1 thread | After, 8 threads |
|---------------------------------------------|----------|-----------------|------------------|
| Base algorithm, embed 512x512 (4096 blocks) | 24.7 s   | 4.7 s           | **1.4 s** (x17)  |
| Quadrant algorithm, embed 1024x1024 (16384) | 158.8 s  | 22.3 s          | **6.2 s** (x25)  |
| Extraction of one watermark (4096 blocks)   | 5.2 ms   | **0.6 ms** (x8) | -                |

In the quadrant pipeline the extraction time is dominated by the attack-type classifier
(ResNet-50 at 1024x1024 with flip TTA, about 2 s per image on this CPU); the bit
extraction itself is the 0.6 ms above.

### 10.1 Threads and reproducibility

```cpp
gbo::setThreads(4);   // default 0 = all hardware threads (or the GBO_THREADS env variable)
gbo::setSeed(42);     // reproducible embedding; gbo::clearSeed() restores random seeding
```

```bash
./build/gbo_app embed cover.png wm.png out.png --threads 4 --seed 42
```

### 10.2 Benchmark and self-test (`gbo_perf`)

```bash
cmake -DGBO_BUILD_PERF=ON .. && make gbo_perf
./build/gbo_perf --mode base --image images/lenna.png --crop 512 --seed 1 --repeat 5
./build/gbo_perf --mode quad --image images/lenna.png --threads 1
./build/gbo_perf --selftest 3 --image images/baboon.png
```

It prints embedding/extraction time, a hash of the watermarked image, PSNR/SSIM and the
BER after a set of attacks. Same seed + same hash = same computation, which is how every
optimization step was checked. `--selftest` compares the fast kernels with the OpenCV
reference (JPEG round trip, contrast, rounding, extracted bits must match exactly).

### 10.3 Fuzzing

Six libFuzzer targets (ASan + UBSan) cover extraction, embedding, the optimizer on a single
block, the 8x8 kernels (differentially, against OpenCV), the metrics/attacks API and the
parser of `embedding_schemes.json`. See
`fuzz/README.md` for what each one checks and how to run it.

```bash
cmake -B build_fuzz -DCMAKE_CXX_COMPILER=clang++ -DGBO_BUILD_FUZZ=ON -DGBO_BUILD_APP=OFF
cmake --build build_fuzz -j
```
