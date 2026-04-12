# Gradient-Based Optimizer for DCT-Domain Digital Watermarking

*[Русская версия этого документа](README.ru.md)*

This repository contains the reference C++ implementation of a gradient-based
meta-heuristic optimizer (GBO) used to embed digital watermarks into the DCT
domain of grayscale images with configurable robustness against JPEG
compression and other attacks. The optimizer works on 8×8 DCT blocks and
supports multiple coefficient-selection strategies ("embedding schemes"), as
well as optional integration with PyTorch classifiers that automatically
choose the best embedding scheme per block or per image quadrant.

The code is the research artifact behind the associated paper; it is intended
to be **reproducible** and **reusable** in downstream watermarking projects.

---

## 1. Contents

```text
.
├── CMakeLists.txt                     # top-level build with configurable targets
├── build.sh                           # convenience wrapper for cmake+make
├── embedding_schemes.json             # JSON definitions of the DCT coefficient sets
├── images/                            # canonical test images + watermarks
└── GRADIENT_BASED_OPTIMIZER/
    ├── gbo_app.cpp                    # standalone CLI entry point
    ├── GRADIENT_BASED_OPTIMIZER.cpp   # test bench entry point
    └── src/
        ├── gbo_api.{h,cpp}            # PUBLIC LIBRARY API (namespace gbo)
        ├── gbo.{h,cpp}                # gradient-based optimizer core
        ├── population.{h,cpp}         # population, fitness, attack-aware OF
        ├── launch.{h,cpp}             # experiment orchestration (bench only)
        ├── embedding_schemes.{h,cpp}  # scheme manager / JSON loader
        ├── attacks.{h,cpp}            # JPEG, contrast, salt-pepper, crop, etc.
        ├── image_metrics.*            # PSNR / SSIM / NCC / BER
        ├── image_processing_custom.*  # 8×8 block split / DCT / reconstruction
        ├── jpeg/                      # block-level JPEG compression + quant. tables
        └── ...                        # classifier integrations (optional, PyTorch)
```

---

## 2. Requirements

Mandatory:

- **C++17** compiler (g++ ≥ 9, clang ≥ 10)
- **CMake** ≥ 3.16
- **OpenCV** ≥ 4.0 (core, imgproc, imgcodecs)

Optional (only needed for classifier-based run modes):

- **PyTorch C++ (libtorch)** — CPU or CUDA build, tested with libtorch 2.x
- trained model weights in TorchScript format (`.pt`)

On Ubuntu 22.04 the mandatory dependencies are installed with:

```bash
sudo apt-get install build-essential cmake libopencv-dev
```

### 2.1 Installing libtorch (optional)

```bash
cd /tmp
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cpu.zip
```

Use either `/tmp/libtorch` or `$HOME/libtorch` — `build.sh` auto-detects both.

---

## 3. Building

The build system exposes four CMake options that let you choose exactly what
to produce.  All options default to `OFF` except `GBO_BUILD_APP`.

| CMake option        | Default | Product                                     |
|---------------------|---------|---------------------------------------------|
| `GBO_BUILD_APP`     | **ON**  | `gbo_app` — standalone CLI tool             |
| `GBO_BUILD_BENCH`   | OFF     | `gbo_bench` — test bench (full experiments) |
| `GBO_BUILD_SHARED`  | OFF     | `libgbo.so` / `.dylib` — shared library     |
| `GBO_BUILD_STATIC`  | OFF     | `libgbo.a` — static library                 |

### 3.1 Standalone app only (default)

```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

### 3.2 All targets at once

```bash
mkdir -p build && cd build
cmake -DGBO_BUILD_APP=ON    \
      -DGBO_BUILD_BENCH=ON  \
      -DGBO_BUILD_SHARED=ON \
      -DGBO_BUILD_STATIC=ON ..
make -j$(nproc)
```

### 3.3 With PyTorch classifier support

Add `-DCMAKE_PREFIX_PATH=$HOME/libtorch` (or wherever libtorch is installed).
CMake auto-detects `/tmp/libtorch` as well.

```bash
cmake -DCMAKE_PREFIX_PATH=$HOME/libtorch \
      -DGBO_BUILD_BENCH=ON ..
```

When libtorch is found, `TORCH_AVAILABLE` is defined and classifier sources
are compiled in.  The bench target additionally builds
`classifier_example` and `single_classifier_example`.

### 3.4 Installing

```bash
cmake --install build --prefix /usr/local
```

Installs:

- `include/gbo/gbo_api.h` — public header
- `lib/libgbo.{so,a}` — libraries (if built)
- `bin/gbo_app`, `bin/gbo_bench` — executables (if built)
- `share/gbo/embedding_schemes.json` — scheme definitions

---

## 4. Standalone CLI (`gbo_app`)

**Important:** run from the repository root so that `embedding_schemes.json`
and `images/` are found.

```bash
# Embed a watermark
./build/gbo_app embed images/lenna.png images/watermark.png output.png --scheme scheme1

# Extract a watermark
./build/gbo_app extract output.png extracted_wm.png

# Compute image quality metrics
./build/gbo_app metrics images/lenna.png output.png

# Simulate an attack
./build/gbo_app attack output.png attacked.png --type jpeg --param 70

# List available schemes
./build/gbo_app schemes
```

Run `./build/gbo_app --help` for the full usage reference.

---

## 4a. Test bench (`gbo_bench`)

The test bench runs the full experiment pipeline from the paper: embeds and
extracts the watermark on 8 canonical images, simulates multiple attacks,
and reports min/avg/max metrics (MSE, PSNR, SSIM, NCC, BER).

Build with `-DGBO_BUILD_BENCH=ON`, then:

```bash
# 10 iterations per image (default)
./build/gbo_bench

# Smoke test (1 iteration)
./build/gbo_bench --test

# Use a specific scheme
./build/gbo_bench --scheme scheme2 --test
```

The bench also supports dataset generation (`--dataset`), classifier
integration (`--classifier`, `--quadrant-classifier`, `--attack-classifier`),
and known-attack ablation modes. Run `./build/gbo_bench --help` for the
full list.

---

## 5. Embedding schemes

A *scheme* is a deterministic selection of DCT coefficients inside an 8×8
block, split into two regions `REG0` and `REG1` (plus a combined `ZONE0`
used by the optimizer as the search space). Schemes are defined in
[`embedding_schemes.json`](embedding_schemes.json) and loaded at startup —
**no recompilation is needed to add a new scheme**.

| Scheme id         | Vector size (ZONE0) | Notes                                        |
|-------------------|---------------------|----------------------------------------------|
| `scheme1`         | 22                  | original scheme, symmetric anti-diagonal     |
| `scheme2`         | 22                  | alternative with shifted mid-frequencies     |
| `scheme3`         | 25                  | 12+13 split, adds `[2,2]`, `[1,3]`, `[3,1]`  |
| `extended_scheme` | 25                  | 12+13 split, adds `[2,3]`, `[1,4]`, `[3,2]`  |
| `standard_scheme` | 22                  | contiguous 11+11 anti-diagonal stripe        |

The global constant `CURRENT_VEC_SIZE` tracks the ZONE0 size of the active
scheme and is used by the GBO to size all candidate vectors.

### 5.1 Adding a scheme

Append an object to `embedding_schemes.json`:

```json
"my_scheme": {
  "name": "My Scheme",
  "description": "...",
  "REG0":  [[7,0],[6,0], "..."],
  "REG1":  [[5,2],[4,2], "..."],
  "ZONE0": [[7,0],[6,0], "..."]
}
```

Then run with `--scheme my_scheme`. No code changes required.

---

## 6. Algorithm overview

1. Load the cover image and convert to grayscale.
2. Split into 8×8 blocks and forward-DCT each block.
3. For every bit of the 1024-bit binary watermark:
   1. Select the target block (index `i mod WM_SIZE`).
   2. (Optional) Use the classifier to pick the embedding scheme for this block.
   3. Initialize a population of `POP_SIZE = 30` perturbation vectors of size
      `CURRENT_VEC_SIZE` inside `[-TH, +TH]`.
   4. Run the gradient-based optimizer for `ITERATIONS = 40` generations
      against an attack-aware objective function (configurable via
      `AttackType`).
   5. Apply the best perturbation to the block's DCT coefficients and
      inverse-DCT.
4. Reconstruct the watermarked image.
5. Optionally simulate attacks (JPEG 10–90, contrast ±, salt-pepper, crop, …)
   and compute PSNR / SSIM / NCC / BER against the original watermark.

All tunable constants live in
[`GRADIENT_BASED_OPTIMIZER/src/config.h`](GRADIENT_BASED_OPTIMIZER/src/config.h):

```cpp
#define POP_SIZE   30      // population size
#define ITERATIONS 40      // optimizer iterations
#define TH         10.0    // perturbation bound for DCT coefficients
#define WM_SIZE    1024    // watermark length in bits
```

---

## 7. Quadrant-based watermarking (optional pipeline)

Instead of embedding a single watermark copy into the whole image, the
quadrant pipeline splits a 1024×1024 image into 4 quadrants and embeds a
copy into each, **each quadrant optimized against a different attack**:

| Quadrant          | Objective                  |
|-------------------|----------------------------|
| N1 (top-left)     | `AttackType::NONE`         |
| N2 (top-right)    | `AttackType::JPEG70`       |
| N3 (bottom-left)  | `AttackType::CONTRAST`     |
| N4 (bottom-right) | `AttackType::SALT_PEPPER`  |

At extraction time a classifier (`best_model_ultrahighres.pt`) predicts
which quadrant is the most reliable source for the current (possibly
attacked) image and the watermark is recovered from that quadrant only.
This mode requires libtorch and the matching model weights.

---

## 8. Library API (`libgbo`)

Build the shared or static library (`-DGBO_BUILD_SHARED=ON` /
`-DGBO_BUILD_STATIC=ON`) and link against it from your project.  The public
header is [`gbo_api.h`](GRADIENT_BASED_OPTIMIZER/src/gbo_api.h)
(installed to `include/gbo/gbo_api.h`).

### 8.1 Quick example

```cpp
#include <gbo/gbo_api.h>
#include <opencv2/opencv.hpp>

int main() {
    // 1. Initialize (loads embedding_schemes.json)
    gbo::init("embedding_schemes.json");
    gbo::setScheme("scheme1");

    // 2. Embed
    cv::Mat cover = cv::imread("cover.png", cv::IMREAD_GRAYSCALE);
    cv::Mat wm    = cv::imread("watermark.png", cv::IMREAD_GRAYSCALE);
    cv::Mat watermarked = gbo::embedWatermark(cover, wm);
    cv::imwrite("watermarked.png", watermarked);

    // 3. Attack
    cv::Mat attacked = gbo::attackJPEG(watermarked, 70);

    // 4. Extract
    cv::Mat extracted = gbo::extractWatermark(attacked);

    // 5. Evaluate
    std::cout << "PSNR: " << gbo::computePSNR(cover, watermarked) << " dB\n";
    std::cout << "BER:  " << gbo::computeBER(wm, extracted) << "\n";
}
```

Compile and link:

```bash
g++ -std=c++17 my_app.cpp -lgbo -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -o my_app
```

### 8.2 API reference

All functions live in the `gbo` namespace.

**Initialization:**

| Function | Description |
|----------|-------------|
| `bool init(path)` | Load schemes from JSON, initialize quantization tables. Call once before any other function. |
| `bool setScheme(id)` | Select the active embedding scheme by name. |
| `vector<string> availableSchemes()` | Return all scheme identifiers. |

**Watermarking:**

| Function | Description |
|----------|-------------|
| `cv::Mat embedWatermark(image, watermark)` | Embed a binary watermark into a grayscale image. Returns the watermarked image. |
| `cv::Mat extractWatermark(image)` | Extract the embedded watermark from a (possibly attacked) image. |

**Metrics:**

| Function | Description |
|----------|-------------|
| `double computeMSE(a, b)` | Mean Squared Error. |
| `double computePSNR(a, b)` | Peak Signal-to-Noise Ratio (dB). |
| `double computeSSIM(a, b)` | Structural Similarity Index. |
| `double computeNCC(a, b)` | Normalized Cross-Correlation. |
| `double computeBER(wm1, wm2)` | Bit Error Rate between two watermarks. |

**Attack simulation:**

| Function | Description |
|----------|-------------|
| `cv::Mat attackJPEG(image, quality)` | JPEG compression. |
| `cv::Mat attackBrightnessIncrease(image, value)` | Increase brightness. |
| `cv::Mat attackBrightnessDecrease(image, value)` | Decrease brightness. |
| `cv::Mat attackContrastIncrease(image, alpha)` | Increase contrast. |
| `cv::Mat attackContrastDecrease(image, alpha)` | Decrease contrast. |
| `cv::Mat attackSaltPepper(image, prob)` | Salt-and-pepper noise. |
| `cv::Mat attackMedianFilter(image, ksize)` | Median filtering. |
| `cv::Mat attackGaussianFilter(image, ksize)` | Gaussian filtering. |

### 8.3 Linking in a CMake project

```cmake
find_package(OpenCV REQUIRED)
add_executable(my_app main.cpp)
target_link_libraries(my_app /usr/local/lib/libgbo.so ${OpenCV_LIBS})
target_include_directories(my_app PRIVATE /usr/local/include)
```
