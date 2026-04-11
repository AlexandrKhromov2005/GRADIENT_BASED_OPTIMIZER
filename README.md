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
├── CMakeLists.txt                     # top-level build (OpenCV required, libtorch optional)
├── build.sh                           # convenience wrapper for cmake+make
├── embedding_schemes.json             # JSON definitions of the DCT coefficient sets
├── images/                            # canonical test images + watermarks
│   ├── airplane.png  baboon.png  boat.png  bridge.png
│   ├── earth_from_space.png  lake.png  lenna.png  pepper.png
│   ├── watermark.png                  # primary 32×32 binary watermark
│   └── watermark_32x32.png            # alias used by some run modes
└── GRADIENT_BASED_OPTIMIZER/
    ├── GRADIENT_BASED_OPTIMIZER.cpp   # CLI entry point
    └── src/
        ├── gbo.{h,cpp}                # gradient-based optimizer core
        ├── population.{h,cpp}         # population, fitness, attack-aware OF
        ├── launch.{h,cpp}             # experiment orchestration
        ├── embedding_schemes.{h,cpp}  # scheme manager / JSON loader
        ├── dataset_generation.{h,cpp} # dataset generation for classifier training
        ├── image_processing_custom.*  # 8×8 block split / DCT / reconstruction
        ├── image_metrics.*            # PSNR / SSIM / NCC / BER
        ├── block_metrics.*            # per-block fitness helpers
        ├── attacks.{h,cpp}            # JPEG, contrast, salt-pepper, crop, etc.
        ├── jpeg/                      # block-level JPEG compression + quant. tables
        ├── random_utils.*             # seeded RNG
        ├── ensemble_classifier.*      # optional: multi-model scheme classifier
        ├── single_classifier.*        # optional: single-model scheme classifier
        ├── embedding_with_classifier.*# optional: classifier ↔ embedding glue
        ├── quadrant_classifier.*      # optional: quadrant-level classifier
        ├── quadrant_embedding.*       # optional: 4-quadrant embedding pipeline
        ├── attack_type_classifier.*   # optional: attack-type classifier
        ├── attack_type_embedding.*    # optional: attack-aware embedding pipeline
        └── example_*.cpp              # standalone demos linked as separate binaries
```

Everything under `build/`, generated datasets, trained model weights (`*.pt`),
archives and auxiliary images is excluded from the repository via
`.gitignore`.

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

### 3.1 Minimal build (OpenCV only)

```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

Produces a single executable `build/gradient_based_optimizer`. Classifier
modes are compiled out (they return an error at runtime).

### 3.2 Full build (with libtorch)

```bash
mkdir -p build && cd build
cmake -DCMAKE_PREFIX_PATH=$HOME/libtorch ..
make -j$(nproc)
```

Produces three executables:

| Executable                   | Purpose                                               |
|------------------------------|-------------------------------------------------------|
| `gradient_based_optimizer`   | main CLI with all run modes                           |
| `classifier_example`         | minimal demo of the ensemble scheme classifier API    |
| `single_classifier_example`  | minimal demo of the single-model classifier API       |

`TORCH_AVAILABLE` is defined automatically by CMake when libtorch is found.

### 3.3 Using the convenience script

```bash
./build.sh
```

This auto-detects libtorch at `~/libtorch` or `/tmp/libtorch` and runs `cmake`
then `make`.

---

## 4. Running experiments

**Important:** every run mode resolves input paths relative to the current
working directory. Always launch from the repository root so that
`embedding_schemes.json` and the `images/` folder are found.

```bash
./build/gradient_based_optimizer --help
```

### 4.1 Main experiment (baseline)

Embeds and extracts the watermark for each of the 8 canonical test images and
evaluates robustness against a fixed attack suite (JPEG quality sweep,
contrast, salt & pepper noise, cropping, …).

```bash
# 10 iterations per image (default)
./build/gradient_based_optimizer

# Single iteration — smoke test
./build/gradient_based_optimizer --test

# Pick a specific embedding scheme
./build/gradient_based_optimizer --scheme scheme2
./build/gradient_based_optimizer --scheme scheme3 --test
```

Per-image metrics (MSE / PSNR / SSIM / NCC / BER, min/avg/max) are written to
`results_<image>.txt` in the working directory.

### 4.2 Dataset generation (scheme2 vs scheme3)

Systematically compares the two best-performing schemes and labels each 8×8
block by which scheme wins. Used to produce training data for the scheme
classifier.

```bash
./build/gradient_based_optimizer --dataset              # default tau_max = 10
./build/gradient_based_optimizer --dataset --tau-max 5
```

Outputs are placed in `dataset/`.

### 4.3 Classifier-based modes (require libtorch + models)

| Flag                      | Required model file                           |
|---------------------------|-----------------------------------------------|
| `--classifier`            | `final_model_torchscript.pt`                  |
| `--dataset-classifier`    | `final_model_torchscript.pt`                  |
| `--example`               | `final_model_torchscript.pt`                  |
| `--quadrant-classifier`   | `best_model_ultrahighres.pt` + 1024×1024 imgs |
| `--quadrant-dataset`      | 1024×1024 input images                        |
| `--attack-classifier`     | `model_torchscript.pt` + 1024×1024 imgs       |
| `--attack-dataset`        | 1024×1024 input images                        |

Place the model weights next to the executable working directory before
running. The weights are not bundled here due to size; reach out to the
authors of the paper for the files used in the published experiments, or
retrain them from the datasets produced by the `--dataset-*` modes.

### 4.4 Known-attack modes (no PyTorch needed)

```bash
./build/gradient_based_optimizer --known-attack           # 512×512 images
./build/gradient_based_optimizer --known-attack-1024      # 1024×1024 images, voting
```

These modes embed with `AttackType::NONE` and use an attack-aware extraction
strategy — useful for ablation studies.

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

## 8. Reusing the algorithm in your own code

The optimizer is decoupled from the CLI. A minimal embedding pipeline looks
like:

```cpp
#include "launch.h"
#include "embedding_schemes.h"

int main() {
    auto& manager = EmbeddingSchemeManager::getInstance();
    manager.loadSchemes();                 // reads embedding_schemes.json
    manager.setCurrentScheme("scheme1");

    launch("cover.png", "watermarked.png",
           "watermark.png", "extracted.png",
           /*iterations=*/10);
    return 0;
}
```

The two headers you typically need are
[`GRADIENT_BASED_OPTIMIZER/src/gbo.h`](GRADIENT_BASED_OPTIMIZER/src/gbo.h)
(to run the optimizer on a single 8×8 block with a given bit and attack
type) and
[`GRADIENT_BASED_OPTIMIZER/src/embedding_schemes.h`](GRADIENT_BASED_OPTIMIZER/src/embedding_schemes.h)
(to switch coefficient selections at runtime).

---

## 9. License

TBD — please contact the authors before redistributing.
