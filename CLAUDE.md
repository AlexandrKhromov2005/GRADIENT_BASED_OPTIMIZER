# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research implementation of gradient-based optimization for digital watermark embedding in images using JPEG compression domain techniques. The system embeds watermarks into 8x8 DCT blocks with robustness against JPEG compression and other attacks.

## Build Commands

### Basic Build (OpenCV only)
```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

### Full Build (with PyTorch classifier support)
```bash
mkdir -p build && cd build
cmake -DCMAKE_PREFIX_PATH=/tmp/libtorch ..
make -j$(nproc)
```

The build system automatically detects PyTorch availability and conditionally compiles classifier integration code using the `TORCH_AVAILABLE` preprocessor flag.

**Executables produced:**
- `gradient_based_optimizer` - main executable (always built)
- `classifier_example` - ensemble classifier demo (only if PyTorch found)
- `single_classifier_example` - single classifier demo (only if PyTorch found)

## Running the System

### Main Executable Modes
```bash
# Standard watermarking (10 iterations per image)
./build/gradient_based_optimizer

# Quick test mode (1 iteration)
./build/gradient_based_optimizer --test

# Generate dataset comparing scheme2 vs scheme3
./build/gradient_based_optimizer --dataset

# Dataset generation WITH classifier (requires PyTorch)
./build/gradient_based_optimizer --dataset-classifier

# Classifier integration example (requires PyTorch)
./build/gradient_based_optimizer --example

# Use specific embedding scheme
./build/gradient_based_optimizer --scheme scheme2

# Show all options
./build/gradient_based_optimizer --help
```

### Experiment Scripts
- `run_quick_experiment.sh` - Quick validation test
- `run_classifier_experiment.sh` - Full classifier experiment with interactive menu
- `run_full_metrics_experiment.sh` - Complete metrics evaluation
- `analyze_existing_results.sh` - Analyze generated results
- `clean_experiment_results.sh` - Clean up experiment outputs

## High-Level Architecture

### Core Optimization System

**GBO (Gradient-Based Optimizer)** - [gbo.h](GRADIENT_BASED_OPTIMIZER/src/gbo.h), [gbo.cpp](GRADIENT_BASED_OPTIMIZER/src/gbo.cpp)
- Population-based optimization algorithm
- Embeds single bits into 8x8 DCT blocks
- Uses `POP_SIZE` (30) individuals and `ITERATIONS` (40) defined in [config.h](GRADIENT_BASED_OPTIMIZER/src/config.h)
- Operates on coefficient vectors selected by the current embedding scheme

**Population Management** - [population.h](GRADIENT_BASED_OPTIMIZER/src/population.h), [population.cpp](GRADIENT_BASED_OPTIMIZER/src/population.cpp)
- Maintains population of candidate solutions
- Handles fitness evaluation and selection

**Launch System** - [launch.h](GRADIENT_BASED_OPTIMIZER/src/launch.h), [launch.cpp](GRADIENT_BASED_OPTIMIZER/src/launch.cpp)
- Main orchestration: reads images, processes blocks, embeds watermark, extracts and verifies
- `launch()` - standard embedding/extraction pipeline
- `launch_with_classifier()` - version using automatic scheme selection (requires TORCH_AVAILABLE)

### Embedding Schemes System

**Critical concept**: The system uses configurable "schemes" that define which DCT coefficients to use for embedding.

**EmbeddingSchemeManager** - [embedding_schemes.h](GRADIENT_BASED_OPTIMIZER/src/embedding_schemes.h), [embedding_schemes.cpp](GRADIENT_BASED_OPTIMIZER/src/embedding_schemes.cpp)
- Loads schemes from [embedding_schemes.json](embedding_schemes.json)
- Singleton pattern: `EmbeddingSchemeManager::getInstance()`
- Global `CURRENT_VEC_SIZE` variable tracks active scheme's vector size
- Each scheme defines three coefficient regions:
  - **REG0**: First set of DCT coefficients for embedding
  - **REG1**: Second set of DCT coefficients for embedding
  - **ZONE0**: Combined set (typically REG0 + REG1) used by GBO

**Available schemes** (in embedding_schemes.json):
- `scheme1` - Original scheme (22 coefficients)
- `scheme2` - Alternative scheme (22 coefficients)
- `scheme3` - Variable size scheme (25 coefficients: 12 REG0 + 13 REG1)
- `extended_scheme` - Extended version (25 coefficients)
- `standard_scheme` - Standard size (22 coefficients: 11+11)

**Key insight**: Different schemes work better for different image block characteristics (smooth vs textured). The classifier integration automatically selects the optimal scheme per block.

### PyTorch Classifier Integration (Optional)

When PyTorch is available, the system can automatically select the best embedding scheme for each 8x8 block.

**Architecture components**:

1. **EnsembleClassifier** - [ensemble_classifier.h](GRADIENT_BASED_OPTIMIZER/src/ensemble_classifier.h), [ensemble_classifier.cpp](GRADIENT_BASED_OPTIMIZER/src/ensemble_classifier.cpp)
   - Uses multiple PyTorch models to classify blocks
   - Averages predictions across models
   - Supports Test Time Augmentation (TTA)
   - Maps predictions to scheme2 (scheme_0) or scheme3 (scheme_1)

2. **SingleClassifier** - [single_classifier.h](GRADIENT_BASED_OPTIMIZER/src/single_classifier.h), [single_classifier.cpp](GRADIENT_BASED_OPTIMIZER/src/single_classifier.cpp)
   - Alternative using a single model (faster, less memory)
   - Same prediction interface as ensemble

3. **EmbeddingWithClassifier** - [embedding_with_classifier.h](GRADIENT_BASED_OPTIMIZER/src/embedding_with_classifier.h), [embedding_with_classifier.cpp](GRADIENT_BASED_OPTIMIZER/src/embedding_with_classifier.cpp)
   - High-level API wrapping classifier + embedding
   - Static methods: `initializeClassifier()`, `initializeSingleClassifier()`
   - `embedBitWithSchemeSelection()` - analyzes block, picks scheme, embeds
   - `extractBitWithSchemePrediction()` - predicts scheme used, extracts bit

**Integration with EmbeddingSchemeManager**:
- Manager stores classifier instance internally
- `selectSchemeForEmbedding()` - uses classifier to choose scheme for embedding
- `predictSchemeForExtraction()` - uses classifier to determine which scheme was used
- Priority: single classifier (if initialized) → ensemble classifier → default scheme

**Model files required**:
- `final_model_torchscript.pt` - single classifier model
- `best_scheme_classifier_torchscript.pt` - ensemble model 1
- `ensemble_model_1_torchscript.pt` - ensemble model 2

Models must be in TorchScript format (.pt), not standard PyTorch (.pth). Use `convert_pth_to_torchscript.py` to convert if needed.

### Image Processing & Attacks

**Image Processing** - [image_processing_custom.h](GRADIENT_BASED_OPTIMIZER/src/image_processing_custom.h), [image_processing_custom.cpp](GRADIENT_BASED_OPTIMIZER/src/image_processing_custom.cpp)
- DCT/IDCT transforms for 8x8 blocks
- Block extraction and reconstruction
- Image quality utilities

**JPEG Compression** - [jpeg/compression.h](GRADIENT_BASED_OPTIMIZER/src/jpeg/compression.h), [jpeg/compression.cpp](GRADIENT_BASED_OPTIMIZER/src/jpeg/compression.cpp)
- Block-level JPEG compression simulation
- Uses quantization tables from [jpeg/quantization_tables.h](GRADIENT_BASED_OPTIMIZER/src/jpeg/quantization_tables.h)
- Quality parameter controls compression strength

**Attack Simulation** - [attacks.h](GRADIENT_BASED_OPTIMIZER/src/attacks.h), [attacks.cpp](GRADIENT_BASED_OPTIMIZER/src/attacks.cpp)
- JPEG compression attacks
- Contrast adjustment attacks
- Used to test watermark robustness

### Metrics & Evaluation

**Block Metrics** - [block_metrics.h](GRADIENT_BASED_OPTIMIZER/src/block_metrics.h), [block_metrics.cpp](GRADIENT_BASED_OPTIMIZER/src/block_metrics.cpp)
- Compute fitness/quality metrics for 8x8 blocks
- Used by GBO during optimization

**Image Metrics** - [image_metrics.h](GRADIENT_BASED_OPTIMIZER/src/image_metrics.h), [image_metrics.cpp](GRADIENT_BASED_OPTIMIZER/src/image_metrics.cpp)
- PSNR, SSIM, and other full-image quality metrics
- Evaluate watermarked image quality

### Dataset Generation

**DatasetGeneration** - [dataset_generation.h](GRADIENT_BASED_OPTIMIZER/src/dataset_generation.h), [dataset_generation.cpp](GRADIENT_BASED_OPTIMIZER/src/dataset_generation.cpp)
- `generate_dataset()` - compares scheme2 vs scheme3 systematically
- `generate_dataset_with_classifier()` - uses classifier to auto-select schemes
- Creates labeled datasets for training or evaluation
- Outputs organized into directories by scheme and correctness

## Key Implementation Details

### Global Configuration
[config.h](GRADIENT_BASED_OPTIMIZER/src/config.h) defines critical constants:
- `POP_SIZE` (30) - population size for GBO
- `ITERATIONS` (40) - optimization iterations
- `TH` (10.0) - default threshold parameter
- `WM_SIZE` (1024) - watermark size in bits
- `CURRENT_VEC_SIZE` - dynamic global tracking current scheme's vector size

### Conditional Compilation Pattern
Classifier code is wrapped in `#ifdef TORCH_AVAILABLE` blocks:
- CMake sets `-DTORCH_AVAILABLE` when PyTorch is found
- Allows building without PyTorch dependency
- Main executable gracefully handles missing classifier features

### Processing Flow
1. Load image and watermark
2. Convert image to DCT domain (8x8 blocks)
3. For each bit of watermark:
   - Extract corresponding 8x8 block
   - Optionally: use classifier to select scheme
   - Run GBO to embed bit into block
4. Reconstruct image from modified blocks
5. Optional: simulate attacks (JPEG, contrast)
6. Extract watermark: reverse the process
   - Optionally: predict scheme used for each block
7. Compare extracted vs original watermark

### Random Utilities
[random_utils.h](GRADIENT_BASED_OPTIMIZER/src/random_utils.h) provides seeded random number generation used throughout optimization.

## Important File Locations

- **Main entry**: [GRADIENT_BASED_OPTIMIZER.cpp](GRADIENT_BASED_OPTIMIZER/GRADIENT_BASED_OPTIMIZER.cpp)
- **Images directory**: `images/` (input images and watermark.png)
- **Results**: `dataset/`, `dataset_classifier/` (created by dataset generation modes)
- **Scheme config**: [embedding_schemes.json](embedding_schemes.json)
- **Example integrations**:
  - [example_classifier_integration.cpp](GRADIENT_BASED_OPTIMIZER/src/example_classifier_integration.cpp)
  - [example_single_classifier_integration.cpp](GRADIENT_BASED_OPTIMIZER/src/example_single_classifier_integration.cpp)

## Documentation Files

- [README.md](README.md) - Basic project overview and Ubuntu setup
- [BUILD_INSTRUCTIONS.md](BUILD_INSTRUCTIONS.md) - Detailed build guide (Russian)
- [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) - Classifier integration architecture (Russian)
- [SINGLE_CLASSIFIER_INTEGRATION.md](SINGLE_CLASSIFIER_INTEGRATION.md) - Single classifier details (Russian)
- [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md) - How to run experiments (Russian)
- [PYTORCH_INSTALL_GUIDE.md](PYTORCH_INSTALL_GUIDE.md) - PyTorch installation
- [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md), [FINAL_EXPERIMENT_SUMMARY.md](FINAL_EXPERIMENT_SUMMARY.md) - Experiment results

## Working with the Code

### Adding a New Embedding Scheme
1. Add scheme definition to [embedding_schemes.json](embedding_schemes.json)
2. Define REG0, REG1, and ZONE0 coefficient coordinates
3. Use via `--scheme your_scheme_id` flag
4. No code changes needed (schemes are loaded dynamically)

### Modifying Optimization Parameters
Edit [config.h](GRADIENT_BASED_OPTIMIZER/src/config.h) and rebuild:
- `POP_SIZE` - increase for better quality (slower)
- `ITERATIONS` - increase for convergence (slower)
- `TH` - threshold for various calculations

### Testing Classifier Integration
```bash
# With ensemble classifier
./build/classifier_example

# With single classifier
./build/single_classifier_example
```

Both examples demonstrate automatic scheme selection and extraction.

### Understanding Scheme Selection
The classifier analyzes an 8x8 block and predicts which scheme (scheme2 or scheme3) will provide better robustness. This is based on block characteristics like texture, edges, frequency content. The prediction uses a neural network trained on block features.