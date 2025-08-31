# Gradient Based Optimizer for Watermark Embedding

This is a research project that implements a gradient-based optimization algorithm for digital watermark embedding in images using JPEG compression domain techniques.

## Building and Running on Ubuntu

### Prerequisites
- CMake 3.16 or later
- OpenCV 4.x
- GCC with C++17 support

### Installation of Dependencies
```bash
# Install OpenCV
sudo apt update
sudo apt install libopencv-dev cmake build-essential

# Verify OpenCV installation
pkg-config --modversion opencv4
```

### Building
```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```

### Running
```bash
./gradient_based_optimizer
```

### Project Structure
- `src/` - Source code files
- `images/` - Directory for input/output images
- Main algorithm components:
  - `gbo.cpp/h` - Gradient-based optimizer implementation  
  - `image_processing_custom.cpp/h` - Custom image processing functions
  - `attacks.cpp/h` - Attack simulation functions
  - `jpeg/compression.cpp/h` - JPEG compression utilities

### Input Images
Place your test images in the `images/` directory:
- Input images: airplane.png, baboon.png, boat.png, bridge.png, earth_from_space.png, lake.png, lenna.png, pepper.png
- Watermark: watermark.png

The program will generate watermarked images with "new_" prefix and extracted watermarks with "_wm" suffix.