#!/bin/bash
#
# Build script for gradient_based_optimizer.
# Auto-detects libtorch at $HOME/libtorch or /tmp/libtorch
# (builds with optional PyTorch classifier integration if found).
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CMAKE_ARGS=""
for CANDIDATE in "$HOME/libtorch" "/tmp/libtorch"; do
    if [ -d "$CANDIDATE" ]; then
        echo "libtorch found at: $CANDIDATE"
        CMAKE_ARGS="-DCMAKE_PREFIX_PATH=$CANDIDATE"
        break
    fi
done

if [ -z "$CMAKE_ARGS" ]; then
    echo "libtorch not found - building without classifier integration"
fi

mkdir -p build
cd build
cmake $CMAKE_ARGS ..
make -j"$(nproc)"

echo
echo "Build finished. Executables:"
ls -1 gradient_based_optimizer classifier_example single_classifier_example 2>/dev/null || true
