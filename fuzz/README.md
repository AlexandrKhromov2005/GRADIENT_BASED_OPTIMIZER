# Fuzz targets

libFuzzer targets, built with AddressSanitizer and UndefinedBehaviorSanitizer.

| Target | Input | Checks besides "no crash, no sanitizer report, no hang" |
|--------|-------|----------------------------------------------------------|
| `fuzz_extract` | image of any type and layout, up to 96x96, optionally tiled up to 768x768 | supported images are never rejected, unsupported ones always are; output is 32x32 of 0/255; block path == whole-image path |
| `fuzz_embed` | small image + watermark, seed, thread count, base or quadrant mode | same accept/reject contract; output has the input size; same seed gives the same image with 1 and N threads; border pixels and the input image are untouched |
| `fuzz_block` | one 8x8 block, bit, attack type, scheme, seed | GBO result is a valid block and reproducible from the seed |
| `fuzz_kernels` | 8x8 block, JPEG quality, contrast gain, 64 doubles | JPEG emulation == `imencode`+`imdecode`; contrast table == `convertTo`; fast DCT == definition; rounding == `convertTo(CV_8U)`, including NaN and infinities |
| `fuzz_metrics_attacks` | two images of any type, integer and real parameter | calls on 8-bit gray images with valid parameters never fail; attacks keep size and type |

A "supported" image is what `gbo_api.h` documents: not empty, 8 bits per channel, 1, 3 or 4
channels. Image files are not fuzzed: decoding is done by OpenCV, the library only ever
sees a `cv::Mat`.

## Build and run

```bash
cmake -B build_fuzz -DCMAKE_CXX_COMPILER=clang++ -DGBO_BUILD_FUZZ=ON -DGBO_BUILD_APP=OFF
cmake --build build_fuzz -j
python3 fuzz/make_seeds.py fuzz/seeds          # valid starting inputs
mkdir -p corpus_embed
./build_fuzz/fuzz_embed corpus_embed fuzz/seeds/embed -max_total_time=3600 -timeout=120 -fork=3
```

Run from the repository root (the targets load `embedding_schemes.json`) or set
`GBO_SCHEMES=/path/to/embedding_schemes.json`. `fuzz_embed` and `fuzz_block` run the full
optimizer and manage only a few inputs per second, hence `-fork`.

ThreadSanitizer instead of ASan/UBSan: `-DGBO_FUZZ_SANITIZERS=thread`.

## Last run

2026-09-17, one hour per target, i5-11300H, ASan + UBSan: no crashes, sanitizer reports,
timeouts or failed checks.

| Target | Runs |
|--------|------|
| `fuzz_extract` | 367 416 |
| `fuzz_embed` (`-fork=3`) | 4 621 |
| `fuzz_block` (`-fork=2`) | 49 747 |
| `fuzz_kernels` | 30 525 594 |
| `fuzz_metrics_attacks` | 57 146 590 |

`fuzz_embed` under ThreadSanitizer, 10 minutes, 641 runs: no data races.

Found while writing the targets and fixed: `-value` overflow in `brightnessDecrease` for
`INT_MIN`; `embedWatermark` threw on images whose sides are not multiples of 8; images of
unsupported types were accepted without an error when smaller than one block.
