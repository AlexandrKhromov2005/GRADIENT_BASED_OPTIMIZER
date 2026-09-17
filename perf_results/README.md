# Performance / equivalence evidence

All files are raw `gbo_perf` output (see `tools/perf_bench.cpp`), Intel Core i5-11300H,
`scheme1`, 32x32 watermark. BER counts a voting tie as half an error.

| File | What |
|------|------|
| `baseline_base.txt`, `baseline_quad.txt` | Original implementation (commit `f38f164`), seeds 1..5 / 1..3, `lenna` 512x512 (base) and 1024x1024 (quadrant). |
| `seq_exact_base.txt`, `seq_exact_quad.txt` | Optimized single-thread code with the original single random stream (commit `27f7abf`). Hashes, PSNR, SSIM and every BER are identical to the baseline files - only the time differs (24.7 s -> 4.7 s, 158.8 s -> 22.3 s). |
| `validation/orig_*` | Original random stream (bit-identical to the original implementation), 8 test images, seeds 100..109 (base) / 100..104 (quadrant). |
| `validation/new_*` | Final multi-threaded code with per-block random streams, same images and seeds. |

Comparing `validation/orig_*` with `validation/new_*` (80 + 40 embeddings per variant):
every metric differs by far less than its run-to-run standard deviation (largest Welch
|t| = 0.77 over 28 metrics), i.e. the per-block random streams change nothing but the
particular random numbers. `python3 perf_results/compare.py` prints the per-metric table.

Reproduce: `gbo_perf --mode base|quad --image <png> --seed <s> --repeat <n> [--threads 1]`.
