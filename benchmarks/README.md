# Fixed-grid pruning proof of concept

This standalone experiment compares exhaustive squared pixel error against early
pixel exit and a safe lower-bound cascade. It does not change the application.

## Reproduce

Requires Bun and Python with Pillow built with RAQM (for actual OpenType shaping).
Supply a local ligature-capable font; the measured run used Fira Code Nerd Font
Regular at 16 px. No font is redistributed.

```sh
python benchmarks/pruning-fixtures.py --font /path/to/FiraCode-Regular.ttf --output /tmp/pruning-fixtures.json
bun test benchmarks/pruning.test.ts
bun benchmarks/pruning.ts /tmp/pruning-fixtures.json > benchmarks/pruning-results.json
```

## Design

- Rasterize all 324 two-character sequences from an 18-character alphabet, with
  `liga` and `calt` enabled. 58 sequences change when these features are disabled;
  this measures shaping changes, not necessarily single-glyph substitutions.
- Keep both shaped pairs and ordinary pairs. Exact-mask deduplication removed
  zero candidates in this font/size. Each candidate occupies the same 20×24 patch.
- Precompute block sums divided by square root of block area for 1×1, 2×2 and
  4×4 partitions. Squared descriptor distance lower-bounds pixel squared error
  by Cauchy–Schwarz. Each stage is tested independently, never added to another.
- Fully score the candidate closest in average coverage to establish an upper
  bound, then reject candidates whose lower bound exceeds it. Stop full pixel
  scoring when its nonnegative partial error exceeds the current best.
- Preserve original candidate-index tie breaking. Use conservative floating
  tolerance in bound comparisons; full scores use integer-valued pixel errors.

## Measured results

Apple arm64, Bun 1.4.2. Median of seven warmed runs per group. Matching times
include target descriptors, seed selection and counters; exclude font loading,
fixture decoding, mask rasterization and atlas preparation. No GPU is used.
Sources are repository examples, resized to at most 640 px. Sample 300 patches
per source with seed 42. Convert grayscale to `255 - luminance`, representing
dark-ink coverage on white paper. No contrast stretching or tone calibration.

| Patch group | Culled before pixel scoring | Exhaustive ms | Early exit only ms | Cascade + early exit ms | Speedup |
|---|---:|---:|---:|---:|---:|
| test.png | 97.3% | 47.62 | 16.62 | 3.33 | 14.31× |
| taylor.png | 64.0% | 46.69 | 41.48 | 18.76 | 2.49× |
| webb-poster.webp | 47.8% | 46.75 | 30.37 | 24.50 | 1.91× |
| Exact atlas masks | 80.5% | 46.61 | 18.01 | 7.97 | 5.85× |
| Uniform patches (64) | 52.6% | 9.82 | 7.91 | 4.72 | 2.08× |
| Random noise | 0.0% | 46.58 | 44.01 | 48.45 | 0.96× |

Across the 900 image patches: **69.7%** of candidates rejected before pixel
scoring and **3.03×** aggregate matching speedup. Across all 1,564 patches:
**zero winner or error mismatches**, including the early-exit-only comparison.
Tests additionally check lower-bound validity on uneven partitions and exact
tie handling on deterministic random masks.

The earlier conversational 80–95% estimate is not a reliable average for these
images. The noisy case is slower than exhaustive matching despite many early
exits: most exits happen near the end, after descriptor overhead has been paid.

## Limits and next decision

This validates pruning for one font, size, dictionary and literal grayscale
squared-error objective. It is not an end-to-end renderer benchmark or visual
quality evaluation. It does not implement mixed-span row composition, color,
contour costs, or context-sensitive shaping across candidate boundaries. Atlas
preparation must be amortized across many patches. A vectorized/GPU exhaustive
baseline may change the speed comparison substantially.

Proceed with the bound cascade as an optional matcher implementation, keeping
the exhaustive path for validation. Benchmark larger dictionaries, representative
images and actual renderer tone/color objectives before production integration.
Measure descriptor overhead before adding more filters; do not assume high
cull percentages guarantee a speedup.
