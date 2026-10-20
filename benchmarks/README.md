# Adaptive quality benchmark

Run from the repository root:

```sh
PYTHONPATH=src uv run --no-sync python benchmarks/adaptive_quality.py
# Include another exact source image:
PYTHONPATH=src uv run --no-sync python benchmarks/adaptive_quality.py --image /path/to/original.png
```

The baseline is the committed heuristic renderer at `95dd352`; the candidate is
the working-tree adaptive renderer. This is **not** a comparison to the exact
intermediate renderer behind a user screenshot, which was never committed.
Both implementations use the same source pixels, 80 columns, measured Fira Code
masks, source-derived paper, monochrome ink, and Acerola settings. Contours are
tested both off and on. The four bundled image files are read directly. The
demo landscape uses the same geometry/colors as `web/app.ts`, rasterized in
Pillow, so its boundary pixels differ from Canvas. Five synthetic cases cover
flat fields, a gradient, straight lines, corners/diagonals, and a circle.

Outputs go to `/tmp/ascii-adaptive-benchmark`: JSON, CSV, and PNG panels ordered
**original | committed baseline | current**. Source and font hashes, candidate
source hash, full baseline commit, dimensions, and library versions are recorded.
`latest.json` and `latest.csv` retain the recorded run in this directory.

## Metrics

All reference measurements come from the **original image**, resized with area
resampling to the output dimensions. No candidate is used as ground truth.

- Primary contours: Canny edges on original and complete rendered image, with
  sigma=1 Gaussian blur and thresholds 50/100. Precision/recall use a fixed
  two-output-pixel tolerance. F1 balances both. False contour rate is the fraction
  of rendered edge pixels beyond that tolerance. Mean position error is the
  symmetric average nearest-edge distance; it is null if exactly one mask is
  empty, and zero if both are empty.
- `directional_*`: the same comparison using only rasterized `-`, `/`, `|`, and
  backslash glyph masks (alpha above 0.15). This isolates directional glyphs but
  excludes artifacts from other characters; it is not the primary quality score.
- Tone MAE: mean absolute luminance error in [0,1], after sigma=8 blur of both
  source and output. Multiply by 100 for percentage points of luminance range.
- Glyphs: visible/nonblank glyph count; sizes include all partition tiles.
  Glyph histograms count visible characters. Ink coverage is summed glyph alpha
  divided by output area, independent of ink color.
- Milliseconds: one native render, excluding calibration and quality scoring.
  These are diagnostic timings, not a controlled performance benchmark.

Empty/empty contour sets score perfect precision/recall; empty output on a source
with contours scores zero recall; invented contours on a blank source score zero
precision. Tests cover these conventions and exact distance/tolerance behavior.

There is no aggregate "quality" score: compactness, contour accuracy and tone can
conflict. Canny is an imperfect reference for photographs and these scores are
threshold-, resolution-, and font-dependent. Glyph texture can count as false
edges even when it is visually acceptable ASCII shading. Inspect the panels
alongside the metrics. Browser font rendering and WASM performance are untested.

## GPU candidate-batch validation

`pursuit_gpu_fixture.ts` captures an actual batch and warm CPU ranking timings.
`pursuit_gpu_validate.py` runs its exact WGSL through optional native `wgpu`,
checks every GPU score against the reference, and measures warm upload/dispatch/
readback time. Usage and limits are documented in `web/README.md`. The recorded
Apple M1 Pro full/scale-aware comparison is `latest-gpu.json`; it is not an
end-to-end browser speedup.


## Complete optimizer latency

`pursuit_speed.ts` runs the full CPU optimizer sequentially with thorough/beam 2,
fast/beam 1, and fast/beam 1 at a larger budget. It records elapsed time, actual
accepted glyphs, objective terms, diversity and stopping reason, without PNG/JSON
snapshot-export time. Example:

```sh
bun benchmarks/pursuit_speed.ts /tmp/glyph-pursuit-fixture.json /tmp/speed.json 64 256
```

`latest-pursuit-speed.json` records diagnostic Taylor and spacesuit measurements.
Fast search trades fixed-glyph-count search breadth for runtime. Compare error
and actual accepted glyph count as well as elapsed time. Native CPU timings do
not measure interactive browser latency or GPU full-run latency. Each recorded
configuration was measured once; thermal/runtime noise may affect results.


## Sequential scale passes

`latest-pursuit-stages.json` compares the same ASCII-only Taylor fixture under
`coarse-to-fine` and `all`, with budget 256, blur 1, fast effort and beam 1.
Each includes stage frame ranges and bounded convergence status. Different
schedules can converge to different reconstructions; compare objective and
actual glyph count alongside speed. These are single native CPU diagnostics.

```sh
bun benchmarks/pursuit_run.ts /tmp/glyph-pursuit-fixture.json /tmp/staged 256 fast 1 1 coarse-to-fine
bun benchmarks/pursuit_run.ts /tmp/glyph-pursuit-fixture.json /tmp/all 256 fast 1 1 all
```


## Glyph contour placement

`pursuit_contours.ts` uses known mask/position pairs for asymmetric strokes and a
corner, rasterizes the actual returned composition, and measures contour F1 and
symmetric contour distance. `latest-pursuit-contours.json` includes before/after
results and the separate Taylor/Canny check. Synthetic exact-match tests do not
establish correctness for arbitrary source contours. `pursuit_preview.py
--last-only` generates just the final frame for fast inspection and scoring.
