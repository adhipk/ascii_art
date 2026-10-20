# Browser demo

## Shared-canvas painting and independent debugging

Painting defaults to **shared-canvas competition** at exactly **8, 16, and 32
analysis pixels**, independent of analysis resolution. All sizes propose edits
against one current image and residual, sharing one total glyph budget. This
lets coarse strokes establish tone while fine marks improve the same result.
All fit presets use this mode. Independent composition is not used normally.

**Debug · independent full-image layers** remains selectable under **Layer
fitting** for inspecting each size in isolation. In this debugging mode, each
size fits the complete original image from fresh
paper, with its own **Maximum glyphs per layer** budget, coverage,
residual, and size-specific blur objective. Layers never consume another
layer's budget or inherit another layer's painted image. Async fits are
launched together; actual concurrency depends on the backend and hardware.

After fitting, glyphs composite in **32 → 16 → 8px** paint order, over one paper
background. The default character sets are blocks at 32px, lines at 16px,
and fine marks at 8px. Empty character sets skip that layer without transferring
their budgets. A slider value of 80 caps each active layer at 80 glyphs, even
when only one layer is enabled. Preview visibility checkboxes do not change
fitting budgets. `layerBudgets` and each `layerRuns` entry's `budget` record the
actual caps; `budgetPerLayer` records the slider value.
Existing penalty
cutoffs remain in effect within each layer. The layer visibility sliders and
PNG export operate on the combined result without rerunning any fits.

**8px detail sharpening** (`pursuit_detail_sharpen`, 0–3, studio default 1,
API default 0) applies luminance unsharp masking to the 8px layer's target:
`target + amount × (luminance − GaussianBlur(luminance))`, clamped to [0,1].
The filter uses a separable `[1,2,1]/4` kernel at analysis resolution. That
independent layer uses sharp scoring when sharpening is enabled; the 16px and
32px layers retain their original targets and blur. Paper selection is based
on the original source, and final composite errors still use the original
image. Debug `layerRuns[].targetSharpen` records the applied amount. Increasing
it can emphasize edges but may introduce halos and noise; it does not make
the layer a pure detail or high-pass layer.

Replay assembles the saved layers coarse-to-fine, rather than showing the
wall-clock interleaving of concurrent fits. Its objective and raw errors are
the mean of independent-layer scores, with unassembled layers starting at
their blank-paper scores. The error map belongs to the active independent
layer. Debug state distinguishes `objectiveScope: independent-layer-mean`,
`budgetPerLayer`, individual `layerRuns`, and separately measured final
`compositeMetrics`. Independently improving layers do not guarantee an
improving final composite. Changing the fitting budget reruns the independent
layers; the separate Replay glyph slider still browses cached assembly frames.

Sequential and independent debugging modes remain available under **Layer
fitting** for comparisons. Explicit API scale lists still support legacy
sizes; the default list is always `[32,16,8]`.

## Adaptive square cells

Select **Layout → Adaptive · 8 / 16 / 32 px** to try mixed-size text. Fixed grid
remains the default. Adaptive mode uses square cells, measures the actual font
at each size, and preserves image proportions. Columns sets the number of
smallest 8-pixel cells; the fixed-mode font-size picker is disabled in adaptive
mode. Larger cells use proportionally larger glyphs centered inside each square.

An aligned quadtree starts at 32×32, splits to 16×16, then to 8×8.
At each region, every calibrated glyph competes against the optimal four-child
representation. The default objective combines summed RGB squared reconstruction error
(normalized to [0, 1], averaged over channels) with a tone term:
`4 × area × (normalized glyph density − target coverage)²`, plus
`adaptive_penalty` per nonblank glyph (default 1.0). Glyph density is normalized
by the densest measured kernel at that size. Target coverage uses source
luminance relative to paper and monochrome ink polarity, including in color
mode. This prevents photographic midtones from saturating to the densest glyph.
Python callers can set `adaptive_objective="pixels"` to omit tone calibration
for literal geometric reconstruction. Blank regions incur error but no glyph penalty.
The dictionary includes the brightness ramp and directional strokes `/`, `|`,
and `\`. Actual Canvas alpha masks are measured at every size, using the same
font, baseline, centering, and clipping as the preview and export.

This computes the exact minimum within the aligned tree, measured dictionary,
and deterministic region colors; it does not optimize free placement or rotation.
Ink is the rounded mean of the base cells' contrast-adjusted RGB colors.
Detected directional contours can use 8, 16, or 32 pixel glyphs. Contour
regions compare directional kernels using symmetric distance between detected
edge pixels and rendered glyph strokes. Missing and extra strokes both cost;
there is a one-pixel tolerance and a four-pixel distance cap. Edge costs are
weighted by 4, plus the same per-glyph penalty. Detector selection/minimum votes
are preserved while geometry is scored at full analysis resolution. Larger tiles
require consistent detected direction; uncertain regions subdivide, and 8-pixel
leaves retain the original detector glyph. Tone-only regions exclude shape-only
kernels such as `/`, `|`, and `\` so they cannot form false contour bands.
The optimum is subject to these contour constraints. Tonal-only learned-edge
emphasis does not create a separate adaptive directional edge map.
Partial outer blocks subdivide to cover the whole image without cropping.

The preview uses SVG **text elements**, with the same positioned glyphs as Canvas
PNG export. Blank cells create no SVG nodes. **Text · fixed grid** exports a
separate 8×8 text grid; PNG preserves the variable-size graphical layout.
Masks are cached per size in the browser and sent to the existing Python worker;
returned tile metrics omit these analysis-only arrays.

Synthetic regression cases verify known optima at all three sizes, zero-cost
blank paper, exact partition coverage at partial borders, decreasing glyph count
with increasing penalty, and agreement between the reported objective and a
direct RGB reconstruction in pixels mode. Real-font grayscale ramps check glyph
variety on light/dark paper in monochrome and color modes. Font-transport and SVG/PNG positioning tests also run.
The former heuristic's timing measurements do not apply to this optimizer.
Live browser/WASM performance and visual quality still require measurement.

## Reference detector

The detector menu now defaults to **Acerola · DoG + direction vote** (contours
remain off initially). Enabling it uses the reference binary-DoG/direction-vote
edge path in Python; **Minimum edge votes** controls the winning bucket count
out of 64 samples. It draws directional glyphs, unlike the optional tonal modes
described below. See the [reference audit](../docs/acerola-reference.md) for
source links, tests, corrected shader bugs and remaining rendering differences.

A basic static HTML page running the existing Python ASCII renderer inside
**Pyodide 0.29.3 (WebAssembly)** plus ONNX Runtime Web for learned edges. Images stay on the device. There is no image
upload endpoint and no server-side Python process.

From the repository root, with [Bun](https://bun.sh) installed:

```sh
bun run --cwd web start
```

Open **http://127.0.0.1:8766**. A built-in sample appears immediately and renders
automatically. Choose or drop your own image, then adjust the sidebar controls.
Changes render after a **180 ms debounce**, with one active job and only the latest
pending edit retained. Controls remain usable while rendering. The previous text preview
stays visible until its replacement grid is ready; superseded results never replace
the current preview. Downloads are enabled only when the latest result is ready.

Use **ASCII / Original** for instant comparison and **Fit / 100%** to inspect the
image. Cancel drops pending work and terminates the worker; changing a setting or
selecting Retry resumes. Contour emphasis is off by default. The Acerola detector
requires no model. The optional PiDiNet Tiny mode supplies a cached confidence
map for tonal emphasis; XDoG is another model-free tonal option. Reset restores
the controls and schedules an update.

To use another port:

```sh
PORT=8080 bun run --cwd web start
```

## Static hosting

```sh
bun run --cwd web build
```

Serve or deploy **only `web/dist/`** with any ordinary static HTTP host. No npm
installation or backend is required. Opening `index.html` using `file://` will
not work because workers and Python source fetches require HTTP(S). The build
copies Python from `src/ascii_art` without changing its implementation; rebuild
after changing Python source. The generated directory is ignored by Git.

The module worker lazily downloads Pyodide, NumPy, OpenCV and Pillow from
`https://cdn.jsdelivr.net/pyodide/v0.29.3/full/`. The initial sample render requires a network
connection and can take a minute. Subsequent jobs reuse the interpreter and
glyph cache. Browser HTTP caching may reuse downloads across visits; full
offline support is not implemented. Do not describe this as a small WASM download.

The MicroPython sandbox in
[Simon Willison's article](https://simonwillison.net/2026/Jun/6/micropython-in-a-sandbox/)
inspired this approach. Pyodide supports the renderer's compiled dependencies;
MicroPython does not provide the required CPython packages. This page runs only
its own renderer code and is not an arbitrary-code sandbox.

## Limits and behavior

- Browser-supported image inputs up to 20 MB and 40 megapixels before resampling.
- Images are resampled to a maximum edge of 1,200 pixels before worker transfer.
- Columns: 20–180. Font sizes: 8, 12, 16 or 20. Output budget: 6 million pixels.
- Transparent pixels composite against the selected output background.
- A contrasting ink color is selected for light/dark backgrounds. In color mode,
  the Python renderer draws each glyph using its cell's average source color.
- The sample is a deterministic landscape drawn locally on a canvas. It loads automatically.
- A job times out after three minutes; Cancel stops it immediately.
- Source pixels are cached inside the worker; subsequent edits transfer settings only.
- One character-grid computation produces the live text and text download. PNG
  export draws the actual font in the browser only when requested; Python does
  not rasterize preview glyphs or encode preview PNGs.
- In the PiDiNet/XDoG tonal modes, Contour strength changes only the amount of gentle tonal enhancement. Zero is
  off; low values do not trigger a large batch of line-glyph substitutions.
- Edge confidence is retained as float32, avoiding quantization of faint responses.
  Confidence selection uses fixed logarithmic thresholds with connected-contour
  hysteresis and isolated-speck rejection, rather than image-relative percentiles.

## Checks

```sh
bun run --cwd web build
bun test web
# With the Python development dependencies installed:
pytest
```

`tests/test_web_bridge.py` verifies PNG/text parity with the Python API, alpha
compositing, malformed buffers and output limits. Static tests verify MIME types,
asset availability, source parity and path handling. Queue/transport tests exercise
rapid edits, bounded concurrency, stale completion/failure suppression, cancellation,
source replacement, worker reuse and recovery after runtime loading errors.
These tests do not substitute
for an actual Pyodide browser run.

Manual browser acceptance: load the page and wait for the sample, drag Columns
rapidly, confirm that controls stay enabled and the text preview never flashes blank,
compare Original/ASCII without rendering, switch Fit/100%, download PNG/text,
toggle color/background/edges, drop a local image with alpha, replace an image
while rendering, cancel a job, retry a runtime network failure, and check a narrow
viewport.

Validation in the implementation environment: Python and build/static tests ran;
live browser/WASM validation was blocked by server-binding restrictions, a
Chromium launch failure, and unavailable CDN DNS access. The page's visual
reference is the generated single-stage editor concept (dark image stage with a
white controls sidebar); desktop/mobile
screenshot comparison could not be completed in this environment.


Interaction references: [glfx.js demo](https://evanw.github.io/glfx.js/demo/) and
[Fabric.js image filters](https://fabric5.fabricjs.com/image-filters). These
informed the direct manipulation controls around the image; this application
retains its Python/WASM renderer rather than using their rendering libraries.


## Source-derived background

The default **From image · gradient** mode samples the source's shadow/highlight
palette, limits chroma, selects light/dark paper based on median image lightness,
and interpolates a subtle vertical gradient in CIE Lab. Monochrome ink polarity
is selected automatically. This is palette extraction, not calibrated white-balance
correction. Transparent RGB is ignored during palette extraction, including
premultiplied downsampling; transparent source pixels composite against the same
paper. The gradient is baked into the exported PNG. **Solid color** retains the
manual picker. Python callers opt in with `background_mode="source"`; the Python
API still defaults to solid paper.

## Tiny learned edges

- **PiDiNet Tiny**, via **ONNX Runtime Web 1.23.2**, explicitly uses the `wasm`
  execution provider, not WebGPU or a remote inference endpoint.
- Pinned model source:
  [bdck/PiDiNet_ONNX](https://huggingface.co/bdck/PiDiNet_ONNX), revision
  `6b3f899da74e697a7d2c2e7cba10cfce057d6e55`.
- Files: `table5_pidinet_tiny.onnx` (~86.5 KB) and
  `table5_pidinet_tiny.onnx.data` (~307 KB). These are remote assets, not vendored.
  **The ~394 KB figure excludes the ONNX WASM runtime, JavaScript, and Pyodide.**
- The model adapter follows the export's RGB/ImageNet normalization and uses its
  fused probability output. Inputs have a maximum edge of 384 pixels, with
  replicated padding to a multiple of eight; probabilities are cropped back
  before ASCII conversion. No per-image min/max normalization amplifies noise.
- The model session and last source's confidence map remain cached in the worker.
  Columns, font size, contour strength, color and paper changes do not rerun inference.
  A new image reruns inference. Cancel releases the worker and its caches.
- Learned confidence controls which local contrast adjustments are eligible.
  The browser keeps the brightness glyph ramp; it does not substitute directional
  punctuation. Strength scales a bounded local contrast adjustment, while
  confidence selection stays fixed. The DoG pass is bypassed for learned maps.
  Legacy Python callers without `edge_amount` retain directional glyph behavior.
  Neural edges do not imply semantic segmentation.
- A runtime/model failure is surfaced with a suggestion to select Classic DoG;
  the application does not silently present classic results as learned output.
- Downloads come from jsDelivr and Hugging Face. Image pixels remain on-device.
- PiDiNet's [upstream license](https://github.com/hellozhuo/pidinet/blob/master/LICENSE)
  includes research-use/commercial-contact language; the community export does
  not establish a separate commercial permission.

TEED is another small candidate (58K parameters,
[original project](https://github.com/xavysp/TEED)), but a verified browser-ready
ONNX export was not available in this implementation environment. PiDiNet Tiny
was selected for its published, compact graph/weights and documented adapter;
this is not a claim that it is the absolute smallest possible edge model.

Tests validate preprocessing, source-map cache reuse, tensor disposal, output
shape/cropping, gradient properties, and rendering with supplied confidence maps.
**Actual neural inference, quality and browser latency remain unverified** because
the implementation environment cannot download or execute the browser runtime.
No browser FPS or speedup claim is made. A native baseline before integration
measured approximately 38 ms without edges and 44 ms with DoG for the supplied
image at 100 columns; that measurement does not predict WASM performance.

## Bundled image examples

The Examples thumbnails select four supplied fixtures: **Taylor**, **Webb poster**,
**Spacesuit**, and **Test image**. Selecting one loads it through the same bounded
image-decoding path as an upload and renders automatically with the current
settings. Rapid selections discard older requests. Uploading or selecting the
synthetic sample clears the example selection.

Originals live in `web/examples/` alongside lightweight WebP thumbnails. The build
copies them into `dist/examples/`, so the app does not depend on absolute paths
or access to the Pictures folder. Only thumbnails load when the page opens;
originals are fetched on demand. The originals remain byte-for-byte copies of
the supplied files. The native regression suite checks all four images using
the source-derived gradient and classic edge pipeline; learned-model inference
still needs browser verification.


## Contour control regression checks

Native renders were inspected for all four supplied examples with contours off,
strength 2, and strength 100 using the classic detector. Automated tests also
stress strengths 1–2 with a maximally permissive confidence map and check that
mean pixel difference from the unfiltered render stays below one intensity unit.
They verify that emphasis never introduces directional glyph indices, zero
strength skips detection, flat fields remain unchanged, float confidence survives
transfer into the Python adapter, and confidence selection is monotonic.
These native and synthetic-map checks do not validate real neural inference in
a live browser.


## Actual text rendering

The browser preview is real colored monospace text in the document, not an image
of pasted masks. Canvas text measurement samples glyph coverage for the browser's
own font and sends cell dimensions/densities to Python. With `text_metrics`, the
Python generator skips Pillow glyph-atlas creation and `generate_grid` returns
characters, per-cell RGB ink and gradient paper rows. The desktop Python image
API continues to support its existing mask-rendering behavior.

Fit changes the text's font size, line height and letter spacing, rather than
scaling a bitmap or composited text layer. 100% uses the chosen base font size.
The selected weight is 600; ligatures/kerning are disabled for an aligned grid.
PNG export uses Canvas `fillText` with the same font, grid and colors, normally
at 2× resolution (capped at 12 megapixels). Plain text remains downloadable.
Font appearance can vary between platforms because the default is a system
monospace family.

Colored ink still starts from each source cell's average color. When that ink is
too similar to the local paper, linear-light adjustment targets 3:1 contrast
against the cell's mean paper color. Already contrasting colors are preserved.
This is a stroke-color target, not an accessibility-conformance claim for an
ASCII image or its antialiased edges.

Tests prove the browser-grid path skips Python glyph-atlas generation, check
metric validation, verify ink contrast/preservation, and check that export draws
real text at the expected positions/colors using a canvas test double. Real
browser font rendering and layout remain unverified in this restricted session.

## Multiscale composition and iteration lab

Start normally with `bun run --cwd web start`, then open
**http://127.0.0.1:8766/?lab=1&example=taylor**. Or choose **Layout → Multiscale ·
iteration lab**. The direct lab URL selects Taylor without starting Python or
fetching a learned edge model. You can also use `example=spacesuit-patent`, another
filename stem, or an encoded full filename from the examples directory.

This is a new TypeScript worker renderer, alongside the existing fixed/adaptive
Python modes. The default painterly schedule optimizes one kernel size at a time,
from largest to smallest. All-scales mode explores all sizes on each step. Both
search free pixel positions near signed residual hotspots and 0°/90° rotations. The measured
font dictionary uses printable ASCII punctuation, straight strokes, parentheses
and brackets. Unicode curves, box-drawing glyphs and blocks are excluded.
Browser and native fixtures share `pursuit-dictionary.json`. Duplicate raster
kernels are removed.
Search keeps 1–3 competing compositions, then revisits up to roughly eight
prior placements. Larger patterns do not permanently reserve regions: later
small glyphs may overlap and correct them.

For original target `I` and current composition `R`, proposals follow the signed
residual `I - R`. Energy is mean RGB squared residual plus the chosen contour
weight times squared horizontal/vertical luminance-residual differences. Glyph
color is fitted against that pixel-and-contour objective. Each scale gets its
own candidate shortlist; ranking includes both pixel and contour residuals.
Final acceptance uses the exact energy change, including gradient pairs crossing
the edited patch boundary. Only improving edits are recorded. There is no
per-glyph penalty; the visible-glyph budget bounds the search instead.

Fitting detail is 128, 192, or 320 pixels on the longest analysis dimension.
The sampled sizes are approximately 1/4, 1/8, 1/16, 1/32, and 1/64 of that
longest dimension, with an additional 3-pixel fine kernel. Display columns only
changes preview/export resolution. Matching in normalized analysis coordinates
keeps decisions consistent under input/output resizing at the same fitting
setting, subject to resampling. This is a bounded beam heuristic, not exhaustive
optimization over continuous positions, sizes, rotations or all compositions.

Once a run finishes, **Optimization history** provides:

- An integer iteration slider, previous/next, and play/pause, including paper-only
  iteration zero and every accepted search/refinement step.
- Original alongside the selected reconstruction; seeking does not rerun fitting.
- A signed luminance residual view: red = reconstruction too dark, blue = too
  light, white = matched. Magnitude is amplified 4×. Thumbnail downsampling keeps
  the strongest signed residual in each cell, rather than canceling opposite
  errors. It is a diagnostic view, not a separate optimization target.
- Energy, pixel/contour error, visible glyph count, distinct glyphs, action, and
  per-scale candidate counts; an energy curve and explicit stopping reason.
- PNG export of the selected iteration and JSON export of the whole trace.
  Free overlapping layouts do not offer a fixed-grid text export.

Trace frames store complete best-so-far compositions and compact residuals;
competing beam hypotheses are not individually recorded. The browser renders
actual positioned SVG text and Canvas PNG using the measured font's scaled
metrics, clipping, and rotation. Small-font hinting and Unicode fallback may
make the display differ from the resampled 64px atlas used for scoring. A fixed
analysis resolution and finite budget can still miss thin contours: inspect the
residual and try finer fitting/larger budgets. A lower search energy alone does
not establish perceptual improvement against the original-image benchmarks.

## Add your own examples

Copy PNG, JPEG, WebP, GIF, AVIF, or BMP files into **`web/examples/`**. The local
server discovers them automatically; the page refreshes the catalog every five
seconds while visible, and **Refresh** requests it immediately. No server restart
or hand-edited TypeScript list is needed. Filenames are stable IDs; names become
labels. Optional `name.thumb.webp` sidecars are used as thumbnails. Without one,
the original image is loaded lazily as its thumbnail. Hidden files, thumbnail
sidecars, unsupported files, subdirectories, and symlinks are excluded.

Static builds generate `dist/examples.json` and copy the example images. Rebuild
before deploying newly added examples to a static host. Local serving reads the
source example folder, so new files are available without rebuilding.

The stylesheet is copied verbatim to `dist/style.css`, rather than relying on
Bun's version-dependent CSS output naming. Clean restart builds have been tested
with Bun 1.0.25 and 1.4.2.

## Native replay harness

The same TypeScript optimizer can run outside the browser:

```sh
uv run --no-sync python benchmarks/pursuit_fixture.py
bun benchmarks/pursuit_run.ts /tmp/glyph-pursuit-fixture.json /tmp/glyph-pursuit-taylor 128
uv run --no-sync python benchmarks/pursuit_preview.py
```

Open `/tmp/glyph-pursuit-taylor/index.html` to scrub every saved iteration next
to the original. The native fixture uses bundled Fira Code through Pillow;
its rasterization differs from the browser's system font. Use `--image` and
`--output` on the fixture script for another original; use `--fixture` and
`--run` on the preview script for another run. The CLI records JSON and CSV
energies. Offline PNGs are rebuilt from actual calibrated masks and recorded
compositions, not simulated progress images.

Automated coverage includes monotonic accepted energy, visible budgets, replay
in both directions, normalized resize consistency, every-scale exploration,
thin off-center contours, signed residuals, transparent pixels, blank input,
rotation/export parity, playback lifecycle, worker execution without network,
live example additions/removals, and stable stylesheet delivery. Browser visual
QA remains unverified when the Browser runtime has no available session.

## GPU candidate scoring and scale-aware vocabulary

Multiscale mode now defaults to **Candidate scorer → Auto · GPU when available**.
The worker uses WebGPU compute to batch the coarse ranking of kernel/position
candidates. Source pixels and alpha atlases stay resident during a run; each
batch uploads the current composition and candidate positions and reads back
scores. Final color refits and exact energy acceptance stay on the CPU.

The first GPU batch checks up to 48 evenly distributed scores against the CPU
reference. Missing adapters, shader errors, device loss, invalid/nonfinite
scores, or a failed score self-check fall back to CPU scoring. History reports
`CPU`, `WEBGPU`, or `MIXED`, with any fallback reason. **CPU · reference** bypasses
adapter creation; **WebGPU · CPU fallback** explicitly requests the same GPU path.
WebGPU requires browser/worker support and a secure origin such as localhost.
GPU floating-point ties can select different but improving compositions; every
accepted edit still passes the exact CPU energy check. Backend diagnostics,
batch count, GPU candidate count, upload/dispatch/readback wall time and score
self-check error are included in downloaded traces.

**Glyph vocabulary → Simple large · detailed small** is the default. Large
placements (at least half the largest sampled size) use ASCII strokes and
parentheses. Medium placements (at least one eighth) add brackets and more
ASCII geometry. Fine placements use the full glyph vocabulary. Every scale
is retained in the painting schedule and gets its own shortlist. Custom characters
outside the product dictionary must also be printable ASCII. **All shapes at every scale**
retains the former full search for comparison. Kernel counts per size are shown
in history and exported in the trace. Both CPU and GPU use the same policy.

The exact WGSL compiled and ran through native WebGPU/Metal on Apple M1 Pro.
For one real Taylor batch with the bundled Fira Code font, balanced fitting,
color enabled and contour weight 3:

| Vocabulary | Candidates | Warm CPU ranking | Warm GPU ranking |
| --- | ---: | ---: | ---: |
| All shapes | 11,316 | 132.1 ms | 6.6 ms |
| Scale-aware | 7,476 | 80.5 ms | 4.1 ms |

Scale-aware kernel counts in that calibrated run were 34 at 48/24 analysis
pixels, 66 at 12/6, and 82 at 3. GPU scores agreed with CPU scores within 2.3e-8.
GPU timings include current-pixel/candidate uploads and score readback, exclude
pipeline creation and CPU final verification, and are medians of five warm
batches. First GPU dispatch was about 103 ms. CPU timings are also warm medians.
These are candidate-stage measurements through native Metal, **not browser or
whole-search speedups**. No live browser session was available for visual QA.
Recorded data is in `benchmarks/latest-gpu.json`.

Reproduce with the optional native validation runtime (no app dependency added):

```sh
# Start with a fixture produced by pursuit_fixture.py.
bun benchmarks/pursuit_gpu_fixture.ts /tmp/glyph-pursuit-fixture.json /tmp/glyph-gpu-check scale-aware
uv run --no-project --with wgpu==0.32.0 python benchmarks/pursuit_gpu_validate.py --input /tmp/glyph-gpu-check
```

The fixture captures a batch from the actual optimizer and CPU reference scores.
The validator runs the same WGSL and checks every candidate, rather than only
the browser's bounded startup probes. Replace `scale-aware` with `full` to test
all shapes. Browser API plumbing, score self-checks, CPU opt-out, failure fallback,
resource cleanup, exact energy acceptance and scale policies also have automated
tests; these tests do not substitute for browser GPU performance measurement.


Iteration history supports budgets up to 2,048 glyphs. Unchanged display glyphs are
shared between snapshots (including worker structured cloning); runs above 512
use 64px residual previews to limit history memory. Every accepted step remains
scrubbable. This reduces history overhead, not the cost of scoring candidates.
The graph spans the observed error range and reports per-step error reduction;
reconstruction error is not a perceptual quality score. Flat-looking global error
can coexist with visible local contour improvements. The search has no plateau
patience cutoff: it stops at budget or when bounded proposals cannot improve.

Unicode blocks, box-drawing glyphs and stippling are excluded at every scale.
Ink overlap defaults to 100% (unrestricted layering), with an adjustable cap,
coverage-aware CPU/GPU scoring and fully hidden glyph cleanup.


### Full-run search speed

The default is **Fast · more iterations** with **1 competing composition**.
Fast search uses three residual hotspots rather than six, and exactly verifies
four finalists per scale rather than twelve. It retains every sampled scale and
all scale-eligible glyphs. If no exact improvement survives, verification expands
to twelve finalists; before stopping, the search retries the full six hotspots.
**Thorough** and 2–3 competing compositions restore the broader search at higher
cost. Fast mode can choose worse edits at a fixed glyph count; it is intended to
spend the saved time on more iterations.

Both modes reuse full-resolution CPU color fits for fine kernels, compute
candidate locations once per scale, skip sorting candidates below the shortlist,
and allocate/render image buffers only for surviving beam branches. GPU scoring
still has exact CPU verification. These optimizations do not relax the energy or
ink-overlap acceptance rules. Trace exports record `effort` and `beamWidth`.

Offline harness arguments are now `fixture output budget [fast|thorough] [beam]`:

```sh
bun benchmarks/pursuit_run.ts /tmp/glyph-pursuit-fixture.json /tmp/pursuit-fast 256 fast 1
bun benchmarks/pursuit_run.ts /tmp/glyph-pursuit-fixture.json /tmp/pursuit-thorough 64 thorough 2
```

Full-run measurements are recorded in `benchmarks/latest-pursuit-speed.json`.
These are native CPU runs; browser/GPU full-run timings remain unverified.


### Kernel-size blur curve

**Kernel blur** (0–3, default 1) derives blur levels from the same analysis-space
kernel sizes used for placement. There is no independent pixel-radius setting.
For kernel size `s` and control `b`, the pyramid level is
`L(s,b) = clamp(log2(max(1, s*b/6)), 0, 5)`. Fractional levels interpolate their
neighboring objective weights, so adjustment is continuous. Effective blur sigma
in analysis pixels is approximately `sqrt((4**L - 1)/6)`. Exported `blurCurve`
records the sizes, levels and sigmas; `blurStrength` records the control.

This uses a Gaussian-pyramid approximation: repeated separable `[1,2,1]/4`
filtering and 2× downsampling with clamped boundaries. Both target and rendered
image pass through the same linear filter, implemented by filtering their signed
residual. The objective combines pixel and contour error at all weighted levels.
15% is reserved for the sharp level; the remaining 85% is distributed across the
kernel-derived blur curve (small kernels add further sharp weight). At zero,
the original sharp objective and search path are recovered.

All candidates and iterations use one fixed combined objective, so gains at
different kernel sizes remain comparable. CPU and GPU rank candidates from
smoothed residual guidance. Finalist colors and gains use the combined objective,
including blurred stroke support outside the original bounds. Cached pyramids
and local projected patches avoid blurring the full image for each finalist.
GPU guidance is uploaded anew each batch. The graph reports multiscale energy;
raw pixel/contour errors and raw residual previews remain separate diagnostics.
Error values between different blur settings are not directly comparable.

A native CPU Taylor fixture at 64 glyphs, fast/beam 1, took 3.31s with blur 1 and
2.87s with blur 0 in single diagnostic runs. This measures cost, not perceptual
quality. `benchmarks/latest-pursuit-blur.json` records the runs. The offline runner
accepts an optional final blur argument:

```sh
bun benchmarks/pursuit_run.ts /tmp/glyph-pursuit-fixture.json /tmp/blur-on 64 fast 1 1
bun benchmarks/pursuit_run.ts /tmp/glyph-pursuit-fixture.json /tmp/blur-off 64 fast 1 0
```


### ASCII-only vocabulary

The current dictionary is restricted to printable ASCII (U+0020–U+007E).
Unicode box-drawing strokes, curves and blocks are excluded in both scale-aware
and full-vocabulary modes, including when loading older measured atlases.
Coarse scales use ASCII strokes and parentheses; finer scales add punctuation
and dense ASCII glyphs. Browser and native fixtures share this dictionary.
Older recorded GPU benchmark counts used the earlier Unicode vocabulary.


### Painterly scale passes

**Painting order → Large to small · converge each scale** is now the default.
Only the current kernel size generates candidates, on both CPU and GPU. Once
fast and broader searches find no improving addition, up to eight current-pass
placements are revisited. If refinement reveals another improving addition,
search resumes at the same size. Then the best composition becomes the
underpainting for the next smaller kernel size. Earlier, larger glyphs are kept;
smaller passes can overlap them subject to the existing ink-overlap limit.
Convergence means no improvement found by this bounded search, not a proven
optimum over all positions and glyph combinations.

The glyph budget remains a total addition budget. Exhausting it stops the
current pass with `converged:false`; it does not force an early transition to
smaller sizes. Increase the budget to allow more passes to finish. The same fixed
multiscale objective applies throughout, so energies remain comparable across
passes. ASCII restrictions and the kernel-linked blur curve still apply.

Frames record `stage` and `activeScale`; traces record `schedule` and `stages`
(size, starting/ending frame, added glyphs and convergence). The scrubber displays
the active pass and the energy graph marks pass boundaries. A pass that adds no
glyphs still appears in the trace; it does not terminate subsequent passes.
**All scales · competing each step** retains the previous placement strategy.
The offline runner accepts `coarse-to-fine` or `all` as its final optional argument.

### Relevant published work

- Aaron Hertzmann, [Painterly Rendering with Curved Brush Strokes of Multiple Sizes](https://mrl.cs.nyu.edu/publications/painterly98/), SIGGRAPH 1998. Large-brush layers followed by smaller strokes where the canvas differs from a blurred reference. Closest precedent for the size/blur schedule. Our repeat-until-no-improvement policy is an extension, not a reproduction of its exact algorithm.
- Xuemiao Xu, Linling Zhang and Tien-Tsin Wong, [Structure-based ASCII Art](https://ttwong12.github.io/papers/asciiart/asciiart.html), SIGGRAPH 2010. Matches character shapes to image line structure using alignment-insensitive shape similarity and constrained deformation. Relevant to improving our contour metric; AISS is not implemented here.
- Aaron Hertzmann, [Paint by Relaxation](https://mrl.cs.nyu.edu/publications/painterly-relaxation/), CGI 2001. Energy minimization with relaxation and search heuristics, balancing image approximation and economical stroke use.
- Songhua Liu et al., [Paint Transformer: Feed Forward Neural Painting with Stroke Prediction](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Paint_Transformer_Feed_Forward_Neural_Painting_With_Stroke_Prediction_ICCV_2021_paper.html), ICCV 2021. Predicts stroke sets in parallel through a trained model rather than an expensive iterative parameter search. Relevant to a future learned proposal system; it is not a replacement implemented by this change.

Our current renderer combines a discrete ASCII mask dictionary, bounded beam
search, a fixed blur-pyramid objective, and staged kernel sizes. It does not
claim the optimality, shape metric, brush model, or learned predictor of these
papers.


### Contour placement and broad effort

Candidates now anchor the centre of their measured ink, rather than the centre
of the square character cell. This matters for underscores, off-centre strokes
and asymmetric glyphs. Refinement anchors use the previous glyph's ink position
as well. The strongest proposal receives bounded translation refinement, first
at a small size-dependent step and then at one-pixel steps. Every translation
uses the same exact image objective and ink-overlap check. Position caches are
shared only between kernels with the same size and ink centre.

`benchmarks/pursuit_contours.ts` tests recoverable underline, vertical-stroke and
corner masks against known original positions. All three now reconstruct exactly
with one coarse glyph. `latest-pursuit-contours.json` records before/after metrics
and a native Taylor comparison. On Taylor, contour F1 increased from .434 to .457
and mean contour distance decreased from 8.57px to 7.85px; runtime increased from
1.65s to 3.07s. These are diagnostics with a fixed edge detector, not a claim that
all photographic contours are correct. The blur objective still measures image
reconstruction rather than the AISS shape metric from the related paper.

**Broad search rounds** (1–8, default 3) adds effort to the two largest sizes in
the coarse-to-fine schedule. After a search stalls, another round checks the next
six residual hotspots and revisits a different subset of up to eight strokes.
Useful additions resume at that same size. Fine stages keep one round, and all
accepted edits still improve the common objective. Exact matches and stages
without eligible kernels skip unnecessary rounds. This is additional real search,
not repeated frames or a delay before changing scale. Traces record configured
`coarseRounds`, per-stage `searchRounds`, and per-frame zero-based `searchRound`.
The global glyph budget still applies; increasing broad effort can leave less of
that budget for fine detail. All-scales scheduling ignores this extra broad effort.

The offline runner accepts a final optional broad-round count:

```sh
bun benchmarks/pursuit_run.ts /tmp/glyph-pursuit-fixture.json /tmp/broad 256 fast 1 1 coarse-to-fine 3
bun benchmarks/pursuit_contours.ts /tmp/contour-report.json
```


The broad-round comparison in `benchmarks/latest-pursuit-broad-rounds.json` uses
the same corrected placement and source at a 256-glyph budget. Three rounds
increased broadest-scale additions from 90 to 115; contour F1 rose from .457 to
.487 and mean contour distance fell from 7.85px to 6.77px. Runtime rose from
3.07s to 3.63s. Both reached every scale. These are single native CPU runs with
a fixed Canny metric, not browser or human-perception validation.

### Developer control panel

#### Soft overlap penalty

The objective also supports **Local repeat penalty** (`pursuit_repeat_penalty`,
0–5, default 0). Each unordered pair of identical non-whitespace characters costs
`min(sizeA,sizeB)² / imageArea * max(0,1-distance/(sizeA+sizeB))²`, where distance
is measured between glyph centers. Rotation and color do not exempt a repeat.
Nearby pairs cost more; pairs at least two average glyph sizes apart cost zero.
Dense runs accumulate multiple pair costs. The term is invariant to uniformly
scaling the image and glyphs together.

Total energy is reconstruction energy plus weighted overlap and repeat errors.
Repeat cost participates in candidate ranking, exact acceptance, refinement,
and hidden-glyph cleanup. Traces expose `repeatPenalty` and per-frame
`repeatError`; Inspect displays the weighted cost. Positive repeat weights use
CPU acceptance with optional GPU candidate scoring, bypassing the resident GPU
loop, which does not yet implement this term. Studio defaults and fit presets
enable overlap at 0.1 and repetition at 0.2, with
`pursuit_penalty_min_size: 24`. Glyphs below 24 analysis pixels are exempt from
both soft penalties, including fine strokes painted over coarse blocks.
Overlap costs count only coarse ink against other coarse ink; repeat costs
count only pairs whose members both meet the cutoff. The hard overlap cap
still applies to every size. Omitted API weights and cutoff remain zero.

**Overlap penalty** is independent of the hard **Ink overlap limit**. The cap
stays at 100% by default; penalty weight is 0 by default. A positive weight adds
an area-normalized cost for ink painted over existing ink, so useful crossings
remain possible while repeated layering has to earn its place.

For glyph opacities `a₁…aₖ` at a pixel, combined coverage is
`C = 1 - ∏(1-aᵢ)`. The unweighted repeated-ink error is the image mean of
`sum(aᵢ) - C`. Adding opacity `a` on existing coverage `C` therefore costs
`weight * a * C / imageArea`. Whitespace and bounding-box intersections with no
ink are free. The term is independent of glyph color and of the order of the
same set of layers.

The total objective is the existing sharp/multiscale image objective plus this
weighted repeated-ink error. CPU/GPU ranking, exact acceptance, refinements and
CPU hidden-glyph cleanup use this objective. Resident GPU statistics retain the
same term when rebuilding layers. Inspect shows overlap cost separately from
raw pixel/contour errors; traces and copied debug artifacts record the weight
and per-frame `overlapError`. This is a preference, not a guaranteed upper bound
on overlap. Editing it waits for Run fit and invalidates cached-budget reuse.

#### Per-scale character sets

Under **Characters at each scale**, edit the field for each analysis size, then
press **Run fit**. Fields initially show the selected starting vocabulary. An
edited field overrides that profile for its scale, including coarse scales.
Any printable ASCII character is supported: letters outside the bundled default
dictionary are measured into the browser atlas before fitting. Duplicate
characters are ignored. An empty field skips that scale; disabling every scale
reports an error rather than silently restoring defaults. Rotated kernels are
still generated automatically.

The default **Blocky → detailed** vocabulary uses `#%@O0[]` for the two broadest
scales, a larger set of strokes/outlines at middle sizes, and all 94 visible
printable ASCII characters at the finest sizes. **Strokes → detailed** retains
the earlier geometric vocabulary; **Same bundled set at every scale** uses the
original bundled dictionary everywhere. Larger sets increase candidate work.

Changing Detail rebuilds the scale list; edits are retained for matching pixel
sizes. Presets retain these edits. **Reset scale sets** clears overrides, and
**Reset settings** resets both the controls and character overrides. CPU search,
GPU batch scoring and the resident GPU plan share the same filtered kernel set.
The completed trace records actual `charactersByScale` and `kernelCounts`; the
panel shows the last run's effective character counts separately from pending
edits. Changing sets invalidates budget-history reuse until a new run completes.

#### Copy a debug artifact

Replay includes a **Visible scale range** with Finest and Coarsest sliders.
Both snap to the saved run's analysis-size checkpoints. Lower Coarsest to hide
broad blocks, or raise Finest to hide small marks. Click a size checkpoint to
isolate that layer; **Show all scales** restores the full range. These controls
share visibility with the per-scale checkboxes and affect both the preview and
PNG export without changing the fit, glyph-budget history, or paint order.
Checkpoint tooltips show the layer's glyph count at the selected replay frame.

For automatic local instrumentation, run `bun web/instrumentation-server.ts`
from the repository root alongside the app. The collector listens on
`127.0.0.1:8767`. After rebuilding and reloading the app, local browser sessions
automatically publish debounced compact state on render completion, control
changes, and replay selection, plus render progress and error events.

Read `http://127.0.0.1:8767/state` for the latest state and
`http://127.0.0.1:8767/events` for the latest 200 events. The same snapshots are
saved in git-ignored `web/.debug/state.json` and `web/.debug/events.json` for
agent inspection. State includes a browser session ID and server receipt time.
Images and raw pixel arrays are excluded. The collector accepts HTTP localhost
origins; the browser does not send telemetry from hosted deployments. A missing
collector does not block rendering. Files survive shutdown; HTTP state starts
fresh when the collector restarts.

**Copy debug state** copies compact JSON for pasting alongside a screenshot.
It includes accepted run settings separately from pending controls, selected
iteration and objective terms, weighted overlap/repeat costs, backend details,
glyph counts by analysis scale and character, and layer visibility. It omits
embedded images, font masks, pixel arrays, individual placements, and replay
frame arrays. This is the preferred way to share state for diagnosis.

**Copy full artifact** copies the accepted original image, selected replay
output and configuration. Rich paste contains both images plus JSON; plain-text
paste is self-contained `ascii-debug-v1` JSON with embedded PNG data URLs. Paste
the plain text into a file to retain a debug artifact. The button requires a
completed render and browser clipboard access on localhost or HTTPS.

`runSettings` contains the exact fitted settings and measured glyph atlas.
`pendingControls` records current edits separately, so copying while editing
does not mislabel the previous output. The payload also includes selected-frame
geometry, colors, residual, trace summary, per-scale sets and browser metadata.
Other history frames are omitted to bound clipboard size; use Download trace
JSON for the full history. Copying does not refit or upload anything.

The panel follows **Fit → Replay → Inspect**. Test images remain at the top.
Fit has Quick experiment, Balanced and Detail study presets; all selected values
remain visible and editable. Changing settings marks a run pending. Press
**Run fit** to compute it; the previous result remains available while editing.
Capacity/resolution, scoring and search controls have separate groups explaining
their effects. The run summary distinguishes analysis resolution from output
geometry. Reset restores defaults and waits for Run fit.

Replay uses one glyph-count slider and one Play/Pause control. Inspect keeps the
full recorded-step slider, error map, error curve and trace export, without a
second playback control. Its scale table reports accepted additions/refinements,
rounds used and whether a scale settled or hit the shared budget. Rejected
proposals are not counted as history steps, and bounded-search convergence is
not a proof of the global optimum.

### Replay by glyph budget

The Replay section includes a **Glyphs** slider and
**Play** button. They replay stored compositions without running
the optimizer. Counts are sorted even when cleanup removes glyphs; repeated
counts use the lowest-energy frame (the latest frame breaks ties). A missing
count selects the closest available count below the requested budget. The
iteration slider remains available for inspecting every refinement.

Changing the main **Glyph budget** control also reuses the completed trace when
the source and all other settings match and the requested budget is within that
run's computed budget. Increasing past the computed budget or changing a fitting
setting marks a new run pending until Run fit is pressed. Cached replay is a selection from the original
trajectory, not an independently optimized run at the smaller budget. PNG export
uses the selected composition. Scrubbing pauses playback, and the loop cycles
through the available counts until paused or another render starts.

### Unicode character presets and cached scale layers

Each scale accepts Unicode graphemes, including emoji sequences and combining marks.
Choose Blocks, Lines & corners, Curves, Fine detail, or ASCII, then edit the set.
The default progression uses blocks at large scales, lines in the middle, and fine
marks at the smallest scales. Reset restores the suggested set for that scale.
Load/Save `.txt` uses UTF-8 text: the filename names the preset and its contents
contain the characters. Loading ignores line breaks and deduplicates whole glyphs.
Loaded presets remain available during the current session. Keep the text file to
reuse them after reloading. Empty character sets skip a scale during fitting.

The optional solid square brush adds an exact filled-square mask at every enabled
(nonempty) scale. All glyph masks, including fallback fonts and emoji silhouettes,
are measured once and shared by CPU/GPU fitting, previews, and PNG export.

Visible toggles hide/show the cached glyphs of a scale at the selected history
step. They start no fitting jobs, preserve original paint order across scales,
and apply to PNG exports. Energy and residuals still describe the complete fit.
Debug artifacts include layer visibility and separate per-scale glyph lists with
original paint indices, alongside the complete fitted frame and measured atlas.
