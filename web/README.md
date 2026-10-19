# Browser demo

## Adaptive square cells

Select **Layout → Adaptive · 8 / 16 / 32 px** to try mixed-size text. Fixed grid
remains the default. Adaptive mode uses square cells, measures the actual font
at each size, and preserves image proportions. Columns sets the number of
smallest 8-pixel cells; the fixed-mode font-size picker is disabled in adaptive
mode. Larger cells use proportionally larger glyphs centered inside each square.

An aligned quadtree starts at 32×32, splits to 16×16, then to 8×8. Tonal variance
and gradient structure-tensor strength/coherence govern splitting. These signals
operate even with Contour emphasis disabled. Selected contour glyphs also prevent
merging. Integral images make region statistics constant-time; leaf cells skip
unnecessary scoring. Partial outer blocks subdivide to cover the whole image.
The thresholds are conservative heuristics, not a proof of optimal glyph size
or a full reconstruction-error optimization.

The preview uses SVG **text elements**, not bitmap masks, with the same measured
font sizes, baselines, cell clipping and colors as Canvas PNG export. Blank cells
do not create SVG nodes. **Text · fixed grid** exports a separate 8×8 text grid;
a plain text file cannot retain mixed font sizes. PNG preserves the adaptive
layout. This is an extension, not part of the fixed-tile Acerola reference.

Native Python timings at 100 base columns, contours off, source colors/gradient,
using real Canvas font metrics, warm runs, median of three:

| Example | Fixed 8×8 | Adaptive | Fixed cells | Adaptive glyphs |
| --- | ---: | ---: | ---: | ---: |
| Taylor | 19.5 ms | 60.6 ms | 10,100 | 4,781 |
| Webb poster | 22.3 ms | 82.6 ms | 16,000 | 8,728 |
| Spacesuit | 21.4 ms | 67.8 ms | 13,100 | 4,346 |
| Test | 15.7 ms | 40.4 ms | 6,500 | 701 |

These measure Python output construction, excluding DOM layout, downloads,
worker transfer and PNG export. WASM/browser performance remains unmeasured.
Adaptive processing adds overhead despite reducing glyph count. Size decisions
are currently recomputed per render; font measurements are cached. Work stays
in the existing worker with the existing debounce and newest-result queue.

Tests cover exact nonoverlapping coverage, partial borders, all three sizes,
fine texture and gradients, retained contours, metric validation, alpha through
the browser adapter, and matching SVG/PNG coordinates and clipping. PNG exports
were also generated for all four examples using real native Canvas fonts.
Live browser layout could not be verified because no browser was available.

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
