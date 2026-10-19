# ASCII Art Library

A CPU image-to-ASCII renderer inspired by Acerola's ASCII and Difference of
Gaussians videos. The Python API accepts OpenCV BGR/BGRA or grayscale arrays,
as uint8 or floating point in [0, 1]. Image output is BGR uint8.

For the reference edge pipeline use `edge_method="acerola"` and
`edge_min_pixels=8`: binary DoG → per-pixel directions → 8×8 direction voting →
edge glyph substitution. See the [source audit and remaining differences](docs/acerola-reference.md).
The default tensor pipeline and browser tonal-emphasis modes are adaptations,
not faithful reproductions of the video.

```sh
pip install -e .
adhipk-ascii-art input.jpg --columns 120 -o output.png
adhipk-ascii-art input.jpg --columns 100 --text -o output.txt
adhipk-ascii-art input.jpg --color --background '#101010' --foreground '#ffffff' -o color.png
```

```python
import cv2
from ascii_art import SpriteASCIIGenerator, DoGFilter

image = cv2.imread("input.jpg")
renderer = SpriteASCIIGenerator({
    "columns": 100,
    "font_size": 12,
    "use_edge": True,
    "edge_threshold": 70,
    # Optional: "font_path": "/absolute/path/to/monospace.ttf"
})
cv2.imwrite("output.png", renderer.generate_ascii(image))
print(renderer.generate_text(image))
cv2.imwrite("lines.png", DoGFilter().flow_dog(image))
```

## Rendering pipeline

1. Measure font cells and rasterize an antialiased glyph atlas. The default is
   Pillow's bundled font, so installed packages work from any directory without
   separately downloaded font files. Supply a monospace font for a specific look.
2. Resample to whole cells while preserving image proportions. `columns` sets
   the character count; otherwise it follows the source width and font metrics.
   Partial source cells are resampled rather than discarded, including tiny inputs.
3. Match cell luminance to glyphs sorted by measured ink coverage. Coverage is
   normalized across the chosen ramp to retain contrast; output tones are stylized,
   rather than photometrically exact. Foreground/background polarity is respected.
4. Compute floating-point extended DoG and Scharr derivatives. Pool gradient
   structure tensors per cell to estimate edge strength, tangent and coherence.
   Strong coherent edges become `-`, `/`, `|` or `\`. No statistical mode or
   ordinary average is applied to circular angles.
5. The Python image API composites its atlas in one vectorized operation. The
   browser requests a text grid and draws real colored text instead. Cell colors
   start from source averages and are adjusted when needed for paper contrast.
   `generate_text` uses the same character grid.

The atlas uses a bounded memory cache keyed by font, size and characters. The
renderer never writes caches into the package and never modifies input arrays.
There are no manually managed worker threads; OpenCV handles filter execution.

## Settings

| Setting | Default | Meaning |
| --- | --- | --- |
| `font_size` | 12 | Font rasterization size, integer >= 2 |
| `font_path` | None | Optional font file; bundled Pillow font otherwise |
| `columns` | None | Positive output width in characters |
| `characters` | ` .:-=+*#%@` | Printable ASCII ramp, sorted by measured density |
| `use_edge` | True | Overlay directional edge glyphs |
| `edge_threshold` | 70 | Percentile of nontrivial cell strengths, 0–100; higher selects fewer |
| `edge_sensitivity` | None | Optional confidence selection, 0–100; zero is off; replaces percentile ranking |
| `edge_amount` | None | Optional gentle tonal emphasis, 0–100; uses brightness glyphs instead of directional substitutions |
| `edge_strength` | 0.01 | Absolute minimum RMS gradient on normalized XDoG |
| `edge_coherence` | 0.3 | Minimum tensor coherence, 0–1 |
| `sigma_e` | 1.0 | Narrow Gaussian scale in pixels |
| `k` | 1.6 | Broad/narrow Gaussian ratio, > 1 |
| `sharpness` | 5.0 | XDoG enhancement weight, nonnegative |
| `use_color` | False | Tint glyphs using source colors |
| `foreground` | `#242424` | RGB hex ink color |
| `background` | `#FFFFFF` | RGB hex paper color |
| `background_mode` | `solid` | `source` derives a subtle light/dark gradient from image colors |
| `edge_map` | None | Optional normalized 2D learned-edge confidence; skips DoG |

`DoGFilter.response` returns signed float32 extended DoG:
`(1 + sharpness) * G(sigma_e) - sharpness * G(k * sigma_e)`.
`difference_of_gaussian` applies a soft XDoG threshold and returns uint8.
`flow_dog` samples across structure-tensor normals, integrates along
sign-consistent tangent streamlines, applies the soft threshold, and optionally
smooths along tangents. All sampling uses bilinear interpolation and explicit
boundary handling.

Additional filter settings: `sigma_c=1` (tensor smoothing), `sigma_m=2`
(streamline smoothing), `sigma_a=0.7` (final smoothing; zero disables),
`phi=10` (threshold steepness), and `white_point=128` (threshold in 0–255 units).
`white_point` and `phi` affect line-image outputs; ASCII edges use the
unthresholded response. Transparent inputs use their BGR channels; composite
against your intended background first when transparency matters.

The public classes and convenience function `generate_ascii_image_sprite` remain
available. Visual defaults have intentionally changed: calibrated glyphs, font
metrics, antialiasing, and a soft DoG replace the old binary, square-cell pipeline.
The old on-disk `.npz` caches are unused. This is an offline CPU renderer, not a
GPU shader or a real-time performance claim.

## Development

```sh
pip install -e '.[dev]'
pytest
black --check src/ascii_art/filters/dog_filter.py src/ascii_art/generators/sprite_generator.py src/ascii_art/cli.py tests
```

References: [AcerolaFX DoG guide](https://github.com/GarrettGunnell/AcerolaFX/wiki/User-Guide),
[OpenCV filtering documentation](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html).

## Browser page (WebAssembly)

A basic local-only HTML interface runs this same Python renderer through Pyodide
in a web worker, with a large live preview, debounced controls, source-derived gradients, tiny
WASM neural edges and PNG/text downloads:

```sh
bun run --cwd web start
# Open http://127.0.0.1:8766
```

See [web/README.md](web/README.md) for static hosting, runtime downloads and limits.
