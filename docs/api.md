# API reference

```python
from ascii_art import SpriteASCIIGenerator, DoGFilter, generate_ascii_image_sprite
```

`SpriteASCIIGenerator(settings=None)` builds a reusable, font-calibrated renderer.
See the [settings table](../readme.md#settings) for defaults and parameter units.

- `generate_ascii(image) -> np.ndarray`: return an antialiased BGR uint8 image.
- `generate_grid(image) -> dict`: return text, per-cell RGB colors, paper rows,
  and grid geometry. Optional browser `text_metrics` bypasses Pillow masks.
- `generate_text(image) -> str`: return the same character grid as plain ASCII.
- `generate_ascii_and_text(image) -> tuple[np.ndarray, str]`: generate both exports with one resampling/filtering pass.
- `generate_ascii_image_sprite(image, settings=None)`: convenience image conversion.

Inputs are nonempty HxW grayscale, HxWx3 BGR, or HxWx4 BGRA arrays, as uint8
or finite normalized floating point in [0, 1]. Alpha is ignored; composite first
if needed. Unsupported shapes, dtypes and settings raise `ValueError`. Input
arrays are never mutated. The default font works independently of the current
working directory. Output dimensions are multiples of the measured cell size.

`DoGFilter(settings=None)` exposes:

- `response(image)`: signed float32 extended DoG response, before thresholding.
- `difference_of_gaussian(image)`: soft-thresholded uint8 XDoG line image.
- `flow_dog(image)`: uint8 lines filtered across edge normals and along tangents.

The [README](../readme.md) describes filtering formulas, scales and thresholds.
Optional `edge_map` settings accept a normalized 2D confidence map and bypass
DoG edge extraction. `edge_sensitivity` enables confidence-based hysteresis
instead of percentile ranking. Set `edge_amount` from 0 to 100 for bounded tonal
emphasis instead of substituting line glyphs; zero skips edge processing. `background_mode="source"` selects palette-derived gradient
paper with automatic monochrome ink polarity. Solid paper remains the default.

Legacy utility exports `sobel`, `hex_to_bgr`, and `downsample_mode` remain available;
the new renderer uses OpenCV derivatives and pooled tensors instead.
