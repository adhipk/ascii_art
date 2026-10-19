# Acerola reference audit

Audited 2026-10-18 against the author's published code accompanying
[I Tried Turning Games Into Text](https://www.youtube.com/watch?v=gg40RWiaHRY):

- [Unity shader](https://github.com/GarrettGunnell/Post-Processing/blob/main/Assets/ASCII/ASCII.shader)
- [Unity tile compute shader](https://github.com/GarrettGunnell/Post-Processing/blob/main/Assets/ASCII/ASCII.compute)
- [Unity pass orchestration and defaults](https://github.com/GarrettGunnell/Post-Processing/blob/main/Assets/ASCII/ASCII.cs)
- [ReShade version](https://github.com/GarrettGunnell/AcerolaFX/blob/main/Shaders/AcerolaFX_ASCII.fx)

The full YouTube transcript could not be retrieved in this environment. This is
a source-code audit, not a claim to have reread a transcript. Upstream links
track mutable branches; they are not pinned shader snapshots.

## Findings

Our previous renderer was inspired by these techniques but was not a faithful
recreation. In particular, the browser's contour slider had become a tonal
enhancement control and did not draw directional glyphs at all.

| Stage | Published approach | Previous Python/browser implementation | New `edge_method="acerola"` |
| --- | --- | --- | --- |
| Edge preprocessing | Finite separable Gaussian pair, difference, binary threshold | Unthresholded enhanced XDoG or learned confidence | Finite separable Gaussian pair, difference, binary threshold |
| Direction extraction | Per-pixel gradient of binary mask; passes labeled Sobel use Scharr coefficients | Gradients followed by pooled structure tensors | Per-pixel Scharr gradient of binary mask |
| Tile selection | Four direction buckets per 8×8 tile; minimum winning-bucket count | Tensor coherence / percentile or confidence hysteresis | Same bucket voting and winning-count criterion |
| Edge composition | Selected edge glyph replaces fill glyph | Python legacy substitution; browser tonal adjustment | Selected edge glyph replaces fill glyph |
| Fill mapping | Exposure/attenuation and fixed ten-level texture lookup | Nearest measured glyph density | Retains nearest measured glyph density |
| Glyph display | Fixed 8×8 texture atlas | Pillow font or real browser text | Retains actual font rendering |
| Extra geometry | ReShade optionally uses scene depth/normals | Image-only input | Image-only input |

The **edge algorithm** now follows the reference architecture. The **entire
renderer is still an adaptation**, because its font metrics, fill mapping,
color treatment, background gradient and resampling differ. Unicode shape
matching and ligature matching are not part of Acerola's reference algorithm
and are not implemented by this change.

## Python usage

```python
from ascii_art import SpriteASCIIGenerator
from ascii_art.filters.acerola import edge_pass

renderer = SpriteASCIIGenerator({
    "columns": 100,
    "edge_method": "acerola",
    "use_edge": True,
    "edge_min_pixels": 8,
    "acerola_sigma": 2.0,
    "acerola_k": 1.6,
    "acerola_tau": 1.0,
    "acerola_threshold": 0.005,
    "acerola_radius": 2,
})
text = renderer.generate_text(image)  # OpenCV BGR or grayscale array
output = renderer.generate_ascii(image)

# Inspect intermediate stages using the same analysis grid as the renderer.
gray, source, rows, cols = renderer._prepare(image)
stages = edge_pass(gray, rows, cols, renderer.settings)
# luminance, dog, binary, gx, gy, directions, tiles
```

Analysis uses an 8×8 tile for every output character; the display cell follows
the selected font. Image area resampling to this analysis grid is an explicit
adaptation. The legacy `tensor` Python default is preserved for existing callers.
In the browser select **Acerola · DoG + direction vote**, enable **Contour
emphasis**, and adjust **Minimum edge votes**. Its default is eight; larger
values retain fewer edge tiles. The prior neural and XDoG tonal modes remain
explicit alternatives. `edge_amount=0` and `use_edge=False` disable reference
edges; positive tonal strength values otherwise do not affect this algorithm.

## Deliberate corrections and limits

- The shader's horizontal pass contains `3 + lum1` where a separable derivative
  requires `3 * lum1`. We use the proper Scharr derivative, preventing spurious
  responses on constant fields.
- The shader indexes a four-element bucket array with the no-edge sentinel
  `-1`. We exclude invalid directions. Empty tiles never become edges, including
  when the minimum is zero. Ties favor the first direction as in the source.
- Directions are converted to top-left image coordinates. Both diagonal
  polarities are tested; gradient normals are mapped to perpendicular strokes.
- Float32/OpenCV operations do not emulate half-float GPU intermediates.
- Original texture atlases, game depth/normals, fog, bloom and other post effects
  are not reproduced. No pixel-exact visual parity or browser FPS is claimed.

Tests compare DoG against an independent explicit 2D convolution and derivatives
against explicit kernels. They cover direction signs, winning-count thresholds,
ties, zero gradients, flat fields and renderer composition. A real shader/browser
comparison remains necessary before claiming full visual parity.
