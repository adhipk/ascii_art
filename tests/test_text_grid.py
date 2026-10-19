"""Browser text-grid output and ink contrast regressions."""
import numpy as np
import pytest

from ascii_art import SpriteASCIIGenerator
from ascii_art.generators import sprite_generator
from ascii_art.utils.color import readable_ink


def metrics():
    return {
        "width": 8,
        "height": 17,
        "densities": [0, 0.01, 0.03, 0.06, 0.09, 0.12, 0.16, 0.20, 0.25, 0.30],
    }


def test_browser_grid_does_not_generate_or_stamp_masks(monkeypatch):
    def forbidden(*_):
        raise AssertionError("Browser grid must not use a Pillow atlas")

    monkeypatch.setattr(sprite_generator, "glyph_atlas", forbidden)
    generator = SpriteASCIIGenerator(
        {
            "text_metrics": metrics(),
            "columns": 20,
            "use_color": True,
            "use_edge": False,
            "background_mode": "source",
        }
    )
    image = np.random.default_rng(55).integers(0, 256, (48, 64, 3), np.uint8)
    result = generator.generate_grid(image)
    assert result["text"] == generator.generate_text(image)
    assert result["cellWidth"] == 8
    assert result["cellHeight"] == 17
    assert result["width"] == 20 * 8
    assert len(result["colors"]) == result["rows"] * 20 * 3
    assert len(result["paper"]) == result["height"] * 3
    assert "png" not in result


def luminance(colors):
    srgb = colors / 255
    linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    return linear @ np.array([0.0722, 0.7152, 0.2126])


@pytest.mark.parametrize("paper", [[15, 20, 25], [230, 240, 250], [120, 128, 134]])
def test_ink_has_readable_contrast_on_light_dark_and_mid_paper(paper):
    bg = np.array(paper, np.float32)
    colors = np.random.default_rng(9).uniform(0, 255, (100, 3)).astype(np.float32)
    result = readable_ink(colors, bg)
    ink_luma, bg_luma = luminance(result), luminance(bg)
    contrast = (np.maximum(ink_luma, bg_luma) + 0.05) / (
        np.minimum(ink_luma, bg_luma) + 0.05
    )
    assert contrast.min() >= 2.999
    assert result.min() >= 0 and result.max() <= 255


def test_already_contrasting_saturated_colors_are_preserved():
    colors = np.array([[255, 0, 0], [0, 0, 255]], np.float32)
    result = readable_ink(colors, np.full(3, 255, np.float32))
    np.testing.assert_allclose(result, colors, atol=1e-3)


@pytest.mark.parametrize(
    "invalid",
    [
        {"width": 0},
        {"width": 8, "height": 17, "densities": [float("nan")] * 10},
        {"width": 8, "height": 17, "densities": [0] * 10},
    ],
)
def test_invalid_text_calibration_rejected(invalid):
    with pytest.raises(ValueError):
        SpriteASCIIGenerator({"text_metrics": invalid})
