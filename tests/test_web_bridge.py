"""Exercise the exact browser adapter with native Python image dependencies."""
import importlib.util
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from ascii_art import SpriteASCIIGenerator

spec = importlib.util.spec_from_file_location(
    "web_bridge", Path(__file__).parents[1] / "web/bridge.py"
)
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)


def test_browser_rgba_matches_python_renderer():
    rgba = np.random.default_rng(5).integers(0, 256, (32, 48, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    settings = {"columns": 20, "font_size": 12, "use_color": True}
    result = bridge.render_rgba(rgba.tobytes(), 48, 32, settings)
    expected = SpriteASCIIGenerator(settings)
    bgr = np.ascontiguousarray(rgba[..., 2::-1])
    png = np.asarray(Image.open(BytesIO(result["png"])))
    np.testing.assert_array_equal(png, expected.generate_ascii(bgr)[..., ::-1])
    assert result["text"] == expected.generate_text(bgr)
    assert png.shape[:2] == (result["height"], result["width"])


def test_transparent_source_composites_on_background():
    rgba = np.zeros((20, 20, 4), dtype=np.uint8)
    rgba[..., 0] = 255
    result = bridge.render_rgba(rgba.tobytes(), 20, 20, {"columns": 20})
    assert set(result["text"]) <= {" ", "\n"}
    assert np.all(np.asarray(Image.open(BytesIO(result["png"]))) == 255)


@pytest.mark.parametrize(
    "pixels,width,height,settings",
    [
        (b"", 20, 20, {}),
        (b"", 0, 20, {}),
        (b"", 1201, 1200, {}),
        (bytes(20 * 20 * 4), 20, 20, {"columns": 200}),
        (bytes(20 * 20 * 4), 20, 20, {"columns": 20.5}),
        (bytes(20 * 20 * 4), 20, 20, {"font_size": 80}),
        (bytes(4 * 1200), 1, 1200, {"columns": 180}),
    ],
)
def test_browser_rejects_malformed_or_unbounded_jobs(pixels, width, height, settings):
    with pytest.raises(ValueError):
        bridge.render_rgba(pixels, width, height, settings)


def test_source_gradient_is_exported_and_ignores_transparent_hidden_rgb():
    first = np.zeros((32, 48, 4), np.uint8)
    first[..., 0] = 255
    second = first.copy()
    second[..., :3] = [0, 0, 255]
    settings = {"columns": 20, "background_mode": "source", "use_edge": False}
    a = bridge.render_rgba(first.tobytes(), 48, 32, settings)
    b = bridge.render_rgba(second.tobytes(), 48, 32, settings)
    image = np.asarray(Image.open(BytesIO(a["png"])))
    assert a["png"] == b["png"]
    assert not np.array_equal(image[0], image[-1])
    assert set(a["text"]) <= {" ", "\n"}


def test_source_gradient_adapts_to_warm_and_cool_images():
    from ascii_art.utils.background import source_palette, gradient_rows

    warm = np.full((16, 16, 3), [0.45, 0.65, 0.95], np.float32)
    cool = warm[..., ::-1].copy()
    warm_paper = gradient_rows(source_palette(warm), 64)
    cool_paper = gradient_rows(source_palette(cool), 64)
    assert warm_paper[..., 2].mean() > warm_paper[..., 0].mean()
    assert cool_paper[..., 0].mean() > cool_paper[..., 2].mean()
    assert np.abs(np.diff(warm_paper, axis=0)).max() < 3


def test_gradient_combined_exports_match_separate_calls():
    image = np.random.default_rng(9).integers(0, 256, (33, 49, 3), np.uint8)
    renderer = SpriteASCIIGenerator({"background_mode": "source", "columns": 20})
    output, text = renderer.generate_ascii_and_text(image)
    np.testing.assert_array_equal(output, renderer.generate_ascii(image))
    assert text == renderer.generate_text(image)


def test_browser_requires_model_map_when_tiny_detector_is_selected():
    with pytest.raises(ValueError, match="confidence map"):
        bridge.render_rgba(bytes(32 * 32 * 4), 32, 32, {"edge_method": "tiny"})
    result = bridge.render_rgba(
        bytes(32 * 32 * 4), 32, 32, {"edge_method": "tiny"}, bytes(16 * 16), 16, 16
    )
    assert result["png"].startswith(b"\x89PNG")


def test_downsampling_palette_ignores_hidden_transparent_color():
    from ascii_art.utils.background import source_palette

    image = np.zeros((256, 256, 4), np.float32)
    image[::2, ::2] = [0.4, 0.6, 0.9, 1]
    other = image.copy()
    image[1::2, :, :3] = [1, 0, 0]
    other[1::2, :, :3] = [0, 0, 1]
    np.testing.assert_allclose(source_palette(image), source_palette(other), atol=1e-4)


@pytest.mark.parametrize(
    "name", ["taylor.png", "webb-poster.webp", "spacesuit-patent.webp", "test.png"]
)
def test_bundled_examples_render_with_source_gradient(name):
    path = Path(__file__).parents[1] / "web/examples" / name
    with Image.open(path) as original:
        image = original.convert("RGBA")
        image.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
        width, height = image.size
        result = bridge.render_rgba(
            image.tobytes(),
            width,
            height,
            {
                "columns": 100,
                "background_mode": "source",
                "use_color": True,
                "edge_method": "dog",
            },
        )
    with Image.open(BytesIO(result["png"])) as output:
        assert output.size == (result["width"], result["height"])
        assert output.mode == "RGB"
    assert all(len(line) == 100 for line in result["text"].splitlines())


def test_float_model_map_preserves_faint_confidence(monkeypatch):
    rgba = np.full((32, 32, 4), 255, np.uint8)
    confidence = np.linspace(0.001, 0.002, 16 * 16, dtype="<f4").reshape(16, 16)
    received = []
    original_init = SpriteASCIIGenerator.__init__

    def capture(self, settings=None):
        received.append(settings["edge_map"].copy())
        original_init(self, settings)

    monkeypatch.setattr(SpriteASCIIGenerator, "__init__", capture)
    bridge.render_rgba(
        rgba.tobytes(),
        32,
        32,
        {"edge_method": "tiny", "edge_sensitivity": 50, "edge_amount": 2},
        confidence.tobytes(),
        16,
        16,
        "float32",
    )
    np.testing.assert_array_equal(received[0], confidence)
    assert np.unique(received[0]).size == 256


def test_browser_text_calibration_selects_grid_output_without_png(monkeypatch):
    from ascii_art.generators import sprite_generator

    def forbidden(*_):
        raise AssertionError("No glyph atlas should be produced for browser text")

    monkeypatch.setattr(sprite_generator, "glyph_atlas", forbidden)
    rgba = np.full((32, 32, 4), 255, np.uint8)
    result = bridge.render_rgba(
        rgba.tobytes(),
        32,
        32,
        {
            "columns": 20,
            "use_edge": False,
            "text_metrics": {
                "width": 8,
                "height": 17,
                "densities": [0, 0.01, 0.03, 0.06, 0.09, 0.12, 0.16, 0.20, 0.25, 0.30],
            },
        },
    )
    assert "png" not in result
    assert result["columns"] == 20
    assert set(result["text"]) <= {" ", "\n"}
