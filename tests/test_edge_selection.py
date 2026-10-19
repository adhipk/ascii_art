"""Confidence stability and conservative contour emphasis regressions."""
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from ascii_art import SpriteASCIIGenerator
from ascii_art.filters.edge_selection import select_edges


def test_hysteresis_keeps_connected_weak_edges_and_rejects_isolated_noise():
    confidence = np.zeros((12, 12), np.float32)
    confidence[2, 2:7] = [0.8, 0.1, 0.1, 0.1, 0.1]
    confidence[8, 2:7] = 0.1
    confidence[10, 10] = 0.9
    mask = select_edges(confidence, 50)
    assert mask[2, 2:7].all()
    assert not mask[8, 2:7].any()
    assert not mask[10, 10]


def test_confidence_selection_is_monotonic_and_ignores_unrelated_histogram():
    source = np.random.default_rng(54).uniform(0, 0.8, (48, 48)).astype(np.float32)
    previous = np.zeros_like(source, dtype=bool)
    for sensitivity in range(101):
        current = select_edges(source, sensitivity)
        assert np.all(~previous | current)
        previous = current
    padded = np.pad(source, ((0, 50), (0, 50)))
    np.testing.assert_array_equal(
        select_edges(source, 50), select_edges(padded, 50)[:48, :48]
    )


def test_zero_strength_does_not_run_detection(monkeypatch):
    image = np.random.default_rng(4).integers(0, 256, (80, 80), np.uint8)
    renderer = SpriteASCIIGenerator({"edge_amount": 0, "edge_sensitivity": 50})

    def forbidden(_):
        raise AssertionError("Detection should not run at zero strength")

    monkeypatch.setattr(renderer.dog, "response", forbidden)
    np.testing.assert_array_equal(
        renderer.generate_ascii(image),
        SpriteASCIIGenerator({"use_edge": False}).generate_ascii(image),
    )


@pytest.mark.parametrize("strength", [1, 2, 20, 100])
def test_emphasis_never_introduces_orientation_glyphs_or_flat_field_noise(strength):
    y, x = np.indices((80, 120))
    image = ((x + y > 90) * 255).astype(np.uint8)
    renderer = SpriteASCIIGenerator(
        {
            "edge_amount": strength,
            "edge_sensitivity": 50,
            "edge_map": np.ones((80, 120), np.float32),
        }
    )
    gray, source, rows, cols = renderer._prepare(image)
    indices = renderer._indices(gray, rows, cols, renderer._paper(source))
    assert np.all(indices < renderer.num_ascii_chars)
    flat = np.full((80, 120), 128, np.uint8)
    np.testing.assert_array_equal(
        renderer.generate_ascii(flat),
        SpriteASCIIGenerator({"use_edge": False}).generate_ascii(flat),
    )


@pytest.mark.parametrize(
    "name", ["taylor.png", "webb-poster.webp", "spacesuit-patent.webp", "test.png"]
)
def test_low_emphasis_has_small_effect_on_supplied_examples(name):
    with Image.open(Path(__file__).parents[1] / "web/examples" / name) as image:
        image = image.convert("RGB")
        image.thumbnail((1200, 1200))
        bgr = np.asarray(image)[..., ::-1].copy()
    baseline = SpriteASCIIGenerator({"columns": 100, "use_edge": False}).generate_ascii(
        bgr
    )
    # Stress test a maximally permissive model map: even this must remain gentle.
    for strength in (1, 2):
        renderer = SpriteASCIIGenerator(
            {
                "columns": 100,
                "edge_amount": strength,
                "edge_sensitivity": 50,
                "edge_map": np.ones((64, 64), np.float32),
            }
        )
        output = renderer.generate_ascii(bgr)
        assert np.abs(output.astype(np.float32) - baseline).mean() < 1
