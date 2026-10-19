"""Adaptive layout must cover the image, preserve fine detail and bound sizes."""
import numpy as np
import pytest

from ascii_art.generators.adaptive import adaptive_grid


def settings():
    return {
        "columns": 20,
        "use_edge": False,
        "edge_method": "acerola",
        "adaptive_metrics": {
            str(size): {
                "width": size,
                "height": size,
                "baseline": size * 0.8,
                "advance": size * 0.6,
                "fontSize": size * 0.8,
                "densities": np.linspace(0, 0.5, 10).tolist(),
            }
            for size in (8, 16, 32)
        },
    }


def assert_partition(result):
    coverage = np.zeros((result["height"], result["width"]), np.uint8)
    for tile in result["tiles"]:
        x, y, size = tile["x"], tile["y"], tile["size"]
        assert size in (8, 16, 32)
        assert x % size == 0 and y % size == 0
        assert 0 <= x < x + size <= result["width"]
        assert 0 <= y < y + size <= result["height"]
        coverage[y : y + size, x : x + size] += 1
        assert len(tile["char"]) == 1
        assert len(tile["color"]) == 3
    assert np.all(coverage == 1)


def test_flat_image_uses_large_cells_without_gaps():
    result = adaptive_grid(np.full((160, 160), 128, np.uint8), settings())
    assert len(result["tiles"]) == 25
    assert {tile["size"] for tile in result["tiles"]} == {32}
    assert_partition(result)


def test_high_frequency_detail_uses_small_cells_even_without_overlay():
    y, x = np.indices((160, 160))
    result = adaptive_grid(((x + y) % 2 * 255).astype(np.uint8), settings())
    assert {tile["size"] for tile in result["tiles"]} == {8}
    assert_partition(result)


def test_gentle_gradient_uses_intermediate_cells():
    image = np.tile(np.linspace(20, 235, 160).astype(np.uint8), (160, 1))
    result = adaptive_grid(image, settings())
    assert 16 in {tile["size"] for tile in result["tiles"]}
    assert_partition(result)


def test_partial_outer_tiles_are_subdivided_not_cropped():
    result = adaptive_grid(np.full((73, 160), 128, np.uint8), settings())
    assert result["rows"] == 9
    assert_partition(result)
    assert len(result["text"].splitlines()) == 9
    assert all(len(line) == 20 for line in result["text"].splitlines())


def test_overlay_contours_cannot_be_merged_away():
    image = np.zeros((160, 160), np.uint8)
    image[:, 79:] = 255
    result = adaptive_grid(image, {**settings(), "use_edge": True})
    lines = result["text"].splitlines()
    for tile in result["tiles"]:
        if tile["size"] > 8:
            x, y, cells = tile["x"] // 8, tile["y"] // 8, tile["size"] // 8
            assert not any(
                c in "-/|\\"
                for line in lines[y : y + cells]
                for c in line[x : x + cells]
            )
    assert_partition(result)


@pytest.mark.parametrize("metrics", [None, {}, {"8": {"width": 9, "height": 8}}])
def test_invalid_square_calibration_fails(metrics):
    with pytest.raises(ValueError):
        adaptive_grid(
            np.zeros((16, 16), np.uint8), {**settings(), "adaptive_metrics": metrics}
        )


def test_browser_bridge_returns_adaptive_tiles_with_alpha_compositing():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "adaptive_bridge", Path(__file__).parents[1] / "web/bridge.py"
    )
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    rgba = np.full((80, 80, 4), 255, np.uint8)
    rgba[..., :3] = 0
    rgba[..., 3] = 0
    result = bridge.render_rgba(
        rgba.tobytes(), 80, 80, {**settings(), "layout": "adaptive", "font_size": 12}
    )
    assert_partition(result)
    assert result["cellWidth"] == result["cellHeight"] == 8
    assert set(result["tileMetrics"]) == {"8", "16", "32"}
    assert all(tile["char"] == " " for tile in result["tiles"])
