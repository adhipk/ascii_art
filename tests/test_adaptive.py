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


def glyph_settings(penalty=1.0):
    config = settings()
    config["adaptive_penalty"] = penalty
    config["adaptive_objective"] = "pixels"
    config["background"] = "#000000"
    config["foreground"] = "#ffffff"
    for size in (8, 16, 32):
        mask = np.zeros((size, size), np.float32)
        mask[:, size // 2] = 1
        config["adaptive_metrics"][str(size)].update(
            {
                "kernelCharacters": " |",
                "masks": [np.zeros(size * size).tolist(), mask.ravel().tolist()],
            }
        )
    return config


def test_flat_paper_uses_large_blank_cells_without_gaps():
    result = adaptive_grid(np.zeros((160, 160), np.uint8), glyph_settings())
    assert len(result["tiles"]) == 25
    assert {tile["size"] for tile in result["tiles"]} == {32}
    assert all(tile["char"] == " " for tile in result["tiles"])
    assert result["adaptiveCost"] == 0
    assert_partition(result)


def test_exact_large_stroke_survives_strong_edges():
    image = np.zeros((160, 160), np.uint8)
    image[:, 16::32] = 255
    result = adaptive_grid(image, glyph_settings())
    assert len(result["tiles"]) == 25
    assert all(tile["size"] == 32 and tile["char"] == "|" for tile in result["tiles"])
    assert result["adaptiveCost"] == pytest.approx(25)
    assert_partition(result)


def test_repeated_small_strokes_require_subdivision():
    image = np.zeros((160, 160), np.uint8)
    image[:, 4::8] = 255
    result = adaptive_grid(image, glyph_settings())
    assert len(result["tiles"]) == 400
    assert all(tile["size"] == 8 and tile["char"] == "|" for tile in result["tiles"])
    assert result["adaptiveCost"] == pytest.approx(400)
    assert_partition(result)


def test_intermediate_strokes_select_sixteen_pixel_cells():
    image = np.zeros((160, 160), np.uint8)
    image[:, 8::16] = 255
    result = adaptive_grid(image, glyph_settings())
    assert len(result["tiles"]) == 100
    assert all(tile["size"] == 16 and tile["char"] == "|" for tile in result["tiles"])
    assert_partition(result)


def test_higher_penalty_reduces_visible_glyph_count():
    image = np.zeros((160, 160), np.uint8)
    image[:, 4::8] = 255
    low = adaptive_grid(image, glyph_settings(0))
    high = adaptive_grid(image, glyph_settings(1000))
    assert sum(t["char"] != " " for t in low["tiles"]) == 400
    assert all(t["char"] == " " for t in high["tiles"])
    assert_partition(high)


def test_partial_outer_tiles_are_subdivided_not_cropped():
    result = adaptive_grid(np.zeros((73, 160), np.uint8), glyph_settings())
    assert result["rows"] == 9
    assert_partition(result)
    assert len(result["text"].splitlines()) == 9
    assert all(len(line) == 20 for line in result["text"].splitlines())


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
        rgba.tobytes(),
        80,
        80,
        {**glyph_settings(), "layout": "adaptive", "font_size": 12},
    )
    assert_partition(result)
    assert result["cellWidth"] == result["cellHeight"] == 8
    assert set(result["tileMetrics"]) == {"8", "16", "32"}
    assert all(tile["char"] == " " for tile in result["tiles"])


@pytest.mark.parametrize("penalty", [-1, float("inf"), float("nan"), True, "1"])
def test_invalid_penalties_fail(penalty):
    with pytest.raises(ValueError, match="adaptive_penalty"):
        adaptive_grid(np.zeros((32, 32), np.uint8), glyph_settings(penalty))


def test_missing_calibrated_masks_fail():
    with pytest.raises(ValueError, match="calibrated glyph masks"):
        adaptive_grid(np.zeros((32, 32), np.uint8), settings())


def test_objective_matches_direct_rendered_rgb_error():
    image = np.zeros((160, 160, 3), np.uint8)
    image[:, 8::16] = [20, 180, 240]
    config = {**glyph_settings(0.1), "use_color": True}
    result = adaptive_grid(image, config)
    reconstructed = np.zeros_like(image, dtype=np.float64)
    count = 0
    for tile in result["tiles"]:
        size, x, y = tile["size"], tile["x"], tile["y"]
        metric = config["adaptive_metrics"][str(size)]
        index = metric["kernelCharacters"].index(tile["char"])
        alpha = np.array(metric["masks"][index]).reshape(size, size, 1)
        reconstructed[y : y + size, x : x + size] = (
            alpha * np.array(tile["color"]) / 255
        )
        count += tile["char"] != " "
    error = np.sum((image[..., ::-1] / 255 - reconstructed) ** 2) / 3
    assert result["adaptiveCost"] == pytest.approx(error + 0.1 * count, abs=1e-3)


@pytest.mark.parametrize("pattern", ["corner", "diagonal", "noise"])
def test_cost_matches_exhaustive_search_over_all_seventeen_partitions(pattern):
    from itertools import product

    y, x = np.indices((32, 32))
    if pattern == "corner":
        image = (((x == 16) & (y >= 16)) | ((y == 16) & (x >= 16))) * 255
    elif pattern == "diagonal":
        image = (x == y) * 255
    else:
        image = np.random.default_rng(7).integers(0, 256, (32, 32))
    image = image.astype(np.uint8)
    config = {**glyph_settings(0.7), "columns": 4}
    result = adaptive_grid(image, config)

    def direct_cost(x0: int, y0: int, size: int) -> float:
        patch = image[y0 : y0 + size, x0 : x0 + size] / 255
        metric = config["adaptive_metrics"][str(size)]
        return min(
            float(np.sum((patch - np.array(mask).reshape(size, size)) ** 2))
            + (0.7 if char != " " else 0)
            for char, mask in zip(metric["kernelCharacters"], metric["masks"])
        )

    possibilities = [direct_cost(0, 0, 32)]
    # Each quadrant can independently be one 16px tile or four 8px tiles.
    for split in product((False, True), repeat=4):
        cost = 0.0
        for (x0, y0), subdivide in zip(((0, 0), (16, 0), (0, 16), (16, 16)), split):
            cost += (
                sum(direct_cost(x0 + dx, y0 + dy, 8) for dy in (0, 8) for dx in (0, 8))
                if subdivide
                else direct_cost(x0, y0, 16)
            )
        possibilities.append(cost)
    assert result["adaptiveCost"] == pytest.approx(min(possibilities), abs=1e-4)
    assert_partition(result)


@pytest.mark.parametrize("use_color", [False, True])
@pytest.mark.parametrize("light_paper", [False, True])
def test_photographic_ramp_uses_varied_real_font_glyphs(use_color, light_paper):
    from PIL import Image, ImageDraw, ImageFont
    from pathlib import Path

    config = glyph_settings()
    config["adaptive_objective"] = "tone"
    config["use_color"] = use_color
    if light_paper:
        config.update(background="#ffffff", foreground="#000000")
    chars = " .:-=+*#%@/|\\"
    font_path = str(Path(__file__).parents[1] / "fonts/FiraCode-Regular.ttf")
    for size in (8, 16, 32):
        font = ImageFont.truetype(font_path, round(size * 0.8))
        masks = []
        for char in chars:
            cell = Image.new("L", (size, size))
            draw = ImageDraw.Draw(cell)
            draw.text((size / 2, size / 2), char, font=font, anchor="mm", fill=255)
            masks.append((np.array(cell) / 255).ravel().tolist())
        config["adaptive_metrics"][str(size)].update(
            {
                "kernelCharacters": chars,
                "masks": masks,
                "densities": [float(np.mean(mask)) for mask in masks[:10]],
            }
        )
    image = np.tile(np.linspace(0, 255, 160).astype(np.uint8), (160, 1))
    result = adaptive_grid(image, config)
    chosen = [tile["char"] for tile in result["tiles"]]
    assert len(set(chosen)) >= 5
    assert chosen.count("@") < len(chosen) / 2
    assert_partition(result)


@pytest.mark.parametrize("pattern", ["vertical", "diagonal", "corner"])
def test_detected_contours_remain_directional_with_adaptive_sizes(pattern):
    from ascii_art.generators.sprite_generator import SpriteASCIIGenerator

    y, x = np.indices((160, 160))
    if pattern == "vertical":
        image = (x >= 79) * 255
    elif pattern == "diagonal":
        image = (x >= y) * 255
    else:
        image = ((x >= 79) & (y >= 79)) * 255
    config = {**glyph_settings(), "use_edge": True, "edge_min_pixels": 1}
    for size in (8, 16, 32):
        metric = config["adaptive_metrics"][str(size)]
        metric["kernelCharacters"] = " |-/\\"
        metric["masks"].extend([metric["masks"][1]] * 3)
    result = adaptive_grid(image.astype(np.uint8), config)
    renderer = SpriteASCIIGenerator(
        {**config, "text_metrics": config["adaptive_metrics"]["8"]}
    )
    gray, source, rows, cols = renderer._prepare(image.astype(np.uint8))
    indices = renderer._indices(gray, rows, cols, renderer._paper(source))
    expected = indices >= renderer.num_ascii_chars
    assert expected.any()
    preserved = np.zeros_like(expected)
    for tile in result["tiles"]:
        tx, ty, cells = tile["x"] // 8, tile["y"] // 8, tile["size"] // 8
        if expected[ty : ty + cells, tx : tx + cells].any():
            assert tile["char"] in "-|/\\"
            preserved[ty : ty + cells, tx : tx + cells] |= expected[
                ty : ty + cells, tx : tx + cells
            ]
    assert np.array_equal(preserved, expected)
    assert_partition(result)


@pytest.mark.parametrize("spacing, expected_size", [(32, 32), (16, 16), (8, 8)])
def test_contour_optimizer_selects_known_stroke_scales(
    monkeypatch, spacing, expected_size
):
    import ascii_art.generators.adaptive as adaptive
    import ascii_art.generators.sprite_generator as sprite

    pixels = np.zeros((32, 32), bool)
    pixels[:, spacing // 2 :: spacing] = True
    tiles = np.where(pixels.reshape(4, 8, 4, 8).any(axis=(1, 3)), 0, -1)

    def detector(*args, **kwargs):
        return {"directions": np.where(pixels, 0, -1), "tiles": tiles}

    monkeypatch.setattr(adaptive, "edge_pass", detector)
    monkeypatch.setattr(sprite, "edge_pass", detector)
    config = {**glyph_settings(), "columns": 4, "use_edge": True}
    result = adaptive_grid(np.zeros((32, 32), np.uint8), config)
    strokes = [tile for tile in result["tiles"] if tile["char"] != " "]
    assert strokes
    assert {tile["size"] for tile in strokes} == {expected_size}
    assert all(tile["char"] == "|" for tile in strokes)
    assert_partition(result)


def test_edge_distance_penalizes_missing_and_extra_strokes():
    from ascii_art.generators.adaptive import edge_reconstruction_costs

    target = np.zeros((32, 32), bool)
    target[:, 16] = True
    exact = target.astype(np.float32)
    extra = exact.copy()
    extra[:, 4] = 1
    missing = exact.copy()
    missing[8:24] = 0
    shifted = np.roll(exact, 1, axis=1)
    costs = edge_reconstruction_costs(
        target, np.stack([exact, extra, missing, shifted])
    )
    assert costs[0] == costs[3] == 0
    assert costs[1] > 0
    assert costs[2] > 0


def test_offset_strokes_require_smaller_contour_tiles(monkeypatch):
    import ascii_art.generators.adaptive as adaptive
    import ascii_art.generators.sprite_generator as sprite

    pixels = np.zeros((32, 32), bool)
    pixels[:16, 8] = True
    pixels[16:, 24] = True
    tiles = np.where(pixels.reshape(4, 8, 4, 8).any(axis=(1, 3)), 0, -1)

    def detector(*args, **kwargs):
        return {"directions": np.where(pixels, 0, -1), "tiles": tiles}

    monkeypatch.setattr(adaptive, "edge_pass", detector)
    monkeypatch.setattr(sprite, "edge_pass", detector)
    config = {**glyph_settings(), "columns": 4, "use_edge": True}
    result = adaptive_grid(np.zeros((32, 32), np.uint8), config)
    assert max(t["size"] for t in result["tiles"] if t["char"] == "|") == 16
    assert_partition(result)


@pytest.mark.parametrize("level", [48, 96, 144, 192, 240])
def test_smooth_tones_cannot_use_shape_only_kernels(level):
    config = glyph_settings()
    config["adaptive_objective"] = "tone"
    # Give the shape-only stroke a tempting middle density. It must still
    # be excluded outside detected contour regions.
    for size in (8, 16, 32):
        metric = config["adaptive_metrics"][str(size)]
        metric["kernelCharacters"] = " @/|\\"
        full = np.ones((size, size), np.float32)
        middle = np.zeros_like(full)
        middle[:, ::2] = 1
        metric["masks"] = [np.zeros(size * size).tolist(), full.ravel().tolist()] + [
            middle.ravel().tolist()
        ] * 3
    result = adaptive_grid(np.full((160, 160), level, np.uint8), config)
    assert all(tile["char"] not in "/|\\" for tile in result["tiles"])
    assert_partition(result)
