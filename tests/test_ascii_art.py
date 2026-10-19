"""Numerical, rendering and CLI regressions for the ASCII pipeline."""
import cv2
import numpy as np
import pytest

from ascii_art import DoGFilter, SpriteASCIIGenerator
from ascii_art.cli import main
from ascii_art.generators.sprite_generator import glyph_atlas


def test_dog_matches_independent_gaussians_and_preserves_input():
    image = np.random.default_rng(42).integers(0, 256, (41, 53), dtype=np.uint8)
    original = image.copy()
    dog = DoGFilter()
    gray = image.astype(np.float32) / 255
    expected = 6 * cv2.GaussianBlur(gray, (0, 0), 1) - 5 * cv2.GaussianBlur(
        gray, (0, 0), 1.6
    )
    np.testing.assert_allclose(dog.response(image), expected, atol=1e-6)
    np.testing.assert_array_equal(image, original)
    np.testing.assert_array_equal(dog.response(image), dog.response(image))


@pytest.mark.parametrize("shape", [(1, 1), (1, 13), (17, 1), (19, 23)])
def test_flow_preserves_constant_at_boundaries(shape):
    image = np.full(shape, 0.6, np.float32)
    dog = DoGFilter()
    result = dog.flow_dog(image)
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result, np.full(shape, 255, np.uint8))
    np.testing.assert_allclose(image, 0.6)


def test_flow_retains_antialiasing():
    image = np.tile(np.linspace(0, 1, 31, dtype=np.float32), (29, 1))
    output = DoGFilter().flow_dog(image)
    assert np.any((output > 0) & (output < 255))


@pytest.mark.parametrize("shape", [(1, 1), (13, 17), (61, 79)])
def test_tiny_odd_images_and_text_agree_on_grid(shape):
    generator = SpriteASCIIGenerator({"columns": 9})
    image = np.zeros((*shape, 3), np.uint8)
    result = generator.generate_ascii(image)
    lines = generator.generate_text(image).splitlines()
    assert all(len(line) == 9 for line in lines)
    assert result.shape == (
        len(lines) * generator.grid_size,
        9 * generator.char_width,
        3,
    )
    assert result.dtype == np.uint8


def test_density_mapping_and_no_edges_on_flat_fields():
    generator = SpriteASCIIGenerator({"columns": 8})
    white = generator.generate_text(np.full((80, 80), 255, np.uint8))
    assert set(white) <= {" ", "\n"}
    dark = generator.generate_text(np.zeros((80, 80), np.uint8))
    assert set(dark) <= {generator.characters[generator._density_order[-1]], "\n"}


@pytest.mark.parametrize(
    "kind,glyph",
    [("vertical", "|"), ("horizontal", "-"), ("diagonal", "\\"), ("antidiagonal", "/")],
)
def test_edge_tangents(kind, glyph):
    y, x = np.indices((128, 128))
    conditions = {
        "vertical": x > 64,
        "horizontal": y > 64,
        "diagonal": y > x,
        "antidiagonal": y + x > 128,
    }
    image = conditions[kind].astype(np.uint8) * 255
    gen = SpriteASCIIGenerator(
        {"columns": 16, "edge_threshold": 0, "edge_coherence": 0.5}
    )
    text = gen.generate_text(image)
    assert text.count(glyph) >= 4


def test_font_cache_size_and_working_directory(tmp_path, monkeypatch):
    glyph_atlas.cache_clear()
    monkeypatch.chdir(tmp_path)
    small = SpriteASCIIGenerator({"font_size": 8})
    large = SpriteASCIIGenerator({"font_size": 24})
    again = SpriteASCIIGenerator({"font_size": 8})
    assert large.grid_size > small.grid_size
    assert small.char_masks is again.char_masks
    assert not list(tmp_path.iterdir())


def test_color_uses_bgr_source():
    gen = SpriteASCIIGenerator({"use_color": True, "use_edge": False, "columns": 3})
    image = np.zeros((32, 32, 3), np.uint8)
    image[..., 2] = 128
    result = gen.generate_ascii(image)
    assert np.any(result[..., 2] > result[..., 0])


@pytest.mark.parametrize(
    "image",
    [
        None,
        np.empty((0, 2)),
        np.zeros((2, 2, 2)),
        np.ones((2, 2)) * np.nan,
        np.ones((2, 2)) * 256,
        np.zeros((2, 2), np.uint16),
    ],
)
def test_invalid_images(image):
    with pytest.raises(ValueError):
        SpriteASCIIGenerator().generate_ascii(image)


@pytest.mark.parametrize(
    "settings",
    [
        {"font_size": 0},
        {"columns": -1},
        {"edge_threshold": 101},
        {"characters": "x"},
        {"sigma_e": 0},
        {"foreground": "bad"},
    ],
)
def test_invalid_settings(settings):
    with pytest.raises(ValueError):
        SpriteASCIIGenerator(settings)


def test_float_and_uint8_equivalent():
    image = np.random.default_rng(7).integers(0, 256, (42, 51, 3), np.uint8)
    gen = SpriteASCIIGenerator()
    np.testing.assert_array_equal(
        gen.generate_ascii(image), gen.generate_ascii(image.astype(np.float32) / 255)
    )


def test_cli_text_and_errors(tmp_path):
    source = tmp_path / "input.png"
    cv2.imwrite(str(source), np.full((32, 32, 3), 128, np.uint8))
    output = tmp_path / "output.txt"
    main([str(source), "--columns", "5", "--text", "-o", str(output)])
    assert all(len(line) == 5 for line in output.read_text().splitlines())
    with pytest.raises(SystemExit) as error:
        main([str(source), "-o", str(tmp_path / "missing" / "output.png")])
    assert error.value.code == 2


def test_combined_exports_match_separate_calls_with_one_filter_pass(monkeypatch):
    image = np.random.default_rng(11).integers(0, 256, (63, 91, 3), np.uint8)
    generator = SpriteASCIIGenerator({"use_color": True})
    expected_image = generator.generate_ascii(image)
    expected_text = generator.generate_text(image)
    calls = []
    original_response = generator.dog.response

    def response(image):
        calls.append(True)
        return original_response(image)

    monkeypatch.setattr(generator.dog, "response", response)
    combined_image, combined_text = generator.generate_ascii_and_text(image)
    np.testing.assert_array_equal(combined_image, expected_image)
    assert combined_text == expected_text
    assert len(calls) == 1


def test_learned_edges_bypass_dog_and_preserve_tangent_direction(monkeypatch):
    image = np.zeros((128, 128), np.uint8)
    image[:, 64:] = 255
    confidence = np.zeros((128, 128), np.float32)
    confidence[:, 62:66] = 0.9
    generator = SpriteASCIIGenerator(
        {"columns": 16, "edge_map": confidence, "edge_threshold": 0}
    )

    def unexpected_dog(_):
        raise AssertionError("DoG must not run when a learned map is supplied")

    monkeypatch.setattr(generator.dog, "response", unexpected_dog)
    assert generator.generate_text(image).count("|") >= 4


def test_empty_learned_map_suppresses_edge_glyphs():
    image = np.random.default_rng(23).integers(0, 256, (64, 64), np.uint8)
    generator = SpriteASCIIGenerator({"edge_map": np.zeros((16, 16), np.float32)})
    assert set(generator.generate_text(image)) <= set(generator.characters + "\n")


@pytest.mark.parametrize(
    "invalid", [np.ones((2, 2)) * 2, np.ones((2, 2)) * np.nan, np.zeros((2, 2, 3))]
)
def test_reject_invalid_learned_edges(invalid):
    with pytest.raises(ValueError):
        SpriteASCIIGenerator({"edge_map": invalid})


def test_legacy_utility_exports_remain_available():
    from ascii_art import hex_to_bgr, sobel
    from ascii_art.utils import downsample_mode

    assert hex_to_bgr("#ff0000") == (0, 0, 255)
    magnitude, _ = sobel(np.zeros((8, 8), np.float32))
    assert np.all(magnitude == 0)
    assert downsample_mode(np.ones((4, 4)), 2).shape == (2, 2)
