"""Independent numeric checks for the reference edge pipeline."""
import cv2
import numpy as np
import pytest

from ascii_art import SpriteASCIIGenerator
from ascii_art.filters.acerola import direction_votes, edge_pass, tile_directions


def test_reference_dog_matches_explicit_finite_2d_convolution():
    gray = np.random.default_rng(8).random((16, 24), dtype=np.float32)
    result = edge_pass(gray, 2, 3, {})
    padded = np.pad(gray, 2, mode="edge")
    outputs = []
    for sigma in (2, 3.2):
        kernel = np.exp(-0.5 * (np.arange(-2, 3) / sigma) ** 2)
        kernel /= kernel.sum()
        expected = np.zeros(gray.shape)
        for y in range(5):
            for x in range(5):
                expected += kernel[y] * kernel[x] * padded[y : y + 16, x : x + 24]
        outputs.append(expected)
    np.testing.assert_allclose(result["dog"], outputs[0] - outputs[1], atol=2e-7)
    np.testing.assert_array_equal(result["binary"], result["dog"] >= 0.005)
    # Independently evaluate the derivative kernels with point-clamp borders.
    mask = np.pad(result["binary"], 1, mode="edge")
    gx = np.zeros(gray.shape)
    gy = np.zeros(gray.shape)
    for i, smooth in enumerate((3, 10, 3)):
        gx += smooth * (mask[i : i + 16, 2:26] - mask[i : i + 16, :24])
        gy += smooth * (mask[2:18, i : i + 24] - mask[:16, i : i + 24])
    np.testing.assert_array_equal(result["gx"], gx)
    np.testing.assert_array_equal(result["gy"], gy)


def test_axes_and_diagonals_are_strokes_in_top_left_coordinates():
    gx = np.array([1, -1, 0, 0, 1, -1, 1, -1, 0], np.float32)
    gy = np.array([0, 0, 1, -1, 1, -1, -1, 1, 0], np.float32)
    np.testing.assert_array_equal(direction_votes(gx, gy), [0, 0, 1, 1, 3, 3, 2, 2, -1])


def test_dominant_bucket_count_not_total_edge_count_controls_selection():
    directions = np.full((8, 16), -1, np.int8)
    directions[0, :8] = 2
    directions[0, 8:12] = 0
    directions[0, 12:] = 1
    np.testing.assert_array_equal(tile_directions(directions, 8), [[2, -1]])
    np.testing.assert_array_equal(tile_directions(directions, 4), [[2, 0]])
    assert tile_directions(directions, 9).tolist() == [[-1, -1]]
    assert tile_directions(np.full((8, 8), -1), 0).item() == -1


@pytest.mark.parametrize("level", [0, 0.25, 0.5, 1])
def test_uniform_fields_never_invent_edges(level):
    result = edge_pass(np.full((32, 48), level, np.float32), 4, 6, {})
    assert np.all(result["tiles"] == -1)
    assert not np.any(result["gx"]) and not np.any(result["gy"])


def test_reference_overlays_voted_glyphs_and_ignores_tensor_pipeline(monkeypatch):
    image = np.zeros((96, 128), np.uint8)
    cv2.line(image, (10, 10), (110, 80), 255, 3)
    renderer = SpriteASCIIGenerator(
        {"columns": 16, "edge_method": "acerola", "edge_amount": 100}
    )
    monkeypatch.setattr(
        renderer.dog, "response", lambda _: pytest.fail("XDoG must not run")
    )
    gray, source, rows, cols = renderer._prepare(image)
    expected = edge_pass(gray, rows, cols, renderer.settings)["tiles"]
    indices = renderer._indices(gray, rows, cols, renderer._paper(source))
    selected = expected >= 0
    assert selected.any()
    np.testing.assert_array_equal(
        indices[selected],
        renderer.num_ascii_chars + np.array([2, 0, 3, 1])[expected[selected]],
    )
    assert np.all(indices[~selected] < renderer.num_ascii_chars)
    assert renderer.generate_text(image) == renderer.generate_ascii_and_text(image)[1]


@pytest.mark.parametrize(
    "settings",
    [
        {"acerola_sigma": 0},
        {"acerola_radius": 0},
        {"acerola_tau": float("nan")},
        {"edge_min_pixels": 65},
    ],
)
def test_invalid_reference_settings_fail(settings):
    with pytest.raises(ValueError):
        edge_pass(np.zeros((8, 8), np.float32), 1, 1, settings)
