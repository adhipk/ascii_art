"""Browser adapter shared by Pyodide and native regression tests."""
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image

from ascii_art import SpriteASCIIGenerator
from ascii_art.generators.adaptive import adaptive_grid
from ascii_art.utils.background import gradient_rows, source_palette


def render_rgba(
    pixels: bytes,
    width: int,
    height: int,
    settings: dict[str, Any],
    edge_pixels: bytes | None = None,
    edge_width: int = 0,
    edge_height: int = 0,
    edge_format: str = "uint8",
) -> dict[str, Any]:
    """Accept bounded browser RGBA, composite alpha, and return PNG + plain text."""
    if width < 1 or height < 1 or width * height > 1_440_000:
        raise ValueError("Source image exceeds the browser pixel budget")
    if len(pixels) != width * height * 4:
        raise ValueError("Invalid RGBA buffer length")
    columns = settings.get("columns", 100)
    font_size = settings.get("font_size", 12)
    if not isinstance(columns, int) or not 20 <= columns <= 180:
        raise ValueError("Columns must be an integer between 20 and 180")
    if font_size not in (8, 12, 16, 20):
        raise ValueError("Unsupported browser font size")
    settings = dict(settings)
    rgba = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 4)
    if settings.get("background_mode") == "source":
        stops = source_palette(rgba[..., [2, 1, 0, 3]].astype(np.float32) / 255)
        settings["background_stops"] = stops.tolist()
        paper = gradient_rows(stops, height)[..., ::-1]
    else:
        paper = None
    if edge_format not in ("uint8", "float32"):
        raise ValueError("Unsupported edge buffer format")
    edge_dtype = np.dtype("<f4") if edge_format == "float32" else np.dtype("uint8")
    if edge_pixels is not None:
        if (
            edge_width < 1
            or edge_height < 1
            or edge_width * edge_height > 384 * 384
            or len(edge_pixels) != edge_width * edge_height * edge_dtype.itemsize
        ):
            raise ValueError("Invalid learned edge map")
        edge_map = (
            np.frombuffer(edge_pixels, edge_dtype)
            .reshape(edge_height, edge_width)
            .astype(np.float32)
        )
        settings["edge_map"] = edge_map if edge_format == "float32" else edge_map / 255
    elif (
        settings.get("use_edge", True)
        and settings.get("edge_method") == "tiny"
        and settings.get("edge_amount") != 0
        and settings.get("edge_sensitivity") != 0
    ):
        raise ValueError("Tiny edge detection requires a model confidence map")
    if settings.get("layout") == "adaptive":
        settings["text_metrics"] = settings.get("adaptive_metrics", {}).get("8")
        if settings["text_metrics"] is None:
            raise ValueError("Adaptive mode requires square font metrics")
    generator = SpriteASCIIGenerator(settings)
    rows = max(
        1, round(height / width * columns * generator.char_width / generator.grid_size)
    )
    if rows * columns * generator.char_width * generator.grid_size > 6_000_000:
        raise ValueError("Output is too large; reduce columns or font size")
    alpha = rgba[..., 3:4].astype(np.float32) / 255
    if paper is None:
        paper = generator.background[::-1]
    rgb = np.rint(rgba[..., :3] * alpha + paper * (1 - alpha)).astype(np.uint8)
    bgr = np.ascontiguousarray(rgb[..., ::-1])
    if settings.get("layout") == "adaptive":
        return adaptive_grid(bgr, settings)
    if settings.get("text_metrics") is not None:
        return generator.generate_grid(bgr)
    result, text = generator.generate_ascii_and_text(bgr)
    output = BytesIO()
    Image.fromarray(result[..., ::-1]).save(output, format="PNG", compress_level=1)
    return {
        "png": output.getvalue(),
        "text": text,
        "width": result.shape[1],
        "height": result.shape[0],
    }
