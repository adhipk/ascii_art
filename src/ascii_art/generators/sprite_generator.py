"""Vectorized, font-calibrated ASCII rendering with DoG-guided edge glyphs."""
from functools import lru_cache
from math import ceil
from typing import Any, Mapping

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ascii_art.filters.dog_filter import DoGFilter, blur, gradient, grayscale
from ascii_art.filters.edge_selection import select_edges
from ascii_art.filters.acerola import edge_pass
from ascii_art.utils.background import gradient_rows, source_palette
from ascii_art.utils.color import readable_ink


@lru_cache(maxsize=32)
def glyph_atlas(
    font_path: str | None, size: int, chars: str
) -> tuple[np.ndarray, int, int]:
    """Cache by all rasterization inputs, in memory; never write into installation."""
    font = (
        ImageFont.truetype(font_path, size)
        if font_path
        else ImageFont.load_default(size=size)
    )
    bounds = [font.getbbox(char) for char in chars]
    width = max(1, ceil(max(font.getlength(char) for char in chars)))
    top = min(0, min(b[1] for b in bounds))
    height = max(size, max(b[3] for b in bounds) - top)
    atlas = []
    for char in chars:
        tile = Image.new("L", (width, height))
        ImageDraw.Draw(tile).text((0, -top), char, font=font, fill=255)
        atlas.append(np.asarray(tile, dtype=np.float32) / 255)
    masks = np.stack(atlas)
    masks.setflags(write=False)
    return masks, width, height


def color(value: str) -> np.ndarray:
    """Parse an RGB hex color into OpenCV BGR."""
    if not isinstance(value, str) or len(value) != 7 or not value.startswith("#"):
        raise ValueError("colors must be #RRGGBB strings")
    try:
        return np.array(
            [int(value[i : i + 2], 16) for i in (5, 3, 1)], dtype=np.float32
        )
    except ValueError as exc:
        raise ValueError("colors must be #RRGGBB strings") from exc


class SpriteASCIIGenerator:
    """Accept BGR/BGRA uint8 or normalized floats; return a BGR uint8 image.

    Character cell dimensions come from the selected font. Edge thresholds are
    percentiles (0–100); larger values select fewer edge cells.
    """

    def __init__(self, settings: Mapping[str, Any] | None = None) -> None:
        self.settings = {
            "use_edge": True,
            "edge_threshold": 70.0,
            "edge_sensitivity": None,
            "edge_amount": None,
            "edge_method": "tensor",
            "edge_min_pixels": 8,
            "use_color": False,
            "font_size": 12,
            "font_path": None,
            "text_metrics": None,
            "characters": " .:-=+*#%@",
            "foreground": "#242424",
            "background": "#FFFFFF",
            "background_mode": "solid",
            "background_stops": None,
            "columns": None,
            "edge_coherence": 0.3,
            "edge_strength": 0.01,
            "edge_map": None,
            "sigma_e": 1.0,
            "k": 1.6,
            "sharpness": 5.0,
            "white_point": 128,
            **(settings or {}),
        }
        s = self.settings
        if (
            isinstance(s["font_size"], bool)
            or not isinstance(s["font_size"], int)
            or s["font_size"] < 2
        ):
            raise ValueError("font_size must be an integer >= 2")
        if s["columns"] is not None and (
            isinstance(s["columns"], bool)
            or not isinstance(s["columns"], int)
            or s["columns"] < 1
        ):
            raise ValueError("columns must be a positive integer")
        for name, upper in (
            ("edge_threshold", 100),
            ("edge_coherence", 1),
            ("edge_strength", 1),
        ):
            if not np.isfinite(s[name]) or not 0 <= s[name] <= upper:
                raise ValueError(f"{name} must be in [0, {upper}]")
        if s["edge_sensitivity"] is not None and (
            not isinstance(s["edge_sensitivity"], (int, float))
            or not np.isfinite(s["edge_sensitivity"])
            or not 0 <= s["edge_sensitivity"] <= 100
        ):
            raise ValueError("edge_sensitivity must be in [0, 100]")
        if s["edge_amount"] is not None and (
            not isinstance(s["edge_amount"], (int, float))
            or not np.isfinite(s["edge_amount"])
            or not 0 <= s["edge_amount"] <= 100
        ):
            raise ValueError("edge_amount must be in [0, 100]")
        if s["background_mode"] not in ("solid", "source"):
            raise ValueError("background_mode must be solid or source")
        self.background_stops = None
        if s["background_stops"] is not None:
            stops = np.asarray(s["background_stops"], dtype=np.float32)
            if (
                stops.shape != (2, 3)
                or not np.isfinite(stops).all()
                or stops.min() < 0
                or stops.max() > 255
            ):
                raise ValueError("background_stops must be two BGR colors in [0, 255]")
            self.background_stops = stops.copy()
        chars = s["characters"]
        if (
            not isinstance(chars, str)
            or len(set(chars)) < 2
            or not all(c.isprintable() and c.isascii() for c in chars)
        ):
            raise ValueError(
                "characters must contain at least two distinct printable ASCII characters"
            )
        self.edge_map = None
        if s["edge_map"] is not None:
            edge_map = np.asarray(s["edge_map"], dtype=np.float32)
            if (
                edge_map.ndim != 2
                or edge_map.size == 0
                or not np.isfinite(edge_map).all()
                or edge_map.min() < 0
                or edge_map.max() > 1
            ):
                raise ValueError(
                    "edge_map must contain finite confidence values in [0, 1]"
                )
            self.edge_map = edge_map.copy()
        self.characters = "".join(dict.fromkeys(chars))
        self.edge_characters = "-/|\\"
        self.num_ascii_chars = len(self.characters)
        self.num_edge_chars = 4
        metrics = s["text_metrics"]
        if metrics is None:
            self.char_masks, self.char_width, self.grid_size = glyph_atlas(
                s["font_path"], s["font_size"], self.characters + self.edge_characters
            )
            densities = self.char_masks[: self.num_ascii_chars].mean(axis=(1, 2))
        else:
            if not isinstance(metrics, Mapping):
                raise ValueError("text_metrics must be a mapping")
            width, height = metrics.get("width"), metrics.get("height")
            densities = np.asarray(metrics.get("densities"), dtype=np.float32)
            if (
                not isinstance(width, int)
                or not isinstance(height, int)
                or not 1 <= width <= 64
                or not 2 <= height <= 96
                or densities.shape != (self.num_ascii_chars,)
                or not np.isfinite(densities).all()
                or densities.min() < 0
                or densities.max() > 1
                or np.ptp(densities) <= 1e-8
            ):
                raise ValueError("Invalid browser text metrics")
            self.char_masks = None
            self.char_width, self.grid_size = width, height
        self.foreground, self.background = color(s["foreground"]), color(
            s["background"]
        )
        self._density_order = np.argsort(densities, kind="stable")
        ordered = densities[self._density_order]
        self._densities = (ordered - ordered.min()) / max(float(np.ptp(ordered)), 1e-8)
        self.dog = DoGFilter(s)

    def _prepare(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
        gray = grayscale(image)
        h, w = gray.shape
        cols = self.settings["columns"] or max(1, w // self.char_width)
        rows = max(1, round(h / w * cols * self.char_width / self.grid_size))
        target = (cols * self.char_width, rows * self.grid_size)
        analysis_target = (
            (cols * 8, rows * 8)
            if self.settings["edge_method"] == "acerola"
            else target
        )
        gray = cv2.resize(gray, analysis_target, interpolation=cv2.INTER_AREA)
        source = image.astype(np.float32)
        if image.dtype == np.uint8:
            source /= 255
        if source.ndim == 2:
            source = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
        source = cv2.resize(source[..., :3], target, interpolation=cv2.INTER_AREA)
        return gray, source, rows, cols

    def _cells(self, values: np.ndarray, rows: int, cols: int) -> np.ndarray:
        return values.reshape(rows, self.grid_size, cols, self.char_width).mean(
            axis=(1, 3)
        )

    def _paper(self, source: np.ndarray) -> np.ndarray:
        if self.settings["background_mode"] == "solid":
            return self.background
        stops = (
            self.background_stops
            if self.background_stops is not None
            else source_palette(source)
        )
        return gradient_rows(stops, source.shape[0])

    def _ink(self, paper: np.ndarray) -> np.ndarray:
        if self.settings["background_mode"] == "solid":
            return self.foreground
        return np.full(3, 255 if paper.mean() < 128 else 36, np.float32)

    def _indices(
        self, gray: np.ndarray, rows: int, cols: int, paper: np.ndarray | None = None
    ) -> np.ndarray:
        reference = self.settings["edge_method"] == "acerola"
        luminance = (
            gray.reshape(rows, 8, cols, 8).mean(axis=(1, 3))
            if reference
            else self._cells(gray, rows, cols)
        )
        paper = self.background if paper is None else paper
        ink = self._ink(paper)
        fg = float(ink @ np.array([0.114, 0.587, 0.299]) / 255)
        bg = paper @ np.array([0.114, 0.587, 0.299]) / 255
        if paper.ndim == 3:
            bg = bg.reshape(rows, self.grid_size, 1).mean(axis=1)
        delta = fg - bg
        coverage = np.clip(
            (luminance - bg) / np.where(np.abs(delta) > 1e-6, delta, 1), 0, 1
        )
        indices = self._density_order[
            np.abs(coverage[..., None] - self._densities).argmin(axis=-1)
        ]
        sensitivity = self.settings["edge_sensitivity"]
        amount = self.settings["edge_amount"]
        if (
            self.settings["use_edge"]
            and amount != 0
            and self.settings["edge_method"] == "acerola"
        ):
            # Reference pipeline: threshold DoG first, then vote per pixel.
            # This deliberately bypasses tensor pooling and tonal emphasis.
            winners = edge_pass(gray, rows, cols, self.settings)["tiles"]
            retained = winners >= 0
            lookup = np.array([2, 0, 3, 1])  # |, -, backslash, slash
            indices[retained] = self.num_ascii_chars + lookup[winners[retained]]
            return indices
        if self.settings["use_edge"] and sensitivity != 0 and amount != 0:
            # Differentiate smooth, unthresholded XDoG to avoid binary contour noise.
            confidence = None
            if self.edge_map is not None:
                confidence = cv2.resize(
                    self.edge_map,
                    (gray.shape[1], gray.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
                gx, gy = gradient(blur(gray, 0.7))
                gx, gy = gx * confidence, gy * confidence
            else:
                gx, gy = gradient(blur(self.dog.response(gray), 0.7))
            xx, xy, yy = (
                self._cells(v, rows, cols) for v in (gx * gx, gx * gy, gy * gy)
            )
            energy = xx + yy
            coherence = np.sqrt((xx - yy) ** 2 + 4 * xy * xy) / np.maximum(
                energy, 1e-10
            )
            strength = np.sqrt(energy)
            if confidence is not None:
                strength = confidence.reshape(
                    rows, self.grid_size, cols, self.char_width
                ).max(axis=(1, 3))
            valid = (strength > self.settings["edge_strength"]) & (
                coherence >= self.settings["edge_coherence"]
            )
            if sensitivity is not None:
                if confidence is not None:
                    # Select at the model's native resolution, independent of the
                    # current font/grid. Then project retained contours to cells.
                    retained = select_edges(self.edge_map, sensitivity)
                    retained = cv2.resize(
                        retained.astype(np.uint8),
                        (gray.shape[1], gray.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    mask = (
                        retained.reshape(rows, self.grid_size, cols, self.char_width)
                        .max(axis=(1, 3))
                        .astype(bool)
                        & valid
                    )
                else:
                    # Fixed derivative scale; never normalize by each image's max.
                    mask = select_edges(
                        np.clip(strength / 0.15, 0, 1),
                        sensitivity,
                        eligible=valid,
                        min_size=2,
                    )
            else:
                # Preserve legacy percentile settings for existing Python callers.
                candidates = strength > self.settings["edge_strength"]
                threshold = (
                    np.percentile(strength[candidates], self.settings["edge_threshold"])
                    if np.any(candidates)
                    else np.inf
                )
                mask = valid & (strength >= threshold)
            if amount is not None:
                # Gentle local contrast within the density ramp. Never substitute
                # orientation glyphs: they destroy the underlying image's tone.
                detail = luminance - cv2.GaussianBlur(luminance, (0, 0), 1.0)
                weight = (
                    strength
                    if confidence is not None
                    else np.clip(strength / 0.15, 0, 1)
                )
                boost = (
                    np.clip(detail * 0.75, -0.08, 0.08) * (amount / 100) * weight * mask
                )
                adjusted = np.clip(coverage + np.sign(delta) * boost, 0, 1)
                return self._density_order[
                    np.abs(adjusted[..., None] - self._densities).argmin(axis=-1)
                ]
            if np.any(mask):
                # Tensor angle is a normal. Rotate to tangent; image y points down.
                tangent = (0.5 * np.arctan2(2 * xy, xx - yy) + np.pi / 2) % np.pi
                bins = np.floor(tangent / (np.pi / 4) + 0.5).astype(int) % 4
                edge_lookup = np.array([0, 3, 2, 1])  # -, backslash, |, slash
                indices[mask] = self.num_ascii_chars + edge_lookup[bins[mask]]
        return indices

    def _text_from_indices(self, indices: np.ndarray) -> str:
        chars = np.array(list(self.characters + self.edge_characters))
        return "\n".join("".join(row) for row in chars[indices])

    def generate_text(self, img: np.ndarray) -> str:
        """Return the same character grid used by the image renderer."""
        gray, source, rows, cols = self._prepare(img)
        return self._text_from_indices(
            self._indices(gray, rows, cols, self._paper(source))
        )

    def generate_ascii(self, img: np.ndarray) -> np.ndarray:
        gray, source, rows, cols = self._prepare(img)
        paper = self._paper(source)
        return self._rasterize(self._indices(gray, rows, cols, paper), source, paper)

    def generate_ascii_and_text(self, img: np.ndarray) -> tuple[np.ndarray, str]:
        """Generate both exports with a single resampling and filtering pass."""
        gray, source, rows, cols = self._prepare(img)
        paper = self._paper(source)
        indices = self._indices(gray, rows, cols, paper)
        return self._rasterize(indices, source, paper), self._text_from_indices(indices)

    def _rasterize(
        self, indices: np.ndarray, source: np.ndarray, paper: np.ndarray
    ) -> np.ndarray:
        if self.char_masks is None:
            raise ValueError(
                "Browser text metrics require generate_grid, not mask rendering"
            )
        rows, cols = indices.shape
        alpha = (
            self.char_masks[indices]
            .transpose(0, 2, 1, 3)
            .reshape(source.shape[:2])[..., None]
        )
        foreground = self._ink(paper)
        if self.settings["use_color"]:
            cell_colors = self._cell_colors(source, paper, indices)
            foreground = np.repeat(
                np.repeat(cell_colors, self.grid_size, axis=0), self.char_width, axis=1
            )
        return np.rint(alpha * foreground + (1 - alpha) * paper).astype(np.uint8)

    def _cell_colors(
        self, source: np.ndarray, paper: np.ndarray, indices: np.ndarray
    ) -> np.ndarray:
        rows, cols = indices.shape
        if self.settings["use_color"]:
            colors = (
                source.reshape(rows, self.grid_size, cols, self.char_width, 3).mean(
                    axis=(1, 3)
                )
                * 255
            )
            cell_paper = (
                paper
                if paper.ndim == 1
                else paper.reshape(rows, self.grid_size, 1, 3).mean(axis=1)
            )
            return readable_ink(colors, cell_paper)
        return np.broadcast_to(self._ink(paper), (rows, cols, 3)).copy()

    def generate_grid(self, img: np.ndarray) -> dict[str, Any]:
        """Return actual text and per-cell RGB ink, without stamping a glyph mask."""
        gray, source, rows, cols = self._prepare(img)
        paper = self._paper(source)
        indices = self._indices(gray, rows, cols, paper)
        colors = self._cell_colors(source, paper, indices)
        paper_rows = np.broadcast_to(paper, (source.shape[0], 1, 3))
        return {
            "text": self._text_from_indices(indices),
            "colors": np.rint(colors[..., ::-1]).astype(np.uint8).tobytes(),
            "paper": np.rint(paper_rows[:, 0, ::-1]).astype(np.uint8).tobytes(),
            "columns": cols,
            "rows": rows,
            "cellWidth": self.char_width,
            "cellHeight": self.grid_size,
            "width": source.shape[1],
            "height": source.shape[0],
        }


def generate_ascii_image_sprite(
    img: np.ndarray, settings: Mapping[str, Any] | None = None
) -> np.ndarray:
    """Generate an ASCII image using a reusable renderer's default pipeline."""
    return SpriteASCIIGenerator(settings).generate_ascii(img)
