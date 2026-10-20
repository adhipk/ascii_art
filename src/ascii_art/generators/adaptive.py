"""Exact quadtree optimization over calibrated, variable-size glyph kernels."""
from typing import Any, Mapping

import cv2
import numpy as np

from ascii_art.filters.acerola import edge_pass

from ascii_art.generators.sprite_generator import SpriteASCIIGenerator


def edge_reconstruction_costs(target: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Symmetric contour distance with one-pixel tolerance, in pixel units.

    Missing target strokes and extra rendered strokes both incur cost. Distances
    are capped at four pixels so a distant outlier cannot dominate a whole tile.
    Empty masks have a finite missing-stroke cost, including at image borders.
    """
    target = target.astype(bool)
    predicted = masks > 0.15

    def distances(strokes: np.ndarray) -> np.ndarray:
        if not strokes.any():
            return np.full(strokes.shape, 16.0, np.float32)
        distance = cv2.distanceTransform((~strokes).astype(np.uint8), cv2.DIST_L2, 3)
        return np.minimum(np.maximum(distance - 1, 0) ** 2, 16)

    target_distance = distances(target)
    return np.array(
        [
            float(distances(stroke)[target].sum() + target_distance[stroke].sum())
            for stroke in predicted
        ]
    )


def adaptive_grid(image: np.ndarray, settings: Mapping[str, Any]) -> dict[str, Any]:
    """Minimize summed RGB squared error plus a penalty per visible glyph.

    Error uses RGB values in [0, 1], averaged over channels, summed over pixels.
    The default tone objective adds 4 * area * squared normalized density error.
    Use adaptive_objective="pixels" for literal RGB reconstruction alone.
    ``adaptive_penalty`` defaults to 1 pixel-error unit per nonblank glyph.
    The optimum is exact within the aligned 8/16/32 tree and measured dictionary,
    with the deterministic per-region ink colors used by the renderer.
    Detected contour regions use 4 * symmetric edge distance plus glyph penalty.
    """
    metrics = settings.get("adaptive_metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("Adaptive mode requires font metrics for 8, 16 and 32")
    penalty = settings.get("adaptive_penalty", 1.0)
    if isinstance(penalty, bool) or not isinstance(penalty, (int, float)):
        raise ValueError("adaptive_penalty must be finite and nonnegative")
    if not np.isfinite(penalty) or penalty < 0:
        raise ValueError("adaptive_penalty must be finite and nonnegative")
    objective_mode = settings.get("adaptive_objective", "tone")
    if objective_mode not in ("tone", "pixels"):
        raise ValueError("adaptive_objective must be tone or pixels")
    dictionaries: dict[int, tuple[str, np.ndarray]] = {}
    for size in (8, 16, 32):
        metric = metrics.get(str(size))
        if (
            not isinstance(metric, Mapping)
            or metric.get("width") != size
            or metric.get("height") != size
        ):
            raise ValueError("Adaptive metrics must describe square 8, 16 and 32 cells")
        SpriteASCIIGenerator({**settings, "text_metrics": metric})
        chars = metric.get("kernelCharacters")
        masks = np.asarray(metric.get("masks"), dtype=np.float32)
        if (
            not isinstance(chars, str)
            or not chars
            or len(set(chars)) != len(chars)
            or " " not in chars
            or any(not c.isprintable() or ord(c) > 126 for c in chars)
            or masks.shape != (len(chars), size * size)
            or not np.isfinite(masks).all()
            or np.any((masks < 0) | (masks > 1))
            or np.any(masks[chars.index(" ")] != 0)
        ):
            raise ValueError(
                "Adaptive metrics require calibrated glyph masks and kernelCharacters"
            )
        dictionaries[size] = (chars, masks.reshape(-1, size, size))

    renderer = SpriteASCIIGenerator({**settings, "text_metrics": metrics["8"]})
    result = renderer.generate_grid(image)
    gray, source, rows, cols = renderer._prepare(image)
    base_indices = renderer._indices(gray, rows, cols, renderer._paper(source))
    contours = base_indices >= renderer.num_ascii_chars
    base_text = np.array([list(line) for line in result["text"].split("\n")])
    edge_pixels = np.zeros(gray.shape, bool)
    if contours.any():
        if renderer.settings["edge_method"] == "acerola":
            edge_pixels = (
                edge_pass(gray, rows, cols, renderer.settings)["directions"] >= 0
            )
        else:
            gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
            gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
            edge_pixels = np.hypot(gx, gy) > 0.01
        # Preserve detector selection/minimum votes without pooling its geometry.
        edge_pixels &= np.repeat(np.repeat(contours, 8, axis=0), 8, axis=1)
    # Match the RGB paper and integer ink colors consumed by SVG and PNG.
    target = source[..., ::-1]
    paper = np.frombuffer(result["paper"], np.uint8).reshape(rows * 8, 3) / 255.0
    colors = np.frombuffer(result["colors"], np.uint8).reshape(rows, cols, 3)

    mono_ink = float(
        renderer._ink(renderer._paper(source)) @ np.array([0.114, 0.587, 0.299]) / 255
    )
    paper_luma = paper @ np.array([0.299, 0.587, 0.114])

    def visit(x: int, y: int, cells: int) -> tuple[float, list[dict[str, Any]]]:
        size = cells * 8
        inside = x + cells <= cols and y + cells <= rows
        best_cost = float("inf")
        best_tiles: list[dict[str, Any]] = []
        has_contour = bool(contours[y : y + cells, x : x + cells].any())
        if inside:
            chars, masks = dictionaries[size]
            color = np.rint(colors[y : y + cells, x : x + cells].mean(axis=(0, 1)))
            x0, y0 = x * 8, y * 8
            background = paper[y0 : y0 + size, None, :]
            delta = color / 255.0 - background
            residual = target[y0 : y0 + size, x0 : x0 + size] - background
            # Expand ||residual - alpha * delta||² without a K×H×W×RGB tensor.
            errors = (
                np.sum(residual * residual) / 3
                - 2
                * np.einsum("khw,hw->k", masks, np.sum(residual * delta, axis=2))
                / 3
                + np.einsum(
                    "khw,hw->k",
                    masks * masks,
                    np.broadcast_to(np.sum(delta * delta, axis=2), (size, size)),
                )
                / 3
            )
            if objective_mode == "tone":
                # Photographic tones span [0, 1], whereas font coverage typically
                # spans only [0, ~0.2]. Calibrate these ranges before ranking.
                # Use monochrome polarity even in color mode: source-colored ink
                # must not make every patch appear to demand full coverage.
                local_paper = paper_luma[y0 : y0 + size, None]
                contrast = mono_ink - local_paper
                coverage = np.clip(
                    (gray[y0 : y0 + size, x0 : x0 + size] - local_paper)
                    / np.where(np.abs(contrast) > 1e-6, contrast, 1),
                    0,
                    1,
                ).mean()
                densities = masks.mean(axis=(1, 2))
                normalized = densities / max(float(densities.max()), 1e-8)
                errors += 4 * size**2 * (normalized - coverage) ** 2
            costs = np.maximum(errors, 0) + penalty * np.array(
                [c != " " for c in chars]
            )
            if not has_contour and objective_mode == "tone":
                # Shape-only kernels must never become extra brightness levels.
                costs[np.array([c not in renderer.characters for c in chars])] = np.inf
            if has_contour:
                directional = np.array([c in renderer.edge_characters for c in chars])
                if not directional.any():
                    raise ValueError(
                        "Adaptive dictionary is missing directional contour glyphs"
                    )
                # A contour region competes using full-resolution edge geometry,
                # rather than photographic brightness or forced base-grid glyphs.
                costs = (
                    4
                    * edge_reconstruction_costs(
                        edge_pixels[y0 : y0 + size, x0 : x0 + size], masks
                    )
                    + penalty
                )
                costs[~directional] = np.inf
                if cells == 1:
                    forced = str(base_text[y, x])
                    if forced not in chars:
                        raise ValueError(
                            "Adaptive dictionary is missing a detected contour glyph"
                        )
                    costs[np.array([c != forced for c in chars])] = np.inf
                else:
                    # Mixed directions indicate a bend/junction, so preserve
                    # the finer detector decisions rather than invent a stroke.
                    detected = set(
                        base_text[y : y + cells, x : x + cells][
                            contours[y : y + cells, x : x + cells]
                        ].tolist()
                    )
                    costs[np.array([c not in detected for c in chars])] = np.inf
                    if len(detected) != 1:
                        costs[:] = np.inf
            index = int(costs.argmin())
            best_cost = float(costs[index])
            best_tiles = [
                {
                    "x": x0,
                    "y": y0,
                    "size": size,
                    "char": chars[index],
                    "color": color.astype(np.uint8).tolist(),
                }
            ]
        if cells > 1:
            half = cells // 2
            child_cost = 0.0
            child_tiles: list[dict[str, Any]] = []
            for dy in (0, half):
                for dx in (0, half):
                    if x + dx < cols and y + dy < rows:
                        cost, tiles = visit(x + dx, y + dy, half)
                        child_cost += cost
                        child_tiles.extend(tiles)
            # Prefer a larger region when both representations cost the same.
            if child_cost < best_cost - 1e-9:
                return child_cost, child_tiles
        return best_cost, best_tiles

    tiles: list[dict[str, Any]] = []
    objective = 0.0
    for y in range(0, rows, 4):
        for x in range(0, cols, 4):
            cost, selected = visit(x, y, 4)
            objective += cost
            tiles.extend(selected)
    result["tiles"] = tiles
    # Pixel masks are analysis-only; avoid returning the large payload to the UI.
    result["tileMetrics"] = {
        key: {k: v for k, v in value.items() if k not in ("masks", "kernelCharacters")}
        for key, value in metrics.items()
    }
    result["adaptiveCost"] = objective
    return result
