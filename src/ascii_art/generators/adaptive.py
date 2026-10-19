"""Aligned square-cell typography, with detail-driven quadtree subdivision."""
from typing import Any, Mapping

import cv2
import numpy as np

from ascii_art.generators.sprite_generator import SpriteASCIIGenerator


def adaptive_grid(image: np.ndarray, settings: Mapping[str, Any]) -> dict[str, Any]:
    """Render actual glyphs in nonoverlapping 8, 16 and 32 pixel square cells.

    Variance includes all analysis pixels, not only cell means: small features
    must not disappear simply because adjacent averages happen to be equal.
    """
    metrics = settings.get("adaptive_metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("Adaptive mode requires font metrics for 8, 16 and 32")
    for size in (8, 16, 32):
        metric = metrics.get(str(size))
        if (
            not isinstance(metric, Mapping)
            or metric.get("width") != size
            or metric.get("height") != size
        ):
            raise ValueError("Adaptive metrics must describe square 8, 16 and 32 cells")
        # Reuse the generator's complete coverage/size validation for each atlas.
        SpriteASCIIGenerator({**settings, "text_metrics": metric})
    renderer = SpriteASCIIGenerator({**settings, "text_metrics": metrics["8"]})
    result = renderer.generate_grid(image)
    gray, source, rows, cols = renderer._prepare(image)
    if gray.shape != source.shape[:2]:
        gray = cv2.resize(gray, (cols * 8, rows * 8), interpolation=cv2.INTER_AREA)
    indices = np.array([list(line) for line in result["text"].split("\n")])
    edge = np.isin(indices, list(renderer.edge_characters))
    colors = np.frombuffer(result["colors"], np.uint8).reshape(rows, cols, 3)
    paper = np.frombuffer(result["paper"], np.uint8).reshape(rows * 8, 3)
    ink = float(
        renderer._ink(renderer._paper(source)) @ np.array([0.114, 0.587, 0.299]) / 255
    )
    sums, squares = cv2.integral2(gray, sdepth=cv2.CV_64F, sqdepth=cv2.CV_64F)
    # Independently inspect edges even when the contour overlay is disabled.
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0, scale=1 / 32)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1, scale=1 / 32)
    tensors = [cv2.integral(v, sdepth=cv2.CV_64F) for v in (gx * gx, gx * gy, gy * gy)]
    tiles: list[dict[str, Any]] = []

    def visit(x: int, y: int, cells: int) -> None:
        size = cells * 8
        if cells == 1:
            tiles.append(
                {
                    "x": x * 8,
                    "y": y * 8,
                    "size": 8,
                    "char": str(indices[y, x]),
                    "color": colors[y, x].tolist(),
                }
            )
            return
        inside = x + cells <= cols and y + cells <= rows
        if inside:
            x0, y0, x1, y1 = x * 8, y * 8, (x + cells) * 8, (y + cells) * 8
            mean = (
                sums[y1, x1] - sums[y0, x1] - sums[y1, x0] + sums[y0, x0]
            ) / size**2
            square = (
                squares[y1, x1] - squares[y0, x1] - squares[y1, x0] + squares[y0, x0]
            ) / size**2
            deviation = np.sqrt(max(0, square - mean**2))
            contour = edge[y : y + cells, x : x + cells].any()
            xx, xy, yy = [
                (v[y1, x1] - v[y0, x1] - v[y1, x0] + v[y0, x0]) / size**2
                for v in tensors
            ]
            energy = xx + yy
            coherence = np.sqrt((xx - yy) ** 2 + 4 * xy**2) / max(energy, 1e-12)
            # Corners / competing directions are more expensive to coarsen
            # than one coherent stroke. Absolute units avoid amplifying noise.
            edge_detail = np.sqrt(max(0, energy)) * (2 - coherence)
            # Conservative larger cells; progressively tolerate more variation
            # as a patch gets smaller. These are absolute normalized units.
            merge = (
                not contour
                and deviation <= (0.045 if cells == 4 else 0.07)
                and edge_detail <= (0.012 if cells == 4 else 0.02)
            )
            if merge:
                char = str(indices[y, x])
                if cells > 1:
                    densities = np.asarray(metrics[str(size)]["densities"], np.float32)
                    densities = (densities - densities.min()) / np.ptp(densities)
                    background = float(
                        paper[y0:y1].mean(axis=0)
                        @ np.array([0.299, 0.587, 0.114])
                        / 255
                    )
                    coverage = np.clip(
                        (mean - background)
                        / (ink - background if abs(ink - background) > 1e-6 else 1),
                        0,
                        1,
                    )
                    char = renderer.characters[
                        int(np.abs(densities - coverage).argmin())
                    ]
                color = (
                    np.rint(colors[y : y + cells, x : x + cells].mean(axis=(0, 1)))
                    .astype(np.uint8)
                    .tolist()
                )
                tiles.append(
                    {"x": x * 8, "y": y * 8, "size": size, "char": char, "color": color}
                )
                return
        if cells > 1:
            half = cells // 2
            for dy in (0, half):
                for dx in (0, half):
                    if x + dx < cols and y + dy < rows:
                        visit(x + dx, y + dy, half)

    for y in range(0, rows, 4):
        for x in range(0, cols, 4):
            visit(x, y, 4)
    result["tiles"] = tiles
    result["tileMetrics"] = dict(metrics)
    return result
