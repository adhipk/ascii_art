"""Repeatable native comparison against original images, not another render.

Run: PYTHONPATH=src uv run --no-sync python benchmarks/adaptive_quality.py
Outputs default to /tmp/ascii-adaptive-benchmark. No browser dependencies.
"""
import argparse
import csv
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ascii_art.generators.adaptive import adaptive_grid

ROOT = Path(__file__).resolve().parents[1]
CHARACTERS = " .:-=+*#%@/|\\"


def contour_metrics(
    reference: np.ndarray, predicted: np.ndarray, tolerance: float = 2
) -> dict[str, float | None]:
    """Pixel precision/recall at a fixed output-pixel tolerance.

    Empty/empty is perfect; an empty prediction on a nonempty source has zero
    recall. Position is undefined (null) when only one mask contains strokes.
    Position is symmetric nearest-stroke distance, not distance to tile centers.
    """
    reference, predicted = reference.astype(bool), predicted.astype(bool)
    diagonal = float(np.hypot(*reference.shape))

    def distance(mask: np.ndarray) -> np.ndarray:
        if not mask.any():
            return np.full(mask.shape, diagonal, np.float32)
        return cv2.distanceTransform(
            (~mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )

    to_reference, to_predicted = distance(reference), distance(predicted)
    precision = (
        float((to_reference[predicted] <= tolerance).mean())
        if predicted.any()
        else float(not reference.any())
    )
    recall = (
        float((to_predicted[reference] <= tolerance).mean()) if reference.any() else 1.0
    )
    distances = []
    if predicted.any():
        distances.append(float(to_reference[predicted].mean()))
    if reference.any():
        distances.append(float(to_predicted[reference].mean()))
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
        "false_contour_rate": 1 - precision if predicted.any() else 0.0,
        "position_error_px": (
            float(np.mean(distances))
            if reference.any() and predicted.any()
            else (None if reference.any() or predicted.any() else 0.0)
        ),
    }


def calibration() -> dict[str, Any]:
    metrics = {}
    for size in (8, 16, 32):
        font = ImageFont.truetype(
            str(ROOT / "fonts/FiraCode-Regular.ttf"), round(size * 0.8)
        )
        ascent, descent = font.getmetrics()
        advance = font.getlength("M")
        baseline = (size - ascent - descent) / 2 + ascent
        masks = []
        for char in CHARACTERS:
            cell = Image.new("L", (size, size))
            ImageDraw.Draw(cell).text(
                ((size - advance) / 2, baseline), char, font=font, fill=255, anchor="ls"
            )
            masks.append((np.asarray(cell, np.float32) / 255).ravel().tolist())
        metrics[str(size)] = {
            "width": size,
            "height": size,
            "baseline": baseline,
            "advance": advance,
            "fontSize": round(size * 0.8),
            "densities": [float(np.mean(m)) for m in masks[:10]],
            "kernelCharacters": CHARACTERS,
            "masks": masks,
        }
    return metrics


def sample_landscape() -> np.ndarray:
    """Same geometry/colors as web/app.ts; Pillow rasterization differs at edges."""
    top, bottom = np.array([115, 162, 187]), np.array([228, 216, 183])
    t = (np.arange(560) + 0.5) / 560
    image = Image.fromarray(
        np.broadcast_to(
            np.rint(top + t[:, None] * (bottom - top)).astype(np.uint8)[:, None],
            (560, 800, 3),
        ).copy()
    )
    draw = ImageDraw.Draw(image)
    draw.ellipse((542, 72, 638, 168), fill="#f6e9b6")
    draw.polygon(
        [(0, 400), (230, 180), (400, 370), (540, 220), (800, 410)], fill="#667b89"
    )
    draw.polygon(
        [(157, 250), (230, 180), (300, 257), (244, 238), (215, 258)], fill="#f0ece1"
    )
    draw.polygon(
        [(0, 470), (330, 295), (630, 500), (800, 350), (800, 560), (0, 560)],
        fill="#334f4f",
    )
    return np.asarray(image).copy()


def cases() -> dict[str, np.ndarray]:
    gradient = np.broadcast_to(np.linspace(0, 255, 320).astype(np.uint8), (224, 320))
    straight = np.full((224, 320), 255, np.uint8)
    cv2.line(straight, (32, 16), (32, 208), 0, 2)
    cv2.line(straight, (64, 112), (304, 112), 0, 2)
    corner = np.full_like(straight, 255)
    cv2.polylines(
        corner, [np.array([(32, 192), (32, 32), (160, 32), (288, 192)])], False, 0, 2
    )
    curves = np.full_like(straight, 255)
    cv2.circle(curves, (160, 112), 80, 0, 2)
    images = {
        "flat": np.full_like(straight, 160),
        "gradient": gradient,
        "straight": straight,
        "corner-diagonal": corner,
        "circle": curves,
        "landscape": sample_landscape(),
    }
    for name in ("taylor.png", "webb-poster.webp", "spacesuit-patent.webp", "test.png"):
        images[Path(name).stem] = np.asarray(
            Image.open(ROOT / "web/examples" / name).convert("RGB")
        )
    return {
        name: cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if image.ndim == 2 else image
        for name, image in images.items()
    }


def rasterize(
    result: dict[str, Any], metrics: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    height, width = result["height"], result["width"]
    paper = np.frombuffer(result["paper"], np.uint8).reshape(height, 3)
    rendered = (
        np.broadcast_to(paper[:, None, :], (height, width, 3)).astype(np.float32).copy()
    )
    contour = np.zeros((height, width), bool)
    coverage = np.zeros((height, width), np.uint8)
    for tile in result["tiles"]:
        size, x, y, char = (tile[k] for k in ("size", "x", "y", "char"))
        metric = metrics[str(size)]
        alpha = np.array(metric["masks"][CHARACTERS.index(char)], np.float32).reshape(
            size, size
        )
        region = rendered[y : y + size, x : x + size]
        region[:] = (
            region * (1 - alpha[..., None]) + np.array(tile["color"]) * alpha[..., None]
        )
        if char in "-/|\\":
            contour[y : y + size, x : x + size] |= alpha > 0.15
        coverage[y : y + size, x : x + size] += 1
    if not np.all(coverage == 1):
        raise AssertionError("Renderer produced overlapping tiles or gaps")
    return np.rint(rendered).clip(0, 255).astype(np.uint8), contour


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="95dd352")
    parser.add_argument("--columns", type=int, default=80)
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/ascii-adaptive-benchmark")
    )
    parser.add_argument(
        "--image", type=Path, help="Also compare this exact original image"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    baseline_source = subprocess.check_output(
        ["git", "show", f"{args.baseline}:src/ascii_art/generators/adaptive.py"],
        cwd=ROOT,
        text=True,
    )
    namespace: dict[str, Any] = {}
    exec(compile(baseline_source, "baseline_adaptive.py", "exec"), namespace)
    current_source = (ROOT / "src/ascii_art/generators/adaptive.py").read_bytes()
    metrics, inputs = calibration(), cases()
    if args.image:
        inputs[args.image.stem] = np.asarray(Image.open(args.image).convert("RGB"))
    records = []
    for name, original in inputs.items():
        # Both detector on/off; monochrome isolates geometry from color adjustment.
        for edges in (False, True):
            settings = {
                "columns": args.columns,
                "numpy": np.__version__,
                "opencv": cv2.__version__,
                "panel_order": ["original", "baseline", "current"],
                "primary_edges": "Canny on complete rendered RGB, same settings as original",
                "directional_edges": "Alpha > 0.15 of characters - / | and backslash only",
                "landscape_note": "Recreated from web/app.ts geometry/colors with Pillow; not browser-identical rasterization",
                "use_edge": edges,
                "edge_method": "acerola",
                "edge_amount": 100,
                "edge_min_pixels": 8,
                "use_color": False,
                "background_mode": "source",
                "adaptive_metrics": metrics,
            }
            panels = []
            for version, function in (
                ("baseline", namespace["adaptive_grid"]),
                ("current", adaptive_grid),
            ):
                start = time.perf_counter()
                result = function(cv2.cvtColor(original, cv2.COLOR_RGB2BGR), settings)
                elapsed = (time.perf_counter() - start) * 1000
                rendered, contour = rasterize(result, metrics)
                source = cv2.resize(
                    original,
                    (result["width"], result["height"]),
                    interpolation=cv2.INTER_AREA,
                )
                gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
                # Independent Canny reference, not the encoder's DoG/direction map.
                reference = cv2.Canny(cv2.GaussianBlur(gray, (0, 0), 1), 50, 100) > 0
                rendered_edges = (
                    cv2.Canny(
                        cv2.GaussianBlur(
                            cv2.cvtColor(rendered, cv2.COLOR_RGB2GRAY), (0, 0), 1
                        ),
                        50,
                        100,
                    )
                    > 0
                )
                quality = contour_metrics(reference, rendered_edges)
                directional_quality = {
                    "directional_" + key: value
                    for key, value in contour_metrics(reference, contour).items()
                }
                source_luma = source @ np.array([0.299, 0.587, 0.114]) / 255
                output_luma = rendered @ np.array([0.299, 0.587, 0.114]) / 255
                tone = float(
                    np.mean(
                        np.abs(
                            cv2.GaussianBlur(source_luma, (0, 0), 8)
                            - cv2.GaussianBlur(output_luma, (0, 0), 8)
                        )
                    )
                )
                record = {
                    "image": name,
                    "source_sha256": hashlib.sha256(original.tobytes()).hexdigest(),
                    "output_width": result["width"],
                    "output_height": result["height"],
                    "edges": edges,
                    "version": version,
                    **quality,
                    **directional_quality,
                    "tone_mae": tone,
                    "glyphs": sum(t["char"] != " " for t in result["tiles"]),
                    "glyph_histogram": dict(
                        Counter(t["char"] for t in result["tiles"] if t["char"] != " ")
                    ),
                    "ink_coverage": sum(
                        sum(
                            metrics[str(t["size"])]["masks"][
                                CHARACTERS.index(t["char"])
                            ]
                        )
                        for t in result["tiles"]
                    )
                    / (result["width"] * result["height"]),
                    "ms": elapsed,
                    "sizes": {
                        str(s): sum(t["size"] == s for t in result["tiles"])
                        for s in (8, 16, 32)
                    },
                }
                records.append(record)
                panels.append(rendered)
            # Original | committed baseline | current, at identical resolution.
            panel = np.concatenate([source, *panels], axis=1)
            Image.fromarray(panel).save(args.output / f"{name}-edges-{int(edges)}.png")
    report = {
        "baseline_commit": subprocess.check_output(
            ["git", "rev-parse", args.baseline], cwd=ROOT, text=True
        ).strip(),
        "current_sha256": hashlib.sha256(current_source).hexdigest(),
        "columns": args.columns,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "panel_order": ["original", "baseline", "current"],
        "primary_edges": "Canny on complete rendered RGB, same settings as original",
        "directional_edges": "Alpha > 0.15 of characters - / | and backslash only",
        "landscape_note": "Recreated from web/app.ts geometry/colors with Pillow; not browser-identical rasterization",
        "font": "fonts/FiraCode-Regular.ttf",
        "font_sha256": hashlib.sha256(
            (ROOT / "fonts/FiraCode-Regular.ttf").read_bytes()
        ).hexdigest(),
        "reference": "Original RGB; Canny 50/100 after sigma=1 blur; contour tolerance=2 output pixels; tone blur sigma=8",
        "records": records,
    }
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.output / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(
        f"Wrote {len(records)} measurements and {len(records)//2} comparison panels to {args.output}"
    )


if __name__ == "__main__":
    main()
