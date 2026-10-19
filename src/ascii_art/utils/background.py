"""Restrained image-derived paper gradients, interpolated in CIE Lab."""
import cv2
import numpy as np


def source_palette(source: np.ndarray) -> np.ndarray:
    """Return two BGR stops in 0–255 units from normalized BGR/BGRA pixels.

    This estimates a pleasing image palette, not a calibrated illuminant. Alpha
    weights sampling so invisible pixels never determine the paper colors.
    """
    h, w = source.shape[:2]
    scale = min(1.0, 128 / max(h, w))
    sampled = source
    if source.shape[2] == 4:
        sampled = source.copy()
        sampled[..., :3] *= sampled[..., 3:4]
    small = cv2.resize(
        sampled,
        (max(1, round(w * scale)), max(1, round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )
    bgr = small[..., :3]
    weights = (
        small[..., 3] if small.shape[2] == 4 else np.ones(small.shape[:2], np.float32)
    )
    if source.shape[2] == 4:
        bgr = bgr / np.maximum(weights[..., None], 1e-8)
    visible = weights > 0.1
    if not np.any(visible):
        lab_stops = np.array([[96, 0, 0], [89, 0, 0]], np.float32)
    else:
        lab = cv2.cvtColor(np.ascontiguousarray(bgr), cv2.COLOR_BGR2LAB)[visible]
        weights = weights[visible]
        split = float(np.median(lab[:, 0]))
        dark = split < 48
        groups = (lab[:, 0] <= split, lab[:, 0] > split)
        chroma = []
        for group in groups:
            if not np.any(group):
                group = np.ones(len(lab), dtype=bool)
            # Limit color dominance and keep a coherent, subdued backdrop.
            mean = np.average(lab[group, 1:], axis=0, weights=weights[group])
            mean *= min(1.0, 16 / max(float(np.linalg.norm(mean)), 1e-6))
            chroma.append(mean)
        lightness = (11, 21) if dark else (96, 88)
        lab_stops = np.array(
            [[lightness[0], *(chroma[0] * 0.55)], [lightness[1], *chroma[1]]],
            np.float32,
        )
    return np.clip(cv2.cvtColor(lab_stops[None], cv2.COLOR_LAB2BGR)[0] * 255, 0, 255)


def gradient_rows(stops: np.ndarray, height: int) -> np.ndarray:
    """Smoothstep Lab interpolation; returns H×1×3 BGR float32 in [0, 255]."""
    lab = cv2.cvtColor((stops[None] / 255).astype(np.float32), cv2.COLOR_BGR2LAB)[0]
    t = np.linspace(0, 1, height, dtype=np.float32)
    t = (t * t * (3 - 2 * t))[:, None, None]
    colors = (1 - t) * lab[0] + t * lab[1]
    return np.clip(cv2.cvtColor(colors, cv2.COLOR_LAB2BGR) * 255, 0, 255)
