"""Image-only reconstruction of Acerola's thresholded DoG / direction-vote pass.

Reference: GarrettGunnell/Post-Processing, Assets/ASCII (Unity version).
Uses top-left image coordinates and corrects the shader's additive Sobel typo
and invalid bucket access. This is not a bit-exact GPU emulation.
"""
from typing import Any, Mapping

import cv2
import numpy as np


def direction_votes(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Classify gradient normals as vertical, horizontal, backslash or slash.

    Keep the reference's narrow axial bins and broad diagonal bins. A zero
    gradient has no vote; directions describe strokes, not gradient normals.
    """
    theta = np.arctan2(gy, gx)
    angle = np.abs(theta) / np.pi
    valid = np.hypot(gx, gy) > 0
    result = np.full(gx.shape, -1, np.int8)
    result[valid & ((angle < 0.05) | (angle > 0.9))] = 0
    result[valid & (angle > 0.45) & (angle < 0.55)] = 1
    first = valid & (angle > 0.05) & (angle < 0.45)
    second = valid & (angle > 0.55) & (angle < 0.9)
    result[first] = np.where(theta[first] > 0, 3, 2)
    result[second] = np.where(theta[second] > 0, 2, 3)
    return result


def tile_directions(directions: np.ndarray, minimum: int = 8) -> np.ndarray:
    """Vote over 8x8 tiles; ties favor the first bucket, as in the shader."""
    height, width = directions.shape
    if height % 8 or width % 8:
        raise ValueError("Direction map dimensions must be multiples of eight")
    if (
        isinstance(minimum, bool)
        or not isinstance(minimum, int)
        or not 0 <= minimum <= 64
    ):
        raise ValueError("edge_min_pixels must be an integer in [0, 64]")
    tiles = directions.reshape(height // 8, 8, width // 8, 8)
    counts = np.stack([(tiles == i).sum(axis=(1, 3)) for i in range(4)], axis=-1)
    winner = counts.argmax(axis=-1).astype(np.int8)
    peak = counts.max(axis=-1)
    return np.where((peak > 0) & (peak >= minimum), winner, -1).astype(np.int8)


def edge_pass(
    gray: np.ndarray, rows: int, cols: int, settings: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    """Return inspectable DoG, binary mask, gradients, votes and tile winners.

    Analyze eight samples per cell independently of the display font. Replicated
    boundaries reproduce the Unity point-clamp sampler; finite Gaussian support
    reproduces its explicitly truncated, renormalized separable kernels.
    """
    sigma = settings.get("acerola_sigma", 2.0)
    ratio = settings.get("acerola_k", 1.6)
    tau = settings.get("acerola_tau", 1.0)
    threshold = settings.get("acerola_threshold", 0.005)
    radius = settings.get("acerola_radius", 2)
    minimum = settings.get("edge_min_pixels", 8)
    for name, value in (
        ("sigma", sigma),
        ("k", ratio),
        ("tau", tau),
        ("threshold", threshold),
    ):
        if not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"acerola_{name} must be finite")
    if sigma <= 0 or ratio <= 1 or tau < 0 or threshold < 0:
        raise ValueError("Invalid Acerola DoG settings")
    if isinstance(radius, bool) or not isinstance(radius, int) or not 1 <= radius <= 10:
        raise ValueError("acerola_radius must be an integer in [1, 10]")
    luminance = cv2.resize(gray, (cols * 8, rows * 8), interpolation=cv2.INTER_AREA)
    positions = np.arange(-radius, radius + 1, dtype=np.float32)

    def gaussian(scale: float) -> np.ndarray:
        kernel = np.exp(-0.5 * (positions / scale) ** 2)
        kernel /= kernel.sum()
        return cv2.sepFilter2D(
            luminance, cv2.CV_32F, kernel, kernel, borderType=cv2.BORDER_REPLICATE
        )

    dog = gaussian(sigma) - tau * gaussian(sigma * ratio)
    binary = (dog >= threshold).astype(np.float32)
    # The published "Sobel" passes use Scharr coefficients (3, 10, 3).
    gx = cv2.Scharr(binary, cv2.CV_32F, 1, 0, borderType=cv2.BORDER_REPLICATE)
    gy = cv2.Scharr(binary, cv2.CV_32F, 0, 1, borderType=cv2.BORDER_REPLICATE)
    votes = direction_votes(gx, gy)
    return {
        "luminance": luminance,
        "dog": dog,
        "binary": binary,
        "gx": gx,
        "gy": gy,
        "directions": votes,
        "tiles": tile_directions(votes, minimum),
    }
