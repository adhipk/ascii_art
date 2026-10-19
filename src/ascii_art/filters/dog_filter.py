"""Non-mutating extended DoG and flow-guided line extraction."""
from typing import Any, Mapping

import cv2
import numpy as np


def grayscale(image: np.ndarray) -> np.ndarray:
    """Validate uint8 or normalized float images; return grayscale float32 [0, 1]."""
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("image must be a nonempty NumPy array")
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] not in (3, 4)):
        raise ValueError("image must have shape HxW, HxWx3 (BGR), or HxWx4 (BGRA)")
    if image.dtype == np.uint8:
        result = image.astype(np.float32) / 255.0
    elif np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("float images must contain finite values in [0, 1]")
        result = image.astype(np.float32, copy=True)
    else:
        raise ValueError("image must be uint8 or normalized floating point")
    if result.ndim == 3:
        result = cv2.cvtColor(
            result, cv2.COLOR_BGRA2GRAY if result.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        )
    return result


def blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian with reflected boundary conditions."""
    return cv2.GaussianBlur(image, (0, 0), sigma, borderType=cv2.BORDER_REFLECT_101)


def gradient(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalized Scharr derivatives, preserving signed floating-point responses."""
    return (
        cv2.Scharr(image, cv2.CV_32F, 1, 0, scale=1 / 32),
        cv2.Scharr(image, cv2.CV_32F, 0, 1, scale=1 / 32),
    )


def directional_blur(
    image: np.ndarray, field: np.ndarray, sigma: float, *, follow: bool = False
) -> np.ndarray:
    """Bilinear Gaussian sampling along normals or sign-consistent streamlines."""
    yy, xx = np.indices(image.shape, dtype=np.float32)
    result = image.copy()
    weight_sum = 1.0
    for sign in (-1, 1):
        x, y = xx.copy(), yy.copy()
        previous = field * sign
        for step in range(1, int(np.ceil(3 * sigma)) + 1):
            if follow:
                direction = cv2.remap(
                    field, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
                )
                flip = np.sum(direction * previous, axis=2) < 0
                direction[flip] *= -1
                direction /= np.maximum(
                    np.linalg.norm(direction, axis=2, keepdims=True), 1e-8
                )
                previous = direction
                x = np.clip(x + direction[..., 0], 0, image.shape[1] - 1)
                y = np.clip(y + direction[..., 1], 0, image.shape[0] - 1)
            else:
                x = xx + sign * step * field[..., 0]
                y = yy + sign * step * field[..., 1]
            weight = float(np.exp(-0.5 * (step / sigma) ** 2))
            result += weight * cv2.remap(
                image, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
            )
            weight_sum += weight
    return result / weight_sum


class DoGFilter:
    """Extended DoG; thresholds use legacy 0–255 units, calculations use [0, 1]."""

    def __init__(self, settings: Mapping[str, Any] | None = None) -> None:
        self.settings = {
            "sigma_c": 1.0,
            "sigma_e": 1.0,
            "sigma_m": 2.0,
            "sigma_a": 0.7,
            "k": 1.6,
            "phi": 10.0,
            "white_point": 128.0,
            "sharpness": 5.0,
            **(settings or {}),
        }
        for name in (
            "sigma_c",
            "sigma_e",
            "sigma_m",
            "sigma_a",
            "k",
            "phi",
            "white_point",
            "sharpness",
        ):
            value = self.settings[name]
            if not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if any(self.settings[k] <= 0 for k in ("sigma_c", "sigma_e", "sigma_m", "phi")):
            raise ValueError("sigmas and phi must be positive")
        if (
            self.settings["sigma_a"] < 0
            or self.settings["k"] <= 1
            or self.settings["sharpness"] < 0
        ):
            raise ValueError(
                "sigma_a and sharpness must be nonnegative and k must exceed 1"
            )
        if not 0 <= self.settings["white_point"] <= 255:
            raise ValueError("white_point must be in [0, 255]")

    def response(self, image: np.ndarray) -> np.ndarray:
        """Return the unthresholded extended DoG response without changing input."""
        gray = grayscale(image)
        sigma, p = self.settings["sigma_e"], self.settings["sharpness"]
        return (1 + p) * blur(gray, sigma) - p * blur(gray, sigma * self.settings["k"])

    def _threshold(self, response: np.ndarray) -> np.ndarray:
        delta = response - self.settings["white_point"] / 255.0
        return np.where(delta >= 0, 1.0, 1.0 + np.tanh(self.settings["phi"] * delta))

    def difference_of_gaussian(self, image_obj: np.ndarray) -> np.ndarray:
        """Return an antialiased uint8 XDoG line image (dark lines on white)."""
        return np.rint(255 * self._threshold(self.response(image_obj))).astype(np.uint8)

    def flow_dog(self, image_obj: np.ndarray) -> np.ndarray:
        """Filter across edges, then smooth along structure-tensor tangent flow."""
        gray = grayscale(image_obj)
        gx, gy = gradient(blur(gray, self.settings["sigma_c"]))
        xx = blur(gx * gx, self.settings["sigma_c"])
        xy = blur(gx * gy, self.settings["sigma_c"])
        yy = blur(gy * gy, self.settings["sigma_c"])
        theta = 0.5 * np.arctan2(2 * xy, xx - yy)
        normals = np.stack((np.cos(theta), np.sin(theta)), axis=-1)
        tangents = np.stack((-np.sin(theta), np.cos(theta)), axis=-1)
        sigma, p = self.settings["sigma_e"], self.settings["sharpness"]
        narrow = directional_blur(gray, normals, sigma)
        wide = directional_blur(gray, normals, sigma * self.settings["k"])
        response = directional_blur(
            (1 + p) * narrow - p * wide, tangents, self.settings["sigma_m"], follow=True
        )
        lines = self._threshold(response).astype(np.float32)
        if self.settings["sigma_a"] > 0:
            lines = directional_blur(
                lines, tangents, self.settings["sigma_a"], follow=True
            )
        return np.rint(np.clip(lines, 0, 1) * 255).astype(np.uint8)
