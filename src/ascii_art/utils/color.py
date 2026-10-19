"""Make colored ink readable against paper without changing channel order."""
import numpy as np


def readable_ink(
    colors: np.ndarray, paper: np.ndarray, ratio: float = 3.0
) -> np.ndarray:
    """BGR [0,255] ink; adjust linear-light luminance only when contrast is low."""
    rgb = np.clip(colors / 255, 0, 1)
    bg = np.clip(paper / 255, 0, 1)
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    bg_linear = np.where(bg <= 0.04045, bg / 12.92, ((bg + 0.055) / 1.055) ** 2.4)
    weights = np.array([0.0722, 0.7152, 0.2126], np.float32)
    luminance = linear @ weights
    background = bg_linear @ weights
    contrast = (np.maximum(luminance, background) + 0.05) / (
        np.minimum(luminance, background) + 0.05
    )
    darken = background > np.sqrt(0.0525) - 0.05
    target = np.where(
        darken, (background + 0.05) / ratio - 0.05, (background + 0.05) * ratio - 0.05
    )
    target = np.clip(target, 0, 1)
    darker = linear * (target / np.maximum(luminance, 1e-8))[..., None]
    blend = ((target - luminance) / np.maximum(1 - luminance, 1e-8))[..., None]
    lighter = linear + blend * (1 - linear)
    adjusted = np.where(darken[..., None], darker, lighter)
    linear = np.where((contrast < ratio)[..., None], adjusted, linear)
    linear = np.clip(linear, 0, 1)
    srgb = np.where(
        linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - 0.055
    )
    return np.clip(srgb * 255, 0, 255).astype(np.float32)
