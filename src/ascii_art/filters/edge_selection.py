"""Stable confidence selection without image-relative percentile cutoffs."""
import cv2
import numpy as np


def sensitivity_thresholds(sensitivity: float) -> tuple[float, float]:
    """Return low/high confidence cutoffs on a logarithmic 1–100 scale.

    Zero is handled as disabled by select_edges. Log spacing makes weak edges
    adjustable over a useful range instead of crowding them near the last step.
    """
    high = 0.65 * (0.025 / 0.65) ** (sensitivity / 100)
    return high * 0.45, high


def select_edges(
    confidence: np.ndarray,
    sensitivity: float,
    *,
    eligible: np.ndarray | None = None,
    min_size: int = 3,
) -> np.ndarray:
    """Retain weak 8-connected contours seeded by strong confidence pixels.

    Removing other image content cannot change this local confidence criterion.
    Raising sensitivity only adds contours: it never drops previously kept ones.
    """
    if sensitivity <= 0:
        return np.zeros(confidence.shape, dtype=bool)
    low, high = sensitivity_thresholds(sensitivity)
    weak = confidence >= low
    if eligible is not None:
        weak &= eligible
    strong = weak & (confidence >= high)
    if not np.any(strong):
        return np.zeros(confidence.shape, dtype=bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        weak.astype(np.uint8), connectivity=8
    )
    seeded = np.zeros(count, dtype=bool)
    seeded[np.unique(labels[strong])] = True
    seeded &= stats[:, cv2.CC_STAT_AREA] >= min_size
    seeded[0] = False
    return seeded[labels]
