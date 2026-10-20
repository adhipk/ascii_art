"""Quality metrics must distinguish missing, invented and displaced contours."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "adaptive_quality", Path(__file__).parents[1] / "benchmarks/adaptive_quality.py"
)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_perfect_and_empty_contours():
    line = np.zeros((32, 32), bool)
    line[:, 16] = True
    for reference in (line, np.zeros_like(line)):
        result = benchmark.contour_metrics(reference, reference)
        assert result["precision"] == result["recall"] == result["f1"] == 1
        assert result["position_error_px"] == result["false_contour_rate"] == 0


def test_invented_and_missing_strokes_are_not_perfect():
    blank = np.zeros((32, 32), bool)
    line = blank.copy()
    line[:, 16] = True
    invented = benchmark.contour_metrics(blank, line)
    assert invented["false_contour_rate"] == 1
    assert invented["f1"] == 0
    missing = benchmark.contour_metrics(line, blank)
    assert missing["recall"] == missing["f1"] == 0
    assert missing["position_error_px"] is None


def test_fixed_tolerance_and_distance_are_measured_in_pixels():
    reference = np.zeros((32, 32), bool)
    reference[:, 16] = True
    near = benchmark.contour_metrics(reference, np.roll(reference, 2, axis=1))
    far = benchmark.contour_metrics(reference, np.roll(reference, 3, axis=1))
    assert near["precision"] == near["recall"] == 1
    assert near["position_error_px"] == pytest.approx(2)
    assert far["precision"] == far["recall"] == 0
    assert far["position_error_px"] == pytest.approx(3)
