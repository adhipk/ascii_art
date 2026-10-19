"""Utility exports; legacy SciPy helpers load only when requested."""
from typing import Any
from importlib import import_module

__all__ = ["sobel", "hex_to_bgr", "downsample_mode"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        value = getattr(import_module("ascii_art.utils.helpers"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
