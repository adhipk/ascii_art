"""
ASCII Art Library
=================

A Python library for generating ASCII art from images with advanced filtering capabilities.

Main Components:
- SpriteASCIIGenerator: Main class for generating ASCII art using sprite-based approach
- DoGFilter: Difference of Gaussians filter for edge detection
- Various utility functions for image processing

Basic Usage:
    from ascii_art import SpriteASCIIGenerator
    import cv2
    
    # Create generator with default settings
    generator = SpriteASCIIGenerator()
    
    # Load and process an image
    img = cv2.imread("input.jpg")
    ascii_img = generator.generate_ascii(img)
    
    # Save result
    cv2.imwrite("output.png", ascii_img)
"""

__version__ = "0.1.0"
__author__ = "Adhip Kashyap"

# Public API exports
from .generators.sprite_generator import (
    SpriteASCIIGenerator,
    generate_ascii_image_sprite,
)
from .filters.dog_filter import DoGFilter
from typing import Any


def __getattr__(name: str) -> Any:
    """Load legacy SciPy utilities only when requested by a caller."""
    if name in {"sobel", "hex_to_bgr", "downsample_mode"}:
        from ascii_art.utils import helpers

        value = getattr(helpers, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SpriteASCIIGenerator",
    "DoGFilter",
    "generate_ascii_image_sprite",
    "sobel",
    "hex_to_bgr",
    "downsample_mode",
]
