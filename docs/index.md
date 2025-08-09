# ASCII Art Library Documentation

Welcome to the ASCII Art Library documentation. This library provides tools for generating ASCII art from images with advanced filtering capabilities.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](api.md)
- [Examples](examples.md)
- [CLI Usage](cli.md)

## Installation

To install the library, run:

```bash
pip install -e .
```

## Quick Start

### Command Line Usage

Generate ASCII art from an image:

```bash
ascii-art input.jpg -o output.png
```

### Programmatic Usage

```python
from ascii_art import SpriteASCIIGenerator
import cv2

# Create generator with default settings
generator = SpriteASCIIGenerator()

# Load and process an image
img = cv2.imread("input.jpg")
ascii_img = generator.generate_ascii(img)

# Save result
cv2.imwrite("output.png", ascii_img)
```

## Features

- Sprite-based character rendering for high-quality output
- Edge detection using Difference of Gaussians (DoG) filtering
- Configurable character sets and rendering parameters
- Both programmatic API and command-line interface
- Caching mechanisms for improved performance