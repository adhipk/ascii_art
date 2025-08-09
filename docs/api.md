# API Reference

## Main Modules

### ascii_art

Main package module.

#### Classes

##### SpriteASCIIGenerator

Main class for generating ASCII art using sprite-based approach.

```python
from ascii_art import SpriteASCIIGenerator
```

**Methods:**

- `__init__(self, settings: Dict[str, Any])`: Initialize the generator with settings
  - `settings`: Configuration dictionary with the following keys:
    - `use_edge` (bool): Enable edge detection (default: False)
    - `edge_threshold` (int): Edge detection threshold (default: 90)
    - `font_size` (int): Font size for characters (default: 8)
    - `sharpness` (int): Image sharpness (default: 100)
    - `white_point` (int): White point threshold (default: 2)

- `generate_ascii(self, img: np.ndarray) -> np.ndarray`: Generate ASCII art from an image
  - `img`: Input image as numpy array
  - Returns: ASCII art image as numpy array

##### DoGFilter

Difference of Gaussians filter for edge detection.

```python
from ascii_art import DoGFilter
```

**Methods:**

- `__init__(self, settings: Dict[str, Any])`: Initialize the filter with settings
  - `settings`: Configuration dictionary with the following keys:
    - `sigma_c` (float): Structure tensor sigma (default: 0.1)
    - `sigma_e` (float): Edge sigma (default: 0.5)
    - `sigma_m` (float): Median sigma (default: 1)
    - `sigma_a` (float): Anti-aliasing sigma (default: 2)
    - `k` (float): DoG multiplier (default: 1.4)
    - `phi` (float): Flow DoG parameter (default: 0.01)
    - `white_point` (int): White point threshold (default: 60)
    - `sharpness` (int): Sharpness factor (default: 25)

- `difference_of_gaussian(self, image_obj)`: Apply DoG filter to an image
- `flow_dog(self, image_obj)`: Apply flow-based DoG filtering

#### Functions

##### generate_ascii_image_sprite

Convenience function to generate colored ASCII art.

```python
from ascii_art import generate_ascii_image_sprite
```

**Parameters:**
- `img`: Input image as numpy array
- `settings`: Configuration dictionary

**Returns:**
- ASCII art image as numpy array

### ascii_art.generators

Module containing different approaches for generating ASCII art from images.

### ascii_art.filters

Module containing various image filters that can be used to preprocess images before converting them to ASCII art.

### ascii_art.utils

Module containing various utility functions used throughout the ASCII art library.

#### Functions

##### sobel

Applies Sobel operator for edge detection.

```python
from ascii_art.utils import sobel
```

**Parameters:**
- `image_array`: Input image as numpy array

**Returns:**
- Tuple of (magnitude, direction) arrays

##### hex_to_bgr

Convert hex color to BGR tuple.

```python
from ascii_art.utils import hex_to_bgr
```

**Parameters:**
- `value`: Hex color string (e.g., "#FFFFFF")

**Returns:**
- BGR tuple (blue, green, red)

##### downsample_mode

Downsample image using mode filtering.

```python
from ascii_art.utils import downsample_mode
```

**Parameters:**
- `img`: Input image as numpy array
- `kernel_size`: Size of downsampling kernel

**Returns:**
- Downsampled image as numpy array