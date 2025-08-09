# Examples

## Basic Usage

### Simple ASCII Art Generation

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

### With Edge Detection

```python
from ascii_art import SpriteASCIIGenerator
import cv2

# Create generator with edge detection enabled
settings = {
    "use_edge": True,
    "edge_threshold": 90,
    "font_size": 8,
    "sharpness": 5,
    "white_point": 128
}

generator = SpriteASCIIGenerator(settings)

# Load and process an image
img = cv2.imread("input.jpg")
ascii_img = generator.generate_ascii(img)

# Save result
cv2.imwrite("output_with_edges.png", ascii_img)
```

## Advanced Usage

### Custom Filter Settings

```python
from ascii_art import SpriteASCIIGenerator, DoGFilter
import cv2

# Create generator with custom settings
settings = {
    "use_edge": True,
    "edge_threshold": 85,
    "font_size": 10,
    "sharpness": 10,
    "white_point": 64
}

generator = SpriteASCIIGenerator(settings)

# Load and process an image
img = cv2.imread("input.jpg")
ascii_img = generator.generate_ascii(img)

# Save result
cv2.imwrite("high_res_output.png", ascii_img)
```

### Using DoGFilter Directly

```python
from ascii_art import DoGFilter
import cv2

# Create filter with custom settings
filter_settings = {
    "sigma_e": 0.8,
    "k": 1.6,
    "sharpness": 30,
    "white_point": 70
}

dog_filter = DoGFilter(filter_settings)

# Load grayscale image
img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)

# Apply DoG filter
filtered_img = dog_filter.difference_of_gaussian(img)

# Save result
cv2.imwrite("dog_filtered.png", filtered_img)
```

## Command Line Examples

### Basic Usage

```bash
ascii-art input.jpg
```

### With Custom Output

```bash
ascii-art input.jpg -o my_ascii_art.png
```

### With Edge Detection

```bash
ascii-art input.jpg --use-edge --edge-threshold 90 -o output.png
```

### With Custom Font Size

```bash
ascii-art input.jpg --font-size 12 -o large_font_output.png
```