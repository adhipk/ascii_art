# Command Line Interface

The ASCII Art library provides a command-line interface for generating ASCII art from images.

## Usage

```bash
ascii-art [OPTIONS] IMAGE_PATH
```

## Arguments

- `IMAGE_PATH`: Path to the input image file

## Options

- `-o, --output PATH`: Output file path (default: result.png)
- `--font-size INTEGER`: Font size for ASCII characters (default: 8)
- `--use-edge`: Use edge detection in processing
- `--edge-threshold INTEGER`: Edge detection threshold (default: 90)
- `-h, --help`: Show help message and exit

## Examples

### Basic Usage

```bash
ascii-art input.jpg
```

This will generate ASCII art from `input.jpg` and save it as `result.png`.

### Custom Output Path

```bash
ascii-art input.jpg -o my_art.png
```

This will save the ASCII art to `my_art.png`.

### With Edge Detection

```bash
ascii-art input.jpg --use-edge --edge-threshold 90 -o output.png
```

This will generate ASCII art with edge detection enabled.

### With Larger Font Size

```bash
ascii-art input.jpg --font-size 12 -o large_output.png
```

This will generate ASCII art with a larger font size.