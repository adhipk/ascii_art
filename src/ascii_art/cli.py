"""
Command Line Interface for ASCII Art Library
============================================

This module provides a command-line interface for generating ASCII art from images.
"""

import argparse
import cv2
from .generators.sprite_generator import SpriteASCIIGenerator, generate_ascii_image_sprite


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="Generate ASCII art from an image.")
    parser.add_argument("image_path", help="Path to the image file.")
    parser.add_argument("-o", "--output", default="result.png", help="Output file path (default: result.png)")
    parser.add_argument("--font-size", type=int, default=8, help="Font size for ASCII characters (default: 8)")
    parser.add_argument("--use-edge", action="store_true", help="Use edge detection in processing")
    parser.add_argument("--edge-threshold", type=int, default=90, help="Edge detection threshold (default: 90)")
    
    args = parser.parse_args()
    
    # Load image
    img = cv2.imread(args.image_path)
    if img is None:
        print(f"Error: Could not load image from {args.image_path}")
        return
    
    # Configure settings
    settings = {
        "use_edge": args.use_edge,
        "edge_threshold": args.edge_threshold,
        "font_size": args.font_size,
        "sharpness": 5,
        "white_point": 128
    }
    
    # Generate ASCII art
    print("Generating ASCII art...")
    rescaled_img = generate_ascii_image_sprite(img, settings)
    
    # Save result
    success = cv2.imwrite(args.output, rescaled_img)
    if success:
        print(f"ASCII art saved to {args.output}")
    else:
        print(f"Error: Could not save ASCII art to {args.output}")


if __name__ == "__main__":
    main()