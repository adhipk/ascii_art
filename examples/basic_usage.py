"""
Basic usage example for ASCII Art Library
"""

import cv2
from ascii_art import SpriteASCIIGenerator

def main():
    # Create generator with default settings
    generator = SpriteASCIIGenerator()
    
    # Load an image
    # Replace 'input.jpg' with the path to your image
    img = cv2.imread('input.jpg')
    
    if img is None:
        print("Error: Could not load image. Please provide a valid image path.")
        return
    
    # Generate ASCII art
    print("Generating ASCII art...")
    ascii_img = generator.generate_ascii(img)
    
    # Save result
    cv2.imwrite('output.png', ascii_img)
    print("ASCII art saved to output.png")

if __name__ == "__main__":
    main()