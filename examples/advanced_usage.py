"""
Advanced usage example for ASCII Art Library
"""

import cv2
from ascii_art import SpriteASCIIGenerator

def main():
    # Create generator with custom settings
    settings = {
        "use_edge": True,        # Enable edge detection
        "edge_threshold": 85,    # Edge detection threshold
        "font_size": 10,         # Larger font size
        "sharpness": 10,         # Increased sharpness
        "white_point": 64        # Adjusted white point
    }
    
    generator = SpriteASCIIGenerator(settings)
    
    # Load an image
    # Replace 'input.jpg' with the path to your image
    img = cv2.imread('input.jpg')
    
    if img is None:
        print("Error: Could not load image. Please provide a valid image path.")
        return
    
    # Generate ASCII art
    print("Generating advanced ASCII art with edge detection...")
    ascii_img = generator.generate_ascii(img)
    
    # Save result
    cv2.imwrite('advanced_output.png', ascii_img)
    print("Advanced ASCII art saved to advanced_output.png")

if __name__ == "__main__":
    main()