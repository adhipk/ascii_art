from utils import difference_of_gaussian
from PIL import Image
import numpy as np
import argparse
import cv2

def main():
    parser = argparse.ArgumentParser(description="Generate ASCII art from an image.")
    parser.add_argument("image_path", help="Path to the image file.")
    args = parser.parse_args()
    image_path = args.image_path
    img = np.array(Image.open(image_path).convert('L'),dtype=np.float32)
    
    dog_image = difference_of_gaussian(img)
    result = cv2.imwrite("result.png",dog_image)

if __name__ == "__main__":
    main()