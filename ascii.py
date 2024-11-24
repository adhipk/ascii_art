
import argparse
from utils import generate_ascii_image_sprite
import cv2

def main():
    parser = argparse.ArgumentParser(description="Generate ASCII art from an image.")
    parser.add_argument("image_path", help="Path to the image file.")
    args = parser.parse_args()
    image_path = args.image_path
    img = cv2.imread(image_path)
    rescaled_img = generate_ascii_image_sprite(img,8)
    result = cv2.imwrite("result.png",rescaled_img)
    print(result)
if __name__ == "__main__":
    main()
