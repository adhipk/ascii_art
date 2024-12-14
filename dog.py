from utils import difference_of_gaussian,flow_dog,display_image

import numpy as np
import argparse
import cv2




def main():
    parser = argparse.ArgumentParser(description="Generate ASCII art from an image.")
    parser.add_argument("image_path", help="Path to the image file.")
    args = parser.parse_args()
    image_path = args.image_path
    sigma_c=0.1
    sigma_e=0.5
    sigma_m=1
    sigma_a=2
    k=1.4
    white_point=60
    sharpness=25;
    img = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    fdog_image = flow_dog(img,sigma_c, sigma_e, sigma_m,sigma_a, k, white_point, sharpness)
    dog_image = difference_of_gaussian(img,sigma_c, k, white_point, sharpness)
    result = cv2.imwrite("result1.png",fdog_image)
    result = cv2.imwrite("result2.png",dog_image)

if __name__ == "__main__":
    main()