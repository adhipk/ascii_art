from ..utils.helpers import structure_tensor, edge_aligned_blur, edge_normal_blur, compute_edge_tangent_flow

import numpy as np
import argparse
import cv2
import threading
from scipy.ndimage import gaussian_filter

class DoGFilter:
    
    def __init__(self, settings=None):
        # Default settings
        default_settings = {
            "sigma_c": 0.1,
            "sigma_e": 0.5,
            "sigma_m": 1,
            "sigma_a": 2,
            "k": 1.4,
            "phi": 0.01,
            "white_point": 60,
            "sharpness": 25,
        }
        
        # Merge provided settings with defaults
        if settings is None:
            settings = {}
        self.settings = {**default_settings, **settings}
    
    def difference_of_gaussian(self,image_obj):
        g_k,g_sigma = image_obj,image_obj
        # settings
        sigma = self.settings["sigma_e"]
        sharpness = self.settings["sharpness"]
        white_point = self.settings["white_point"]
        k = self.settings["k"]
        
        threads = []
        # Dispatch threads
        threads.append(threading.Thread(target=gaussian_filter, args=(image_obj, sigma,0,g_sigma)))
        threads.append(threading.Thread(target=gaussian_filter, args=(image_obj, k*sigma,0,g_k)))

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        dog = (1+sharpness)*g_sigma - sharpness*g_k
        dog_threshold = np.where(dog > white_point, 255.0, 0.0)

        return dog_threshold
    
    def flow_dog(self,image_obj):
        # Ensure image is float32 and normalized
        image_obj = image_obj.astype(np.float32) / 255.0
        
        # get filter settings
        sigma_a = self.settings["sigma_a"];
        sigma_c = self.settings["sigma_c"];
        sigma_e = self.settings["sigma_e"];
        sigma_m = self.settings["sigma_m"];
        sharpness = self.settings["sharpness"];
        white_point = self.settings["white_point"];
        k = self.settings["k"];
        phi = self.settings["phi"];
        
        # Compute structure tensor and flows
        img_structure_tensor = structure_tensor(image_obj, sigma_c)
        edge_tangent_flow, edge_normals = compute_edge_tangent_flow(img_structure_tensor)
        
        # Apply edge-normal blurring
        g_sigma = edge_normal_blur(image_obj, edge_normals, sigma_e)
        g_k = edge_normal_blur(image_obj, edge_normals, k * sigma_e)
        
        # Compute DoG with proper normalization
        dog = (1+sharpness)*g_sigma - sharpness*g_k

        # Apply edge-aligned blur
        dog = edge_aligned_blur(dog, edge_tangent_flow, sigma_m)
        # print(dog)
        # Thresholding with proper scaling
        white_point_normalized = white_point / 255.0
        dog = np.where(
            dog > white_point_normalized,
            1.0,
            1.0 + np.tanh(phi * (dog - white_point_normalized))
        ).astype(np.uint8)
        
        # Final edge-aligned blur for anti-aliasing
        dog = edge_aligned_blur(dog, edge_tangent_flow, sigma_a)
        # Convert back to uint8
        return (255*dog)


def main():
    parser = argparse.ArgumentParser(description="Generate ASCII art from an image.")
    parser.add_argument("image_path", help="Path to the image file.")
    args = parser.parse_args()
    image_path = args.image_path
    
    img = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    dog_filter = DoGFilter()
    fdog_image = dog_filter.flow_dog(img)
    dog_image = dog_filter.difference_of_gaussian(img)
    
    cv2.imwrite("result1.png",fdog_image)
    # cv2.imwrite("result2.png",dog_image)

if __name__ == "__main__":
    main()