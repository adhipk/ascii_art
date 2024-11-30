



import numpy as np
from scipy.signal import convolve2d

from scipy.ndimage import gaussian_filter
import threading



def difference_of_gaussian(image_obj, sigma=1,k=1.6,white_point=100,sharpness=20):
    g_k,g_sigma = image_obj,image_obj
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
    
    
    
def sobel(image_array,treshold=80):
    """Applies Sobel operator for edge detection."""
    
    dx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    dy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    sobel_x = convolve2d(image_array, dx, mode='same', boundary='symm')
    sobel_y = convolve2d(image_array, dy, mode='same', boundary='symm')
    return  (np.hypot(sobel_x, sobel_y), np.arctan2(sobel_y, sobel_x))


def downsample_mode(img, kernel_size):
    
    h, w = img.shape
    new_h, new_w = h // kernel_size, w // kernel_size
    
    # Trim image to be divisible by kernel_size
    img = img[:new_h * kernel_size, :new_w * kernel_size]
    
    # Reshape to group pixels into blocks
    blocks = img.reshape(new_h, kernel_size, new_w, kernel_size)
    result = np.zeros((new_h, new_w), dtype=img.dtype)
    for i in range(new_h):
        for j in range(new_w):
            block = blocks[i, :, j, :].ravel()
            # Get non-zero values
            valid_vals = block[block != 0]
            if len(valid_vals) == 0:
                result[i, j] = 0
                continue
                
            # Find the most common value
            unique_vals, counts = np.unique(valid_vals, return_counts=True)
            result[i, j] = unique_vals[np.argmax(counts)]
    
    return result
def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def hex_to_bgr(value):
    r,g,b = hex_to_rgb(value)
    return b,g,r


