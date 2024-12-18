
import numpy as np
from scipy.signal import convolve2d

from scipy.ndimage import gaussian_filter
import cv2
    
    
def sobel(image_array):
    """Applies Sobel operator for edge detection."""
    
    sobel_x, sobel_y = sobel_components(image_array)
    return  (np.hypot(sobel_x, sobel_y), np.arctan2(sobel_y, sobel_x))

def sobel_components(image_array):
    dx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    dy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    sobel_x = convolve2d(image_array, dx, mode='same', boundary='symm')
    sobel_y = convolve2d(image_array, dy, mode='same', boundary='symm')
    return sobel_x,sobel_y

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

def display_image(image,title='debug'):
    cv2.imshow(title, image)
    cv2.waitKey(0)  # Wait for any key press
    cv2.destroyAllWindows()  # Close the window

def structure_tensor(image, sigma_c=1):
    Ix, Iy = sobel_components(image)
    
    # Compute structure tensor components
    Ixx = gaussian_filter(Ix * Ix, sigma_c)
    Iyy = gaussian_filter(Iy * Iy, sigma_c)
    Ixy = gaussian_filter(Ix * Iy, sigma_c)
    
    return Ixx, Ixy, Iyy

def structure_tensor_eigenvalues(structure_tensor):
    E, F, G = structure_tensor
    H = np.sqrt((E - G)**2 + 4*F**2)
    lambda1 = 0.5*((E+G) + H)
    lambda2 = 0.5*((E+G) - H)
    
    return lambda1, lambda2

def compute_edge_tangent_flow(structure_tensor):
    xx, xy, yy = structure_tensor
    lambda1, lambda2 = structure_tensor_eigenvalues([xx, xy, yy])
    
    def eigen_vectors(eigen_value):
        tangent_x = eigen_value - xx
        tangent_y = -1*xy
        
        # Normalize with epsilon to prevent division by zero
        magnitude = np.sqrt(tangent_x**2 + tangent_y**2)
        epsilon = 1e-10
        magnitude = np.maximum(magnitude, epsilon)
        
        tangent_x = tangent_x/magnitude
        tangent_y = tangent_y/magnitude
        
        return np.dstack([tangent_x, tangent_y])
    
    return eigen_vectors(lambda1), eigen_vectors(lambda2)

def edge_normal_blur(image, edge_normals, sigma_e):
    blurred_image = np.zeros_like(image, dtype=np.float32)
    weight_sum = np.zeros_like(image, dtype=np.float32)
    height, width = image.shape
    
    # get gaussian kernal
    kernel_size = int(4 * sigma_e + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    x_kernel = np.linspace(-(kernel_size-1)/2, (kernel_size-1)/2, kernel_size)
    gaussian_kernel = np.exp(-(x_kernel**2) / (2 * sigma_e**2))
    gaussian_kernel /= np.sum(gaussian_kernel)
    
    half_width = kernel_size // 2
    
    # Vectorized coordinates
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    
    for j in range(-half_width, half_width + 1):
        # Compute sampling points along normal direction
        px = x_coords + j * edge_normals[..., 0]
        py = y_coords + j * edge_normals[..., 1]
        
        # Floor coordinates for interpolation
        px_low = np.floor(px).astype(int)
        py_low = np.floor(py).astype(int)
        
        # Ensure we're within bounds
        valid_mask = (px_low >= 0) & (px_low < width-1) & (py_low >= 0) & (py_low < height-1)
        
        # Compute interpolation weights
        wx_high = px - px_low
        wy_high = py - py_low
        wx_low = 1 - wx_high
        wy_low = 1 - wy_high
        
        # Interpolate
        where_valid = np.where(valid_mask)
        if where_valid[0].size > 0:
            sample = (wx_low[valid_mask] * wy_low[valid_mask] * image[py_low[valid_mask], px_low[valid_mask]] +
                     wx_high[valid_mask] * wy_low[valid_mask] * image[py_low[valid_mask], np.minimum(px_low[valid_mask] + 1, width-1)] +
                     wx_low[valid_mask] * wy_high[valid_mask] * image[np.minimum(py_low[valid_mask] + 1, height-1), px_low[valid_mask]] +
                     wx_high[valid_mask] * wy_high[valid_mask] * image[np.minimum(py_low[valid_mask] + 1, height-1), np.minimum(px_low[valid_mask] + 1, width-1)])
            
            weight = gaussian_kernel[j + half_width]
            blurred_image[where_valid] += weight * sample
            weight_sum[where_valid] += weight
    
    # Normalize by weight sum
    weight_sum[weight_sum == 0] = 1
    blurred_image /= weight_sum
    
    return blurred_image

def edge_aligned_blur(image, flow, sigma):
    """
    Blur an image by following flow field streamlines.
    
    Parameters:
    -----------
    image : ndarray
        Input image to blur
    flow : ndarray
        Flow field of shape (H, W, 2) containing (dx, dy) vectors
    sigma : float
        Controls the extent of the blur
    """
    blurred_image = np.zeros_like(image, dtype=np.float32)
    weight_sum = np.zeros_like(image, dtype=np.float32)
    height, width = image.shape
    
    # Optimize kernel size
    kernel_size = int(4 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    x_kernel = np.linspace(-(kernel_size-1)/2, (kernel_size-1)/2, kernel_size)
    gaussian_kernel = np.exp(-(x_kernel**2) / (2 * sigma**2))
    gaussian_kernel /= np.sum(gaussian_kernel)
    
    half_width = kernel_size // 2
    
    # Initialize starting positions for all pixels
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    
    # For both forward and backward directions
    for direction in [1, -1]:
        # Start from initial positions
        cur_x = x_coords.astype(np.float32)
        cur_y = y_coords.astype(np.float32)
        
        for j in range(half_width):
            # Floor coordinates for interpolation
            px_low = np.floor(cur_x).astype(np.int32)
            py_low = np.floor(cur_y).astype(np.int32)
            
            # Ensure we're within bounds
            valid_mask = (px_low >= 0) & (px_low < width-1) & (py_low >= 0) & (py_low < height-1)
            
            # Compute interpolation weights
            wx_high = cur_x - px_low
            wy_high = cur_y - py_low
            wx_low = 1 - wx_high
            wy_low = 1 - wy_high
            
            # Interpolate image values
            where_valid = np.where(valid_mask)
            if where_valid[0].size > 0:
                sample = (wx_low[valid_mask] * wy_low[valid_mask] * 
                         image[py_low[valid_mask], px_low[valid_mask]] +
                         wx_high[valid_mask] * wy_low[valid_mask] * 
                         image[py_low[valid_mask], np.minimum(px_low[valid_mask] + 1, width-1)] +
                         wx_low[valid_mask] * wy_high[valid_mask] * 
                         image[np.minimum(py_low[valid_mask] + 1, height-1), px_low[valid_mask]] +
                         wx_high[valid_mask] * wy_high[valid_mask] * 
                         image[np.minimum(py_low[valid_mask] + 1, height-1), 
                               np.minimum(px_low[valid_mask] + 1, width-1)])
                
                weight = gaussian_kernel[j + half_width]
                blurred_image[where_valid] += weight * sample
                weight_sum[where_valid] += weight
            
            # Interpolate flow field values
            flow_x = (wx_low * wy_low * flow[py_low, px_low, 0] +
                     wx_high * wy_low * flow[py_low, np.minimum(px_low + 1, width-1), 0] +
                     wx_low * wy_high * flow[np.minimum(py_low + 1, height-1), px_low, 0] +
                     wx_high * wy_high * flow[np.minimum(py_low + 1, height-1), 
                                            np.minimum(px_low + 1, width-1), 0])
            
            flow_y = (wx_low * wy_low * flow[py_low, px_low, 1] +
                     wx_high * wy_low * flow[py_low, np.minimum(px_low + 1, width-1), 1] +
                     wx_low * wy_high * flow[np.minimum(py_low + 1, height-1), px_low, 1] +
                     wx_high * wy_high * flow[np.minimum(py_low + 1, height-1), 
                                            np.minimum(px_low + 1, width-1), 1])
            
            # Normalize flow vectors
            flow_magnitude = np.sqrt(flow_x**2 + flow_y**2)
            flow_magnitude[flow_magnitude < 1e-10] = 1
            flow_x /= flow_magnitude
            flow_y /= flow_magnitude
            
            # Update positions by following flow
            cur_x = cur_x + direction * flow_x
            cur_y = cur_y + direction * flow_y
            
            # Clip to image boundaries
            cur_x = np.clip(cur_x, 0, width-1)
            cur_y = np.clip(cur_y, 0, height-1)
    
    # Normalize by weight sum
    weight_sum[weight_sum == 0] = 1
    blurred_image /= weight_sum
    
    return blurred_image

