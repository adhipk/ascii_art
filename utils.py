
import cv2

import numpy as np
from scipy.signal import convolve2d
import threading
from scipy.ndimage import gaussian_filter,sobel
from PIL import Image,ImageDraw, ImageFont
from scipy import stats
from numba import njit


def difference_of_gaussian(image_obj, sigma=2,k=1.6,white_point=100,sharpness=4):

    g_sigma = gaussian_filter(image_obj, sigma)
    g_k = gaussian_filter(image_obj, k*sigma)
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
    """
    Downsample the image by taking the most common value in each kernel,
    ignoring zeros. Optimized version using NumPy operations.
    
    Args:
        img: Input image as 2D numpy array
        kernel_size: Size of the downsampling kernel (e.g., 2 for 2x2 blocks)
        
    Returns:
        Downsampled image as 2D numpy array
    """
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



class SpriteASCIIGenerator:
    def __init__(self, grid_size=10,settings={
            "use_edge": False,
            "edge_threshold": 90,
            "use_color": False,
            "font_size": 8,
            "sharpness": 100,
            "white_point": 2
    }):
        self.grid_size = grid_size
        ascii_chars = "@%#*+=:-. "
        # ascii_chars = " "
        edge_chars = "-/|\\"
        self.num_ascii_chars = len(ascii_chars)
        self.num_edge_chars = len(edge_chars)
        self.colors = [(41, 9,53 ),(  209,232,255)]
        self.settings = settings
        
        # Cache font
        self.font = ImageFont.truetype(font="fonts/dogica.ttf", size=self.grid_size)
        
        # Generate and cache sprite sheet
        self.char_sprite_sheet, self.char_width = self._generate_sprite_sheet(ascii_chars+edge_chars)
        
        # Pre-compute character masks
        self.char_masks = self._precompute_char_masks(self.char_sprite_sheet,len(ascii_chars+edge_chars))

        
    def _generate_sprite_sheet(self, char_list):
        """Generate a sprite sheet containing all ASCII characters"""
        # Measure character width
        num_chars = len(char_list)
        temp_img = Image.new('RGB', (self.grid_size * 2, self.grid_size), 'white')
        temp_draw = ImageDraw.Draw(temp_img)
        char_width = temp_draw.textlength("@", font=self.font)
        
        # Create sprite sheet
        sheet_width = int(char_width * num_chars)
        sprite_img = Image.new('RGB', (sheet_width, self.grid_size), 'white')
        sprite_draw = ImageDraw.Draw(sprite_img)
        
        # Draw characters
        for i, char in enumerate(char_list):
            sprite_draw.text((i * char_width, 0), char, fill='black', font=self.font)
        
        # Convert to binary mask
        sprite_sheet = cv2.cvtColor(np.array(sprite_img), cv2.COLOR_RGB2GRAY)
        sprite_sheet = (sprite_sheet < 128).astype(np.uint8) * 255
        
        return sprite_sheet, int(char_width)
    
    def _precompute_char_masks(self,sprite_sheet,num_chars):
        """Pre-compute binary masks for all characters"""
        masks = []
        for i in range(num_chars):
            sprite_x = int(i * self.char_width)
            char_sprite = sprite_sheet[:, sprite_x:sprite_x + self.char_width]
            masks.append(char_sprite > 0)
        return masks
    
    def _paste_characters(self, output_img, char_indices, color_data=None):
        """
        Paste characters onto the output image with optional color data.
        
        Args:
            output_img: Target image array
            char_indices: 2D array of character indices
            color_data: Optional 3D array of RGB colors for each character position
        """
        height, width = char_indices.shape
        
        # Pre-calculate valid dimensions
        max_y = min(height * self.grid_size, output_img.shape[0])
        max_x = min(width * self.char_width, output_img.shape[1])
        
        # Calculate valid grid positions
        y_positions = range(0, max_y, self.grid_size)
        x_positions = range(0, max_x, self.char_width)
        
        # Iterate only over valid positions
        for i, y in enumerate(y_positions):
            for j, x in enumerate(x_positions):
                char_idx = char_indices[i, j]
                mask = self.char_masks[char_idx]
                region = output_img[y:y + self.grid_size, x:x + self.char_width]
                
                if color_data is not None:
                    color = color_data[i, j]
                    for c in range(3):
                        region[..., c][mask] = color[c]
                else:
                    # Greyscale version: simply apply mask
                    for c in range(3):
                        region[...,c][mask] = self.colors[0][c]
                        region[...,c][~mask] = self.colors[1][c]
                


    def process_brightness(self,img,new_width,new_height, results, index):
        

        resized_gray = cv2.cvtColor(cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        char_indices = np.multiply(resized_gray, self.num_ascii_chars / 256, 
                                    dtype=np.float32).astype(np.int32)
        char_indices = np.clip(char_indices, 0, self.num_ascii_chars - 1)
        results[index] = char_indices

    def process_edges(self,img, results, index):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), cv2.BORDER_DEFAULT)
        sobel_mag, sobel_dir = sobel(difference_of_gaussian(blurred, white_point=self.settings["white_point"], sharpness=self.settings["sharpness"]))
        
        edge_threshold = np.percentile(sobel_mag, self.settings["edge_threshold"])
        
        
        sobel_mag_down = downsample_mode(sobel_mag, self.grid_size)
        edge_mask = sobel_mag_down > edge_threshold
        sobel_dir_down = downsample_mode((sobel_dir // (np.pi / 4)) % self.num_edge_chars, self.grid_size).astype(np.uint8)
        
        edge_indices = self.num_ascii_chars + sobel_dir_down[edge_mask]
        results[index] = (edge_mask, edge_indices)

    def generate_ascii(self,img):
        # Placeholder for results
        results = [None, None]
        threads = []
        height, width = img.shape[:2]
        new_width = width // self.grid_size
        new_height = height // self.grid_size
        # Dispatch threads
        threads.append(threading.Thread(target=self.process_brightness, args=(img,new_width,new_height, results, 0)))
        threads.append(threading.Thread(target=self.process_edges, args=(img, results, 1)))

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Combine results
        brightness_indices = results[0]
        edge_mask, edge_indices = results[1]
        
        combined_indices = brightness_indices.copy()
        combined_indices[edge_mask] = edge_indices

        # Create output image (white background)
        output_height = new_height * self.grid_size
        output_width = new_width * self.char_width
        output_img = np.full((output_height, output_width, 3), 
                           255, dtype=np.uint8)
        
        # Paste chars
        
        self._paste_characters(output_img,combined_indices)
        return output_img


def generate_ascii_image_sprite(img, grid_size=16,settings={
            "use_edge": False,
            "edge_threshold": 90,
            "use_color": False,
            "font_size": 8,
            "sharpness": 100,
            "white_point": 2
}):
    """Convenience function to generate colored ASCII art"""
    generator = SpriteASCIIGenerator(grid_size=grid_size,settings=settings)
    return generator.generate_ascii(img)
