
import cv2

import numpy as np
from scipy.signal import convolve2d

from scipy.ndimage import gaussian_filter
from PIL import Image,ImageDraw, ImageFont

import cProfile
import pstats
from pstats import SortKey
from downsampler import FeaturePreservingDownsampler

def display(img):
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def difference_of_gaussian(image_obj, sigma=2,k=1.6,white_point=100,sharpness=2):

    g_sigma = gaussian_filter(image_obj, sigma)
    g_k = gaussian_filter(image_obj, k*sigma)
    dog = (1+sharpness)*g_sigma - sharpness*g_k
    dog_threshold = np.where(dog > white_point, 255.0, 0.0)

    return dog_threshold
    
    
    
def sobel(image_array):
    """Applies Sobel operator for edge detection."""
    
    dx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    dy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    sobel_x = convolve2d(image_array, dx, mode='same', boundary='symm')
    sobel_y = convolve2d(image_array, dy, mode='same', boundary='symm')
    sobel_mag = np.hypot(sobel_x, sobel_y)
    
    # direction is normal to edge, so shift by 90 degrees and round to nearest multiple of 45
    sobel_dir = ((np.arctan2(sobel_y, sobel_x) + np.pi/2) * 180/np.pi) % 180
    sobel_dir = np.round(sobel_dir / 45) * 45
    return sobel_mag, sobel_dir

def non_maximum_suppression(mag, direction):
    """Performs non-maximum suppression on gradient magnitude."""
    rows, cols = mag.shape
    output = np.zeros_like(mag)
    
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            # Get neighbors based on gradient direction
            if direction[i,j] == 0 or direction[i,j] == 180:
                neighbors = [mag[i,j-1], mag[i,j+1]]
            elif direction[i,j] == 45:
                neighbors = [mag[i-1,j+1], mag[i+1,j-1]]
            elif direction[i,j] == 90:
                neighbors = [mag[i-1,j], mag[i+1,j]]
            else:  # 135 degrees
                neighbors = [mag[i-1,j-1], mag[i+1,j+1]]
            
            # Keep pixel if it's maximum in gradient direction
            if mag[i,j] >= max(neighbors):
                output[i,j] = mag[i,j]
                
    return output

def hysteresis_thresholding(img, high_threshold, low_threshold):
    """Applies hysteresis thresholding to get strong and weak edges."""
    rows, cols = img.shape
    
    # Get strong and weak edges
    strong = img > high_threshold
    weak = (img <= high_threshold) & (img >= low_threshold)
    
    # Initialize output with strong edges
    output = np.zeros_like(img)
    output[strong] = 255
    
    # Check weak edges
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            if weak[i,j]:
                # Check 8-connected neighbors
                neighborhood = strong[i-1:i+2, j-1:j+2]
                if np.any(neighborhood):  # If any strong edge in neighborhood
                    output[i,j] = 255
                    
    return output

def edge_detector(image, high_t=30, low_t=10, blur_before=False):
    """Canny edge detector implementation."""
    if blur_before:
        image = cv2.GaussianBlur(image, (3,3), 0)
        
    # Sobel operator
    sobel_mag, sobel_dir = sobel(image)
    
    # Non-maximum suppression
    suppressed = non_maximum_suppression(sobel_mag, sobel_dir)
    
    # Hysteresis thresholding
    edges = hysteresis_thresholding(suppressed, high_t, low_t)
    
    return edges, sobel_dir





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
        ascii_chars = "#@0oc:. "
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
                
    def generate_ascii(self, img):

        # Get original dimensions and calculate new size
        height, width = img.shape[:2]

        new_width = width // self.grid_size
        new_height = height // self.grid_size
        
        
        # Convert to grayscale for ASCII mapping
        img_greyscale  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized_gray = cv2.resize(img_greyscale,(new_width,new_height), cv2.INTER_AREA)
        
        
        # Vectorized quantization for ASCII characters
        char_indices = np.multiply(resized_gray, self.num_ascii_chars / 256, 
                                dtype=np.float32).astype(np.int32)
        char_indices = np.clip(char_indices, 0, self.num_ascii_chars - 1)
        
        
        

        if self.settings["use_edge"]:
            # calculate edges
            edges,sobel_dir = sobel(difference_of_gaussian(img_greyscale))
            
            
            # resize edges
            downsampler = FeaturePreservingDownsampler()
            edges = downsampler.downsample(edges,np.log2(self.grid_size).astype(np.uint8))
            sobel_dir = cv2.resize(sobel_dir, (new_width,new_height), cv2.INTER_LINEAR)
            # display(edges)
            edge_mask = edges > np.percentile(edges,90)
            # display(edges)
            angle_to_index = sobel_dir[edge_mask]
            
            # take mod 4 to get edges from 0-180 degrees
            edge_char_indices =  self.num_ascii_chars + (angle_to_index % 4)
            
            # Replace normal indices with edge indices where edge_mask is True
            char_indices[edge_mask] = edge_char_indices
        
        # Create output image (white background)
        output_height = new_height * self.grid_size
        output_width = new_width * self.char_width
        output_img = np.full((output_height, output_width, 3), 
                           255, dtype=np.uint8)
        
        
        # Paste colored characters
        if self.settings["use_color"]:
            resized_color = resized_color = cv2.resize(img, (new_width, new_height), 
                                 interpolation=cv2.INTER_AREA)
            self._paste_characters(output_img, char_indices, resized_color)
        else:
            self._paste_characters(output_img,char_indices)
            

        
        return output_img

def generate_ascii_image_sprite(img, grid_size=8,settings={
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
