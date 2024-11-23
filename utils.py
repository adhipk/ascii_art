import cv2
import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
from PIL import Image,ImageDraw, ImageFont

ASCII_CHARS = "@#H*+:. "
EDGE_CHARS = ['-','|','/','\\']



def difference_of_gaussian(image_obj, sigma=2,k=1.6,white_point=100,shapness=2):
    g_sigma = gaussian_filter(image_obj, sigma)
    g_k = gaussian_filter(image_obj, k*sigma)
    dog = (1+shapness)*g_sigma - shapness*g_k
    dog_thresholded = np.where(dog > white_point, 255.0, 0.0)
    return dog_thresholded
    
    
    
def sobel(image_array,treshold=80):
    """Applies Sobel operator for edge detection."""
    
    dx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    dy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    sobel_x = convolve2d(image_array, dx, mode='same', boundary='symm')
    sobel_y = convolve2d(image_array, dy, mode='same', boundary='symm')
    return  (np.hypot(sobel_x, sobel_y), np.arctan2(sobel_y, sobel_x))


def generate_ascii_image(img, grid_size=10, ASCII_CHARS = "@MH*+:. "):
    # Load the image in grayscale
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width,_ = img.shape
    new_width, new_height = width // grid_size, height // grid_size

    quantized_img_resized = cv2.resize(grey_img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    # Quantize the grayscale image
    quantized_array = quantized_img_resized // 32

    # Precompute ASCII and edge thresholds
    ascii_chars = np.array(list(ASCII_CHARS))
    ascii_art_array = ascii_chars[quantized_array]

    # Create an output image
    # Reading an image in default mode
     # Create an output image
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font_path = "fonts/dogica.ttf"
    # Load font
    font = ImageFont.truetype(font=font_path,size=grid_size)

    
    # fontScale
    fontScale = 1
    
    # Blue color in BGR
    color = (0, 0, 0)

    # Line thickness of 2 px
    thickness = 1
    y = 0
    for row in ascii_art_array:
        row = "".join(row)
        draw.text((0, y), row, fill=color, font=font)
        y+=grid_size
    
    # 
    return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

class SpriteASCIIGenerator:
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.ascii_chars = "@#MH*+o:. "
        self.num_chars = len(self.ascii_chars)
        
        # Cache font
        self.font = ImageFont.truetype(font="fonts/dogica.ttf", size=self.grid_size)
        
        # Generate and cache sprite sheet
        self.sprite_sheet, self.char_width = self._generate_sprite_sheet()
        
        # Pre-compute character masks
        self.char_masks = self._precompute_char_masks()
        
    def _generate_sprite_sheet(self):
        """Generate a sprite sheet containing all ASCII characters"""
        # Measure character width
        temp_img = Image.new('RGB', (self.grid_size * 2, self.grid_size), 'white')
        temp_draw = ImageDraw.Draw(temp_img)
        char_width = temp_draw.textlength("@", font=self.font)
        
        # Create sprite sheet
        sheet_width = int(char_width * self.num_chars)
        sprite_img = Image.new('RGB', (sheet_width, self.grid_size), 'white')
        sprite_draw = ImageDraw.Draw(sprite_img)
        
        # Draw characters
        for i, char in enumerate(self.ascii_chars):
            sprite_draw.text((i * char_width, 0), char, fill='black', font=self.font)
        
        # Convert to binary mask
        sprite_sheet = cv2.cvtColor(np.array(sprite_img), cv2.COLOR_RGB2GRAY)
        sprite_sheet = (sprite_sheet < 128).astype(np.uint8) * 255
        
        return sprite_sheet, int(char_width)
    
    def _precompute_char_masks(self):
        """Pre-compute binary masks for all characters"""
        masks = []
        for i in range(self.num_chars):
            sprite_x = int(i * self.char_width)
            char_sprite = self.sprite_sheet[:, sprite_x:sprite_x + self.char_width]
            masks.append(char_sprite > 0)
        return masks
    
    def _paste_colored_characters(self, output_img, char_indices, color_data):
        """Paste characters with their corresponding colors"""
        height, width = char_indices.shape
        
        for i in range(height):
            y = i * self.grid_size
            if y + self.grid_size > output_img.shape[0]:
                break
                
            for j in range(width):
                x = j * self.char_width
                if x + self.char_width > output_img.shape[1]:
                    break
                
                char_idx = char_indices[i, j]
                mask = self.char_masks[char_idx]
                color = color_data[i, j]
                
                # Create 3D mask for color channels

                
                # Apply color to the character
                region = output_img[y:y + self.grid_size, x:x + self.char_width]
                    # Apply color to each channel separately
                for c in range(3):
                    region[..., c][mask] = color[c]
                    
    def _paste_greyscale_characters(self, output_img, char_indices):
        """Vectorized version of character pasting"""
        height, width = char_indices.shape
        
        for i in range(height):
            for j in range(width):
                y = i * self.grid_size
                x = j * self.char_width
                
                if y + self.grid_size <= output_img.shape[0] and \
                x + self.char_width <= output_img.shape[1]:
                    char_idx = char_indices[i, j]
                    mask = self.char_masks[char_idx]
                    output_img[y:y + self.grid_size, 
                            x:x + self.char_width][mask] = 0
                
    def generate_ascii(self, img,use_color = False):
        # Get original dimensions and calculate new size
        height, width = img.shape[:2]
        new_width = width // self.grid_size
        new_height = height // self.grid_size
        
        # Resize with color preservation
        resized_color = cv2.resize(img, (new_width, new_height), 
                                 interpolation=cv2.INTER_AREA)
        
        # Convert to grayscale for ASCII mapping
        resized_gray = cv2.cvtColor(resized_color, cv2.COLOR_BGR2GRAY)
        
        # Vectorized quantization for ASCII characters
        char_indices = np.multiply(resized_gray, self.num_chars / 256,
                                 dtype=np.float32).astype(np.int32)
        char_indices = np.clip(char_indices, 0, self.num_chars - 1)
        
        # Create output image (white background)
        output_height = new_height * self.grid_size
        output_width = new_width * self.char_width
        output_img = np.full((output_height, output_width, 3), 
                           255, dtype=np.uint8)
        
        # Paste colored characters
        if use_color:
            self._paste_colored_characters(output_img, char_indices, resized_color)
        else:
            self._paste_greyscale_characters(output_img,char_indices)
        
        return output_img

def generate_ascii_image_sprite(img, grid_size=10):
    """Convenience function to generate colored ASCII art"""
    generator = SpriteASCIIGenerator(grid_size=grid_size)
    return generator.generate_ascii(img)
