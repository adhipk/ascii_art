
import argparse
import numpy as np
import cv2
from PIL import Image,ImageDraw, ImageFont
from utils import difference_of_gaussian,sobel, hex_to_bgr, downsample_mode
import threading
import os
import cv2
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
        ascii_chars = "@%#*+:. "
        # ascii_chars = " "
        edge_chars = "-/|\\"
        self.num_ascii_chars = len(ascii_chars)
        self.num_edge_chars = len(edge_chars)
        self.colors = ["#242424","#FFFFFF"]
        self.settings = settings
        
        # Cache font
        self.font = ImageFont.truetype(font="fonts/dogica.ttf", size=self.grid_size)
        
        # Generate and cache sprite sheet
        self.char_sprite_sheet, self.char_width = self._generate_cached_sprite_sheet(ascii_chars+edge_chars)
        
        # Pre-compute character masks
        self.char_masks = self._precompute_cached_char_masks(self.char_sprite_sheet,len(ascii_chars+edge_chars))

    def _generate_cached_sprite_sheet(self, chars):
        """
        Generate or load cached sprite sheet for characters.
        
        Args:
            chars (str): String of characters to include in sprite sheet
        
        Returns:
            tuple: (sprite_sheet, char_width)
        """
        # Define cache file path
        cache_dir = os.path.join(os.path.dirname(__file__), 'assets')
        os.makedirs(cache_dir, exist_ok=True)
        
        # Create a unique filename based on the character set
        # Use a hash of the characters to create a consistent filename
        import hashlib
        chars_hash = hashlib.md5(chars.encode()).hexdigest()
        cache_file = os.path.join(cache_dir, f'sprite_sheet_{chars_hash}.npz')
        
        # Check if cached file exists
        if os.path.exists(cache_file):
            try:
                # Load cached data
                cached_data = np.load(cache_file)
                sprite_sheet = cached_data['sprite_sheet']
                char_width = int(cached_data['char_width'])
                
                # print(f"Loaded sprite sheet from cache: {cache_file}")
                return sprite_sheet, char_width
            except Exception as e:
                print(f"Error loading cached sprite sheet: {e}")
        
        # If no valid cache, generate sprite sheet
        sprite_sheet, char_width = self._generate_sprite_sheet(chars)
        
        # Cache the generated sprite sheet
        try:
            np.savez_compressed(cache_file, 
                                sprite_sheet=sprite_sheet, 
                                char_width=char_width)
            print(f"Cached sprite sheet to: {cache_file}")
        except Exception as e:
            print(f"Error caching sprite sheet: {e}")
        
        return sprite_sheet, char_width

    def _precompute_cached_char_masks(self, sprite_sheet, num_chars):
        """
        Generate or load cached character masks.
        
        Args:
            sprite_sheet (np.ndarray): Sprite sheet image
            num_chars (int): Number of characters to process
        
        Returns:
            list: List of boolean masks for each character
        """
        # Define cache file path
        cache_dir = os.path.join(os.path.dirname(__file__), 'assets')
        os.makedirs(cache_dir, exist_ok=True)
        
        # Create a unique filename based on the sprite sheet
        import hashlib
        sheet_hash = hashlib.md5(sprite_sheet.tobytes()).hexdigest()
        cache_file = os.path.join(cache_dir, f'char_masks_{sheet_hash}.npz')
        
        # Check if cached file exists
        if os.path.exists(cache_file):
            try:
                # Load cached masks
                cached_data = np.load(cache_file)
                char_masks = [cached_data[f'mask_{i}'] for i in range(num_chars)]
                
                # print(f"Loaded character masks from cache: {cache_file}")
                return char_masks
            except Exception as e:
                print(f"Error loading cached character masks: {e}")
        
        # If no valid cache, generate character masks
        char_masks = self._precompute_char_masks(sprite_sheet, num_chars)
        
        # Cache the generated masks
        try:
            # Save masks individually to avoid issues with variable-sized arrays
            mask_dict = {f'mask_{i}': mask for i, mask in enumerate(char_masks)}
            np.savez_compressed(cache_file, **mask_dict)
            print(f"Cached character masks to: {cache_file}")
        except Exception as e:
            print(f"Error caching character masks: {e}")
        
        return char_masks
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
        Optimized character pasting with vectorized operations.
        
        Args:
            output_img: Target image array
            char_indices: 2D array of character indices
            color_data: Optional 3D array of RGB colors for each character position
        """
        height, width = char_indices.shape
        
        # Pre-calculate valid dimensions
        max_y = min(height * self.grid_size, output_img.shape[0])
        max_x = min(width * self.char_width, output_img.shape[1])
        
        # Compute valid positions
        y_max = len(range(0, max_y, self.grid_size))
        x_max = len(range(0, max_x, self.char_width))
        
        # Prepare character masks and indices for vectorized processing
        valid_char_indices = char_indices[:y_max, :x_max]
        
        if color_data is not None:
            # Color mode - vectorized color application
            color_region = color_data[:y_max, :x_max]
            
            for char_idx in np.unique(valid_char_indices):
                # Create mask for this specific character
                char_mask = self.char_masks[char_idx]
                
                # Find positions of this character
                idx_positions = np.argwhere(valid_char_indices == char_idx)
                
                for y, x in idx_positions:
                    # Compute image region coordinates
                    y_start = y * self.grid_size
                    x_start = x * self.char_width
                    
                    # Get region and apply color through mask
                    region = output_img[y_start:y_start + self.grid_size, 
                                        x_start:x_start + self.char_width]
                    
                    # Vectorized color application
                    region_color = color_region[y, x]
                    for c in range(3):
                        region[..., c][char_mask] = region_color[c]
        
        else:
            # Greyscale mode - precompute background and foreground colors
            bg_color, fg_color = [hex_to_bgr(value) for value in self.colors]
            
            for char_idx in np.unique(valid_char_indices):
                # Create mask for this specific character
                char_mask = self.char_masks[char_idx]
                
                # Find positions of this character
                idx_positions = np.argwhere(valid_char_indices == char_idx)
                
                for y, x in idx_positions:
                    # Compute image region coordinates
                    y_start = y * self.grid_size
                    x_start = x * self.char_width
                    
                    # Get region and apply colors
                    region = output_img[y_start:y_start + self.grid_size, 
                                        x_start:x_start + self.char_width]
                    
                    # Vectorized color application
                    for c in range(3):
                        region[..., c][char_mask] = bg_color[c]
                        region[..., c][~char_mask] = fg_color[c]
        
        return output_img


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
        if self.settings['use_edge']:
            threads.append(threading.Thread(target=self.process_edges, args=(img, results, 1)))

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Combine results
        brightness_indices = results[0]
        combined_indices = brightness_indices.copy()
        if results[1] is not None:
            edge_mask, edge_indices = results[1]
            
            
            combined_indices[edge_mask] = edge_indices

        # Create output image (white background)
        output_height = new_height * self.grid_size
        output_width = new_width * self.char_width
        output_img = np.full((output_height, output_width, 3), 
                           255, dtype=np.uint8)
        
        # Paste chars
        
        self._paste_characters(output_img,combined_indices)
        return output_img


def generate_ascii_image_sprite(img, grid_size, settings):
    """Convenience function to generate colored ASCII art"""
    generator = SpriteASCIIGenerator(grid_size=grid_size,settings=settings)
    return generator.generate_ascii(img)


    

def main():
    parser = argparse.ArgumentParser(description="Generate ASCII art from an image.")
    parser.add_argument("image_path", help="Path to the image file.")
    args = parser.parse_args()
    image_path = args.image_path
    img = cv2.imread(image_path)
    settings = {
        "use_edge": True,
        "edge_threshold": 90,
        "use_color": False,
        "font_size": 8,
        "sharpness": 5,
        "white_point": 128
    }
    rescaled_img = generate_ascii_image_sprite(img,8,settings)
    
    result = cv2.imwrite("result.png",rescaled_img)
    print(result)
if __name__ == "__main__":
    main()
