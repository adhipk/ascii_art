
from typing import Dict, Any, Optional, Tuple, List
import argparse
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from ..filters.dog_filter import DoGFilter
from ..utils.helpers import sobel, hex_to_bgr, downsample_mode
import threading
import os
import hashlib

class SpriteASCIIGenerator:
    def __init__(self, settings: Dict[str, Any] = None):
        # Default settings
        default_settings = {
            "use_edge": False,
            "edge_threshold": 90,
            "use_color": False,
            "font_size": 8,
            "sharpness": 100,
            "white_point": 2,
        }
        
        # Merge provided settings with defaults
        if settings is None:
            settings = {}
        self.settings = {**default_settings, **settings}
        
        self.grid_size = self.settings['font_size']
        ascii_chars = "@%#*+:. "
        edge_chars = "-/|\\"
        self.num_ascii_chars = len(ascii_chars)
        self.num_edge_chars = len(edge_chars)
        self.colors = ["#242424", "#FFFFFF"]

        # Cache font
        self.font = ImageFont.truetype(font="fonts/dogica.ttf", size=self.grid_size)

        # Generate and cache sprite sheet
        self.char_sprite_sheet, self.char_width = self._generate_cached_sprite_sheet(ascii_chars + edge_chars)

        # Pre-compute character masks
        self.char_masks = self._precompute_cached_char_masks(self.char_sprite_sheet, len(ascii_chars + edge_chars))

    def _generate_cached_sprite_sheet(self, chars: str) -> Tuple[np.ndarray, int]:
        """
        Generate or load cached sprite sheet for characters.
        
        Args:
            chars (str): String of characters to include in sprite sheet
        
        Returns:
            tuple: (sprite_sheet, char_width)
        """
        cache_dir = os.path.join(os.path.dirname(__file__), 'assets')
        os.makedirs(cache_dir, exist_ok=True)

        chars_hash = hashlib.md5(chars.encode()).hexdigest()
        cache_file = os.path.join(cache_dir, f'sprite_sheet_{chars_hash}.npz')

        if os.path.exists(cache_file):
            try:
                cached_data = np.load(cache_file)
                sprite_sheet = cached_data['sprite_sheet']
                char_width = int(cached_data['char_width'])
                return sprite_sheet, char_width
            except Exception as e:
                print(f"Error loading cached sprite sheet: {e}")

        sprite_sheet, char_width = self._generate_sprite_sheet(chars)

        try:
            np.savez_compressed(cache_file, sprite_sheet=sprite_sheet, char_width=char_width)
            print(f"Cached sprite sheet to: {cache_file}")
        except Exception as e:
            print(f"Error caching sprite sheet: {e}")

        return sprite_sheet, char_width

    def _precompute_cached_char_masks(self, sprite_sheet: np.ndarray, num_chars: int) -> List[np.ndarray]:
        """
        Generate or load cached character masks.
        
        Args:
            sprite_sheet (np.ndarray): Sprite sheet image
            num_chars (int): Number of characters to process
        
        Returns:
            list: List of boolean masks for each character
        """
        cache_dir = os.path.join(os.path.dirname(__file__), 'assets')
        os.makedirs(cache_dir, exist_ok=True)

        sheet_hash = hashlib.md5(sprite_sheet.tobytes()).hexdigest()
        cache_file = os.path.join(cache_dir, f'char_masks_{sheet_hash}.npz')

        if os.path.exists(cache_file):
            try:
                cached_data = np.load(cache_file)
                char_masks = np.array([cached_data[f'mask_{i}'] for i in range(num_chars)])
                return char_masks
            except Exception as e:
                print(f"Error loading cached character masks: {e}")

        char_masks = self._precompute_char_masks(sprite_sheet, num_chars)

        try:
            mask_dict = {f'mask_{i}': mask for i, mask in enumerate(char_masks)}
            np.savez_compressed(cache_file, **mask_dict)
            print(f"Cached character masks to: {cache_file}")
        except Exception as e:
            print(f"Error caching character masks: {e}")

        return char_masks

    def _generate_sprite_sheet(self, char_list: str) -> Tuple[np.ndarray, int]:
        """Generate a sprite sheet containing all ASCII characters"""
        num_chars = len(char_list)
        temp_img = Image.new('RGB', (self.grid_size * 2, self.grid_size), 'white')
        temp_draw = ImageDraw.Draw(temp_img)
        char_width = temp_draw.textlength("@", font=self.font)

        sheet_width = int(char_width * num_chars)
        sprite_img = Image.new('RGB', (sheet_width, self.grid_size), 'white')
        sprite_draw = ImageDraw.Draw(sprite_img)

        for i, char in enumerate(char_list):
            sprite_draw.text((i * char_width, 0), char, fill='black', font=self.font)

        sprite_sheet = cv2.cvtColor(np.array(sprite_img), cv2.COLOR_RGB2GRAY)
        sprite_sheet = (sprite_sheet < 128).astype(np.uint8) * 255

        return sprite_sheet, int(char_width)

    def _precompute_char_masks(self, sprite_sheet: np.ndarray, num_chars: int) -> List[np.ndarray]:
        """Pre-compute binary masks for all characters"""
        masks = []
        for i in range(num_chars):
            sprite_x = int(i * self.char_width)
            char_sprite = sprite_sheet[:, sprite_x:sprite_x + self.char_width]
            masks.append(char_sprite > 0)
        return masks

    def _paste_characters(self, output_img: np.ndarray, char_indices: np.ndarray) -> None:
        """
        Paste char masks into output image
        
        Args:
            output_img: Target image array (modified in-place)
            char_indices: 2D array of character indices
        """
        height, width = char_indices.shape
        new_height, new_width = height * self.grid_size, width * self.char_width

        char_mask = self.char_masks[char_indices]
        char_mask = char_mask.transpose(0, 2, 1, 3).reshape(new_height, new_width)
        bg_color = hex_to_bgr(self.colors[0])
        output_img[char_mask] = bg_color

    def process_brightness(self, img: np.ndarray, new_width: int, new_height: int, results: List[Any], index: int) -> None:
        resized_gray = cv2.cvtColor(cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        char_indices = np.multiply(resized_gray, self.num_ascii_chars / 256, dtype=np.float32).astype(np.int32)
        char_indices = np.clip(char_indices, 0, self.num_ascii_chars - 1)
        results[index] = char_indices

    def process_edges(self, img: np.ndarray, results: List[Any], index: int) -> None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), cv2.BORDER_DEFAULT)
        dog_filter = DoGFilter()
        sobel_mag, sobel_dir = sobel(dog_filter.difference_of_gaussian(blurred))

        edge_threshold = np.percentile(sobel_mag, self.settings["edge_threshold"])

        sobel_mag_down = downsample_mode(sobel_mag, self.grid_size)
        edge_mask = sobel_mag_down > edge_threshold
        sobel_dir_down = downsample_mode((sobel_dir // (np.pi / 4)) % self.num_edge_chars, self.grid_size).astype(np.uint8)

        edge_indices = self.num_ascii_chars + sobel_dir_down[edge_mask]
        results[index] = (edge_mask, edge_indices)

    def generate_ascii(self, img: np.ndarray) -> np.ndarray:
        results = [None, None]
        threads = []
        height, width = img.shape[:2]
        new_width = width // self.grid_size
        new_height = height // self.grid_size

        threads.append(threading.Thread(target=self.process_brightness, args=(img, new_width, new_height, results, 0)))
        if self.settings['use_edge']:
            threads.append(threading.Thread(target=self.process_edges, args=(img, results, 1)))

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        brightness_indices = results[0]
        combined_indices = brightness_indices.copy()
        if results[1] is not None:
            edge_mask, edge_indices = results[1]
            combined_indices[edge_mask] = edge_indices

        output_height = new_height * self.grid_size
        output_width = new_width * self.char_width
        fg_color = hex_to_bgr(self.colors[1])
        output_img = np.full((output_height, output_width, 3), fg_color, dtype=np.uint8)

        self._paste_characters(output_img, combined_indices)
        return output_img

def generate_ascii_image_sprite(img: np.ndarray, settings: Dict[str, Any]) -> np.ndarray:
    """Convenience function to generate colored ASCII art"""
    generator = SpriteASCIIGenerator(settings)
    return generator.generate_ascii(img)


def main() -> None:
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
    rescaled_img = generate_ascii_image_sprite(img, settings)

    result = cv2.imwrite("result.png", rescaled_img)
    print(result)

if __name__ == "__main__":
    main()