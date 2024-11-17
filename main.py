
from webbrowser import get
from PIL import Image,ImageDraw,ImageFont,ImageFilter, ImageChops
import argparse
import numpy as np
from scipy.signal import convolve2d

# ASCII characters ordered by density from darkest to lightest
ASCII_CHARS = "@#H*+=-:. "
EDGE_CHARS = ['-','|','/','\\']
def difference_of_gausian(image_obj, low_sigma=1, high_sigma=10):   
    low_pass = image_obj.filter(ImageFilter.GaussianBlur(radius=low_sigma))
    high_pass = image_obj.filter(ImageFilter.GaussianBlur(radius=high_sigma))
    return ImageChops.difference(high_pass, low_pass)
    
def sobel(image_obj):
    """Applies Sobel operator for edge detection."""
    image_array = np.array(image_obj, dtype=np.float32)
    dx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    dy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    sobel_x = np.abs(convolve2d(image_array, dx, mode='same', boundary='symm'))
    sobel_y = np.abs(convolve2d(image_array, dy, mode='same', boundary='symm'))
    return np.hypot(sobel_x, sobel_y), np.arctan2(sobel_y, sobel_x)
    
def downsample_and_quantize(image_path):
    # Open and resize the image in one step

    img = Image.open(image_path)
    greyscale_img = img.convert('L')
    
    downscale = 1
    # Calculate dimensions
    width, height = greyscale_img.size
    grid_size = 10
    width = width // downscale
    height = height // downscale
    new_width = width // grid_size
    new_height = height // grid_size

    # calculate edges
    sobel_mag, sobel_dir = sobel(greyscale_img)
    
    #downscale
    sobel_mag = sobel_mag.resize((new_width, new_height), Image.NEAREST)
    sobel_dir = sobel_dir.resize((new_width, new_height), Image.NEAREST)
    greyscale_img = greyscale_img.resize((new_width, new_height), Image.NEAREST)

    
    
    
    
    # Convert to numpy array and quantize in one step
    quantized_array = np.array(greyscale_img) // 26
    
    # Create output image and get drawing context
    ascii_img = Image.new('L', (width, height), "white")
    draw = ImageDraw.Draw(ascii_img)
    
    # Load font once outside the loop
    font = ImageFont.truetype("FiraCode-Regular.ttf", size=grid_size,encoding='utf-8')  # Reduced font size for better performance
    
    
    chars = np.array(list(ASCII_CHARS))
    ascii_chars = chars[quantized_array]
    edges = np.array(list(EDGE_CHARS))
    ascii_edges = np.where(sobel_mag > 0, edges[(sobel_dir/(np.pi/ 4)).astype(int)] ," ")
    ascii_edges = np.where(sobel_mag > 0, edges[(sobel_dir/(np.pi/ 4)).astype(int)] ," ")

    
    
    for y in range(new_height):
        for x in range(new_width):
            char_to_draw = ascii_edges[y,x]
            draw.text(
                (x * grid_size, y * grid_size),
                str(char_to_draw),
                font=font
            )
    
    return ascii_img

def main():
    parser = argparse.ArgumentParser(description="Generate ASCII art from an image.")
    parser.add_argument("image_path", help="Path to the image file.")
    args = parser.parse_args()
    image_path = args.image_path

    rescaled_img = downsample_and_quantize(image_path)
    rescaled_img.save("outputs/result.png")

if __name__ == "__main__":
    main()
