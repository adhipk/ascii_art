"""
Unit tests for ASCII Art Library
"""

import numpy as np
import pytest
from ascii_art import SpriteASCIIGenerator, DoGFilter

def test_sprite_ascii_generator_initialization():
    """Test that SpriteASCIIGenerator can be initialized with default settings."""
    generator = SpriteASCIIGenerator()
    assert isinstance(generator, SpriteASCIIGenerator)

def test_dog_filter_initialization():
    """Test that DoGFilter can be initialized with default settings."""
    filter = DoGFilter()
    assert isinstance(filter, DoGFilter)

def test_generate_ascii_with_dummy_image():
    """Test that SpriteASCIIGenerator can process an image without error."""
    # Create a simple dummy image
    dummy_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    generator = SpriteASCIIGenerator({"font_size": 8})  # Use default font size
    
    # This test just checks that the method doesn't raise an exception
    # The actual output is complex to validate without a real image
    try:
        ascii_img = generator.generate_ascii(dummy_img)
        assert True  # If we get here without exception, the test passes
    except Exception as e:
        pytest.fail(f"generate_ascii raised an exception: {e}")

if __name__ == "__main__":
    test_sprite_ascii_generator_initialization()
    test_dog_filter_initialization()
    test_generate_ascii_with_dummy_image()
    print("All tests passed!")