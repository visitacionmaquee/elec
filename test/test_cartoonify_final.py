# test/test_cartoonify_final.py
import cv2
import numpy as np
import pytest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
exec(open("../src/cartoonify_final.py").read())

def test_cartoonify():
    """Test cartoonify function"""
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    result = cartoonify(img)
    
    assert result is not None
    assert result.shape == img.shape
    assert result.dtype == np.uint8

def test_cartoonify_with_none():
    """Test cartoonify with None input"""
    with pytest.raises(ValueError, match="Input image is None"):
        cartoonify(None)

def test_cartoonify_edge_cases():
    """Test cartoonify with edge cases"""
    # Small image
    small_img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
    result = cartoonify(small_img)
    assert result is not None
    
    # Large image
    large_img = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
    result = cartoonify(large_img)
    assert result is not None