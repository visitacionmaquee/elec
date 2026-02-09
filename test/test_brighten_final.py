# test/test_brighten_final.py
import cv2
import numpy as np
import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from brighten_final import apply_clahe

def test_apply_clahe():
    """Test CLAHE function"""
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    result = apply_clahe(img)
    
    assert result is not None
    assert result.shape == img.shape
    assert result.dtype == np.uint8

def test_apply_clahe_with_none():
    """Test CLAHE with None input"""
    with pytest.raises(ValueError, match="Input image is None"):
        apply_clahe(None)

def test_apply_clahe_custom_params():
    """Test CLAHE with custom parameters"""
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    result = apply_clahe(img, clip_limit=2.0, tile_grid_size=(4, 4))
    
    assert result is not None
    assert result.shape == img.shape