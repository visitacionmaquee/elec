# test/test_slowshutter_final.py
import cv2
import numpy as np
import pytest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
exec(open("../src/slowshutter_final.py").read())

def test_apply_slow_shutter():
    """Test slow shutter function"""
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    result = apply_slow_shutter(img)
    
    assert result is not None
    assert result.shape == img.shape
    assert result.dtype == np.uint8

def test_apply_slow_shutter_with_none():
    """Test slow shutter with None input"""
    with pytest.raises(ValueError, match="Input image is None"):
        apply_slow_shutter(None)

def test_apply_slow_shutter_custom_params():
    """Test slow shutter with custom parameters"""
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    result = apply_slow_shutter(
        img, 
        trail_length=50, 
        step=2, 
        direction=1, 
        blend_original=0.5
    )
    
    assert result is not None
    assert result.shape == img.shape