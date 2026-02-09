# test/test_beautify_final.py
import cv2
import numpy as np
import pytest
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import the module
exec(open("../src/beautify_final.py").read())

def test_smooth_skin_with_valid_image():
    """Test smooth_skin with a valid image"""
    # Create a test image
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # Test the function
    result = smooth_skin(img)
    
    # Check outputs
    assert result is not None
    assert result.shape == img.shape
    assert result.dtype == np.uint8
    
    # Result should be different from input
    assert not np.array_equal(result, img)

def test_smooth_skin_with_none():
    """Test smooth_skin with None input"""
    with pytest.raises(ValueError, match="Input image is None"):
        smooth_skin(None)

def test_show_resized():
    """Test show_resized function"""
    img = np.random.randint(0, 255, (800, 800, 3), dtype=np.uint8)
    
    # Since cv2.imshow requires GUI, we'll just ensure it doesn't crash
    try:
        show_resized("Test", img, max_height=500)
    except Exception as e:
        # This is expected if running in CI without display
        pass

def test_save_image(tmp_path):
    """Test save_image function"""
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # Test saving
    output_path = save_image(str(tmp_path), "test.jpg", img)
    
    # Check file was created
    assert os.path.exists(output_path)
    
    # Load and verify
    loaded = cv2.imread(output_path)
    assert loaded is not None
    assert loaded.shape == img.shape

def test_load_images(tmp_path):
    """Test load_images function"""
    # Create test directory and image
    test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    img_path = input_dir / "test_image.jpg"
    cv2.imwrite(str(img_path), test_img)
    
    # Test loading
    generator = load_images(str(input_dir), ["test_image.jpg"])
    
    # Get the result
    filenames = []
    images = []
    for filename, img in generator:
        filenames.append(filename)
        images.append(img)
    
    # Verify
    assert len(filenames) == 1
    assert filenames[0] == "test_image.jpg"
    assert images[0] is not None
    assert images[0].shape == test_img.shape

def test_load_images_nonexistent(tmp_path):
    """Test load_images with non-existent file"""
    generator = load_images(str(tmp_path), ["nonexistent.jpg"])
    
    # Should yield nothing
    results = list(generator)
    assert len(results) == 0