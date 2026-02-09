# test/test_CandidOrPosed.py
import cv2
import numpy as np
import pytest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
exec(open("../src/CandidOrPosed.py").read())

def test_blur_score():
    """Test blur score calculation"""
    # Create a clear image
    clear_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # Create a blurred image
    blurred_img = cv2.GaussianBlur(clear_img, (15, 15), 5)
    
    clear_score = blur_score(clear_img)
    blurred_score = blur_score(blurred_img)
    
    # Blurred image should have lower score
    assert blurred_score < clear_score
    
def test_classify_image_mock(mocker, tmp_path):
    """Test image classification with mocked image"""
    # Create a test image
    test_img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    img_path = tmp_path / "test.jpg"
    cv2.imwrite(str(img_path), test_img)
    
    # Mock cv2.imshow to prevent GUI issues
    mocker.patch('cv2.imshow')
    mocker.patch('cv2.waitKey')
    mocker.patch('cv2.destroyAllWindows')
    
    # Call the function
    classify_image(str(img_path))
    
    # If we get here without errors, the test passes
    assert True