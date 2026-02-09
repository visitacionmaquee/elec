import os
import cv2
import pytest
from src.beautify_final import smooth_skin, save_image, load_images

INPUT_DIR = "input"
OUTPUT_DIR = "output"


def get_test_image():
    assert os.path.exists(INPUT_DIR), "input folder does not exist"

    images = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    assert len(images) > 0, "No images found in input folder"

    return images[0]


def test_smooth_skin_with_real_image():
    filename = get_test_image()
    img_path = os.path.join(INPUT_DIR, filename)

    img = cv2.imread(img_path)
    assert img is not None, "Failed to load input image"

    result = smooth_skin(img)

    assert result is not None
    assert result.shape == img.shape
    assert result.dtype == img.dtype


def test_smooth_skin_with_none():
    with pytest.raises(ValueError, match="Input image is None"):
        smooth_skin(None)


def test_save_image():
    filename = get_test_image()
    img_path = os.path.join(INPUT_DIR, filename)
    img = cv2.imread(img_path)

    result = smooth_skin(img)
    output_path = save_image(OUTPUT_DIR, filename, result)

    assert os.path.exists(output_path)


def test_load_images():
    filename = get_test_image()

    results = list(load_images(INPUT_DIR, [filename]))

    assert len(results) == 1
    name, img = results[0]
    assert name == filename
    assert img is not None
