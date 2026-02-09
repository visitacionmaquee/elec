# src/process_all.py
"""
Main processing script that applies all filters to all images in input folder
"""
import cv2
import os
import glob
from pathlib import Path

def process_all_images():
    """Process all images in input folder with all filters"""
    input_dir = Path("input")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Import all processing functions
    # Beautify
    exec(open("src/beautify_final.py").read())
    # Brighten
    exec(open("src/brighten_final.py").read())
    # Cartoonify
    exec(open("src/cartoonify_final.py").read())
    # Slow shutter
    exec(open("src/slowshutter_final.py").read())
    
    # Get all images from input folder
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(str(input_dir / ext)))
    
    if not image_files:
        print("No images found in input folder")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    for image_path in image_files:
        filename = Path(image_path).name
        print(f"Processing: {filename}")
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"  Failed to load: {filename}")
            continue
        
        # Apply all filters
        try:
            # Beautify
            beautified = smooth_skin(img)
            cv2.imwrite(str(output_dir / f"beautified_{filename}"), beautified)
            
            # Brighten
            brightened = apply_clahe(img)
            cv2.imwrite(str(output_dir / f"brightened_{filename}"), brightened)
            
            # Cartoonify
            cartoonified = cartoonify(img)
            cv2.imwrite(str(output_dir / f"cartoonified_{filename}"), cartoonified)
            
            # Slow shutter
            slow_shutter = apply_slow_shutter(img)
            cv2.imwrite(str(output_dir / f"slowshutter_{filename}"), slow_shutter)
            
            print(f"  Successfully processed: {filename}")
            
        except Exception as e:
            print(f"  Error processing {filename}: {e}")

if __name__ == "__main__":
    process_all_images()