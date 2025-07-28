#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image cropping tool
Crop all images in the input folder to the left half and overwrite the original file
"""

import os
import sys
from PIL import Image
import argparse
from pathlib import Path


def crop_images_left_half(input_folder):
    """
    Crop all images in the input folder to the left half and overwrite the original file
    
    Args:
        input_folder (str): input folder path
    """
    # supported image formats
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    # get all files in input folder
    input_path = Path(input_folder)
    if not input_path.exists():
        print(f"Failed to get input folder '{input_folder}'")
        return
    
    # statistics
    total_files = 0
    processed_files = 0
    failed_files = 0
    
    print(f"Start processing folder: {input_folder}")
    print("⚠️  Warning: This will directly overwrite the original file!")
    print("-" * 50)
    
    # traverse all files
    for file_path in input_path.rglob('*'):
        if file_path.is_file():
            # check file extension
            file_ext = file_path.suffix.lower()
            if file_ext in supported_formats:
                total_files += 1
                
                try:
                    # open image
                    with Image.open(file_path) as img:
                        # get image size
                        width, height = img.size
                        
                        # calculate left half size
                        left_half_width = width // 2
                        
                        # crop left half (left, top, right, bottom)
                        cropped_img = img.crop((0, 0, left_half_width, height))
                        
                        # overwrite original file
                        cropped_img.save(file_path, quality=95)
                        
                        processed_files += 1
                        print(f"✓ Successfully processed: {file_path.name} ({width}x{height} -> {left_half_width}x{height})")
                        
                except Exception as e:
                    failed_files += 1
                    print(f"✗ Failed to process: {file_path.name} - Error: {str(e)}")
    
    # print statistics
    print("-" * 50)
    print(f"Processing completed!")
    print(f"Total files: {total_files}")
    print(f"Successfully processed: {processed_files}")
    print(f"Failed to process: {failed_files}")
    
    if processed_files > 0:
        print(f"All images have been cropped to left half and overwritten")


def main():
    parser = argparse.ArgumentParser(description='Image cropping tool - crop images to left half and overwrite original file')
    parser.add_argument('input_folder', help='input folder path')
    parser.add_argument('--recursive', '-r', action='store_true', 
                       help='process subfolders recursively')
    
    args = parser.parse_args()
    
    # check if input folder exists
    if not os.path.exists(args.input_folder):
        print(f"Failed to get input folder '{args.input_folder}'")
        sys.exit(1)
    
    # execute cropping
    crop_images_left_half(args.input_folder)


if __name__ == "__main__":
    # if no command line arguments, provide interactive usage
    if len(sys.argv) == 1:
        print("=" * 30)
        print("⚠️  Warning: This will directly overwrite the original file!")
        print()
        
        input_folder = "/home/yimingqin/workspace/threestudio/eval_outputs/clip_score_img"
        
        if input_folder:
            crop_images_left_half(input_folder)
        else:
            print("Failed to get input folder")
    else:
        main()
