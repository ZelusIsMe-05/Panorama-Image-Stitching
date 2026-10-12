#!/usr/bin/env python3
"""
Image Preprocessing and Resizing Utility for CSC14120 Final Project.

Features:
- Validates input file formats (supports .jpg, .jpeg, .png, .bmp, .webp, .tiff; rejects .pnm, etc.)
- Performs center-crop to 16:9 aspect ratio without distortion
- Resizes to standard project resolutions (small: 1280x720, medium: 1920x1080, large: 3840x2160)
  or custom target dimensions
- Uses high-quality Lanczos resampling filter for optimal keypoint detection
"""

import os
import sys
import argparse
from typing import Tuple, Dict, List
from PIL import Image

# Supported image extensions (popular web and camera image formats)
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

# Project standard resolutions (width, height)
STANDARD_RESOLUTIONS: Dict[str, Tuple[int, int]] = {
    "small": (1280, 720),
    "medium": (1920, 1080),
    "large": (3840, 2160),
}


def is_valid_image_file(filename: str) -> bool:
    """
    Check if the file has an allowed extension (.jpg, .jpeg, .png).
    Rejects unsupported formats such as .pnm, .ppm, .pgm, .bmp, etc.
    """
    
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def center_crop_16_9(img: Image.Image) -> Image.Image:
    """
    Crops the image to a 16:9 aspect ratio around the center.
    This prevents geometric stretching and preserves subject proportions.
    """

    target_aspect = 16.0 / 9.0
    w, h = img.size
    current_aspect = w / h

    if abs(current_aspect - target_aspect) < 1e-4:
        return img

    if current_aspect > target_aspect:
        # Image is wider than 16:9 -> crop horizontal edges

        crop_width = int(round(h * target_aspect))
        left = (w - crop_width) // 2
        top = 0
        right = left + crop_width
        bottom = h
    else:
        # Image is taller than 16:9 (common in phone 4:3 photos) -> crop vertical edges

        crop_height = int(round(w / target_aspect))
        left = 0
        top = (h - crop_height) // 2
        right = w
        bottom = top + crop_height

    return img.crop((left, top, right, bottom))


def resize_image(
    input_path: str,
    target_size: Tuple[int, int],
    output_path: str,
    crop_16_9: bool = True
) -> None:
    """
    Loads an image, optionally center-crops to 16:9, resizes using Lanczos interpolation,
    and saves to the specified output path.
    """

    target_w, target_h = target_size

    with Image.open(input_path) as img:
        # Convert to RGB to discard alpha channels and ensure 3-channel output

        if img.mode != "RGB":
            img = img.convert("RGB")

        orig_w, orig_h = img.size

        # Apply 16:9 center cropping if requested
        if crop_16_9:
            img = center_crop_16_9(img)
            crop_w, crop_h = img.size
        else:
            crop_w, crop_h = orig_w, orig_h

        # Use Lanczos resampling (high-quality downscaling / antialiasing)
        resampling_filter = getattr(Image.Resampling, "LANCZOS", Image.LANCZOS)
        resized_img = img.resize((target_w, target_h), resample=resampling_filter)

        # Create output parent directory if it does not exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # Save with high quality
        resized_img.save(output_path, quality=95)
        print(f"[OK] {input_path} ({orig_w}x{orig_h}) -> cropped ({crop_w}x{crop_h}) -> {output_path} ({target_w}x{target_h})")


def parse_resolution_arg(arg: str) -> Dict[str, Tuple[int, int]]:
    """
    Parses the resolution argument. Accepts:
    - 'small', 'medium', 'large'
    - 'all' (produces small, medium, and large)
    - Custom format 'WIDTHxHEIGHT' (e.g., '1280x720')
    """

    arg_lower = arg.lower()
    if arg_lower == "all":
        return STANDARD_RESOLUTIONS
    if arg_lower in STANDARD_RESOLUTIONS:
        return {arg_lower: STANDARD_RESOLUTIONS[arg_lower]}

    if "x" in arg_lower:
        try:
            parts = arg_lower.split("x")
            w, h = int(parts[0]), int(parts[1])
            if w <= 0 or h <= 0:
                raise ValueError
            return {f"{w}x{h}": (w, h)}
        except ValueError:
            pass

    raise argparse.ArgumentTypeError(
        f"Invalid resolution '{arg}'. Choose from 'small', 'medium', 'large', 'all', or 'WIDTHxHEIGHT' (e.g. 1280x720)."
    )


def process_single_file(
    input_path: str,
    output_dir: str,
    resolutions: Dict[str, Tuple[int, int]],
    crop_16_9: bool
) -> None:
    """Process a single valid image file."""

    if not os.path.isfile(input_path):
        print(f"[Error] File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if not is_valid_image_file(input_path):
        ext = os.path.splitext(input_path)[1]
        print(
            f"[Error] Unsupported file format '{ext}' for file '{input_path}'.\n"
            f"        Allowed formats are: {', '.join(sorted(ALLOWED_EXTENSIONS))}.\n"
            f"        Formats such as .pnm, .ppm, .pgm are strictly rejected.",
            file=sys.stderr,
        )
        sys.exit(1)

    base_name = os.path.splitext(os.path.basename(input_path))[0]

    for res_name, size in resolutions.items():
        # Place inside subfolder named after the resolution if multiple, or directly in output_dir
        dest_folder = os.path.join(output_dir, res_name) if len(resolutions) > 1 else output_dir
        out_filename = f"{base_name}.jpg"
        out_path = os.path.join(dest_folder, out_filename)
        resize_image(input_path, size, out_path, crop_16_9=crop_16_9)


def process_directory(
    input_dir: str,
    output_dir: str,
    resolutions: Dict[str, Tuple[int, int]],
    crop_16_9: bool
) -> None:
    """Scan and process all valid images in a directory."""

    if not os.path.isdir(input_dir):
        print(f"[Error] Directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    entries = sorted(os.listdir(input_dir))
    valid_files = [f for f in entries if is_valid_image_file(f)]
    rejected_files = [
        f for f in entries
        if not is_valid_image_file(f) and not os.path.isdir(os.path.join(input_dir, f))
    ]

    if rejected_files:
        print(f"[Warning] Skipped {len(rejected_files)} non-allowed file(s): {rejected_files}")

    if not valid_files:
        print(f"[Warning] No valid images ({', '.join(ALLOWED_EXTENSIONS)}) found in {input_dir}.")
        return

    print(f"Found {len(valid_files)} valid image(s) to process.\n")
    for fname in valid_files:
        full_path = os.path.join(input_dir, fname)
        base_name = os.path.splitext(fname)[0]
        for res_name, size in resolutions.items():
            dest_folder = os.path.join(output_dir, res_name)
            out_path = os.path.join(dest_folder, f"{base_name}.jpg")
            resize_image(full_path, size, out_path, crop_16_9=crop_16_9)


def main():
    parser = argparse.ArgumentParser(
        description="Dataset Preparation Tool: Crop to 16:9 and Resize Images for CSC14120 Final Project."
    )
    parser.add_argument(
        "-i", "--input",
        help="Path to an input image file or an input folder containing raw images."
    )
    parser.add_argument(
        "-o", "--output",
        help="Path to the destination folder where processed images will be stored."
    )
    parser.add_argument(
        "-r", "--resolution",
        default="all",
        type=parse_resolution_arg,
        help="Target resolution: 'small' (720p), 'medium' (1080p), 'large' (4K), 'all' (default), or custom 'WxH'."
    )
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Skip center-crop to 16:9 aspect ratio (resizes directly to target resolution)."
    )

    # If executed without arguments (e.g. clicking Run button in IDE), show helpful instructions
    if len(sys.argv) == 1:
        parser.print_help()
        print("\n" + "=" * 60)
        print("[HINT] To run this tool, please provide -i (input) and -o (output).")
        print("=" * 60)
        sys.exit(0)

    args = parser.parse_args()

    if not args.input or not args.output:
        parser.error("Both -i/--input and -o/--output are required when running.")

    input_path = args.input
    output_dir = args.output
    resolutions = args.resolution
    crop_16_9 = not args.no_crop

    print("=" * 60)
    print(f"Input:        {input_path}")
    print(f"Output Dir:   {output_dir}")
    print(f"Target Sizes: {list(resolutions.keys())}")
    print(f"Crop 16:9:    {crop_16_9}")
    print("=" * 60)

    if os.path.isdir(input_path):
        process_directory(input_path, output_dir, resolutions, crop_16_9)
    else:
        process_single_file(input_path, output_dir, resolutions, crop_16_9)

    print("\n[Done] All requested image operations completed successfully.")


if __name__ == "__main__":
    main()
