#!/usr/bin/env python3
"""
Image Preprocessing and Resizing Utility for CSC14120 Final Project.

Author: Dang Le Duc Thinh (23120360)

Features:
- Validates input file formats (supports .jpg, .jpeg, .png, .bmp, .webp, .tiff, .heic, .heif; rejects .pnm, etc.)
- Reads iPhone HEIC/HEIF photos via pi-heif (HEIF-only build of pillow-heif)
- Applies EXIF orientation and converts embedded color profiles (e.g. Display P3) to sRGB
- Performs center-crop to 16:9 aspect ratio without distortion
- Resizes to standard project resolutions (small: 1280x720, medium: 1920x1080, large: 3840x2160)
  or custom target dimensions
- Uses high-quality Lanczos resampling filter for optimal keypoint detection
- Scans folders recursively; output layout is <output>/<resolution>/<same sub-folders as input>
"""

import io
import os
import sys
import argparse
from typing import Tuple, Dict, List
from PIL import Image, ImageCms, ImageOps

try:
    from pi_heif import register_heif_opener
    register_heif_opener()
    HEIF_SUPPORTED = True
except ImportError:
    HEIF_SUPPORTED = False

HEIF_EXTENSIONS = {".heic", ".heif"}

# Supported image extensions (popular web and camera image formats)
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"} | HEIF_EXTENSIONS

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


def load_image(input_path: str) -> Image.Image:
    """
    Loads an image as an upright 3-channel sRGB image:
    - HEIC/HEIF is decoded through pi-heif
    - EXIF orientation is applied, so portrait photos are not stored sideways
    - An embedded color profile (iPhone uses Display P3) is converted to sRGB,
      since stb_image in the C++ pipeline ignores color profiles
    """

    ext = os.path.splitext(input_path)[1].lower()
    if ext in HEIF_EXTENSIONS and not HEIF_SUPPORTED:
        print(
            f"[Error] Cannot read '{input_path}': HEIC/HEIF support requires pi-heif.\n"
            f"        Install it with: pip install -r tools/requirements.txt",
            file=sys.stderr,
        )
        sys.exit(1)

    with Image.open(input_path) as img:
        img = ImageOps.exif_transpose(img)
        icc = img.info.get("icc_profile")

        # Convert to RGB to discard alpha channels and ensure 3-channel output
        if img.mode != "RGB":
            img = img.convert("RGB")

        if icc:
            try:
                src_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc))
                img = ImageCms.profileToProfile(img, src_profile, ImageCms.createProfile("sRGB"), outputMode="RGB")
            except (ImageCms.PyCMSError, OSError):
                print(f"[Warning] Could not apply color profile of '{input_path}', keeping raw RGB values.")

        img.info.pop("icc_profile", None)
        return img


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
    outputs: List[Tuple[Tuple[int, int], str]],
    crop_16_9: bool = True
) -> None:
    """
    Loads an image once, optionally center-crops to 16:9, then for each (target_size, output_path)
    resizes using Lanczos interpolation and saves (format is taken from the output extension).
    """

    with load_image(input_path) as img:
        orig_w, orig_h = img.size

        # Apply 16:9 center cropping if requested
        if crop_16_9:
            img = center_crop_16_9(img)
        crop_w, crop_h = img.size

        # Use Lanczos resampling (high-quality downscaling / antialiasing)
        resampling_filter = getattr(Image.Resampling, "LANCZOS", Image.LANCZOS)

        for (target_w, target_h), output_path in outputs:
            resized_img = img.resize((target_w, target_h), resample=resampling_filter)

            # Create output parent directory if it does not exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

            # Save with high quality (quality only affects JPEG; PNG is lossless)
            resized_img.save(output_path, quality=95)
            print(f"[OK]   {input_path} ({orig_w}x{orig_h}) -> cropped ({crop_w}x{crop_h}) -> {output_path} ({target_w}x{target_h})")


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


def collect_images(input_dir: str) -> List[str]:
    """Returns paths (relative to input_dir) of all valid images, scanned recursively."""

    found = []
    for root, dirs, files in os.walk(input_dir):
        dirs.sort()
        for fname in sorted(files):
            if is_valid_image_file(fname):
                found.append(os.path.relpath(os.path.join(root, fname), input_dir))
    return found


def process_file(
    input_path: str,
    output_paths: Dict[str, str],
    resolutions: Dict[str, Tuple[int, int]],
    crop_16_9: bool,
    overwrite: bool
) -> str:
    """
    Resizes one image to every requested resolution (output_paths maps resolution name -> path).
    Returns 'converted', 'skipped' or 'failed'.
    """

    pending = [
        (resolutions[res_name], out_path)
        for res_name, out_path in output_paths.items()
        if overwrite or not os.path.exists(out_path)
    ]
    if not pending:
        print(f"[Skip] {input_path}: all resolutions already exist")
        return "skipped"

    try:
        resize_image(input_path, pending, crop_16_9=crop_16_9)
        return "converted"
    except Exception as e:
        print(f"[Fail] {input_path}: {e}", file=sys.stderr)
        return "failed"


def main():
    # Paths may contain non-ASCII characters (e.g. "Năm 4"); Windows falls back to cp1252
    # when output is piped, so force UTF-8 to avoid UnicodeEncodeError
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Dataset Preparation Tool: Crop to 16:9 and Resize Images for CSC14120 Final Project."
    )
    parser.add_argument(
        "-i", "--input",
        help="Input image file, or folder scanned recursively."
    )
    parser.add_argument(
        "-o", "--output",
        help="Destination folder; each resolution gets its own sub-folder."
    )
    parser.add_argument(
        "-r", "--resolution",
        default="all",
        type=parse_resolution_arg,
        help="Target resolution: 'small' (720p), 'medium' (1080p), 'large' (4K), 'all' (default), or custom 'WxH'."
    )
    parser.add_argument(
        "-f", "--format",
        default="png",
        choices=["png", "jpg"],
        help="Output format: 'png' (default, lossless) or 'jpg' (quality 95, much smaller files)."
    )
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Skip center-crop to 16:9 aspect ratio (resizes directly to target resolution)."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-create images whose output already exists."
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

    input_path = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)
    resolutions = args.resolution
    crop_16_9 = not args.no_crop

    print("=" * 60)
    print(f"Input:        {input_path}")
    print(f"Output Dir:   {output_dir}")
    print(f"Target Sizes: {list(resolutions.keys())}")
    print(f"Format:       {args.format}")
    print(f"Crop 16:9:    {crop_16_9}")
    print(f"Overwrite:    {args.overwrite}")
    print("=" * 60)

    if os.path.isdir(input_path):
        if output_dir == input_path or output_dir.startswith(input_path + os.sep):
            print("[Error] Output folder must not be inside the input folder.", file=sys.stderr)
            sys.exit(1)

        rel_paths = collect_images(input_path)
        if not rel_paths:
            print(f"[Warning] No valid images ({', '.join(sorted(ALLOWED_EXTENSIONS))}) found in {input_path}.")
            return
        print(f"Found {len(rel_paths)} valid image(s) to process.\n")
    else:
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
        rel_paths = [os.path.basename(input_path)]
        input_path = os.path.dirname(input_path)

    # Layout: <output>/<resolution>/<same sub-folders as input>/<name>.<format>
    counts = {"converted": 0, "skipped": 0, "failed": 0}
    for rel in rel_paths:
        out_rel = os.path.splitext(rel)[0] + "." + args.format
        output_paths = {res_name: os.path.join(output_dir, res_name, out_rel) for res_name in resolutions}
        status = process_file(os.path.join(input_path, rel), output_paths, resolutions, crop_16_9, args.overwrite)
        counts[status] += 1

    print(f"\n[Done] Processed: {counts['converted']}, skipped: {counts['skipped']}, failed: {counts['failed']}.")
    if counts["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
