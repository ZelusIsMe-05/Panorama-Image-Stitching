#!/usr/bin/env python3
"""
HEIC -> PNG Conversion Utility for CSC14120 Final Project.

Converts every iPhone .HEIC/.HEIF photo in the dataset folder to .PNG so the images
can be inspected in any viewer and fed to resize_images.py.

Features:
- Scans the input folder recursively and mirrors its sub-folder structure in the output
  (e.g. datasets/standard pairs/Indoor/Pair 1/IMG_1147.HEIC
     -> datasets_png/standard pairs/Indoor/Pair 1/IMG_1147.png)
- Keeps full resolution: no crop, no resize; PNG is lossless
- Applies EXIF orientation and converts Display P3 to sRGB (same loader as resize_images.py)
- Skips images that were already converted unless --overwrite is given
"""

import os
import sys
import argparse
from typing import List

from resize_images import HEIF_EXTENSIONS, load_image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUT = os.path.join(PROJECT_ROOT, "datasets")
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "datasets_png")


def collect_heic_images(input_dir: str) -> List[str]:
    """Returns paths (relative to input_dir) of all HEIC/HEIF images, scanned recursively."""

    found = []
    for root, dirs, files in os.walk(input_dir):
        dirs.sort()
        for fname in sorted(files):
            if os.path.splitext(fname)[1].lower() in HEIF_EXTENSIONS:
                found.append(os.path.relpath(os.path.join(root, fname), input_dir))
    return found


def convert_to_png(input_path: str, output_path: str) -> None:
    """Loads an image upright in sRGB and saves it as PNG at full resolution."""

    with load_image(input_path) as img:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path, "PNG")

        w, h = img.size
        print(f"[OK]   {input_path} -> {output_path} ({w}x{h}, {'portrait' if h > w else 'landscape'})")


def main():
    # Paths may contain non-ASCII characters (e.g. "Năm 4"); Windows falls back to cp1252
    # when output is piped, so force UTF-8 to avoid UnicodeEncodeError
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Convert all HEIC/HEIF photos in a folder (recursively) to PNG, keeping the folder structure."
    )
    parser.add_argument(
        "-i", "--input",
        default=DEFAULT_INPUT,
        help=f"Folder containing HEIC images (default: {DEFAULT_INPUT})."
    )
    parser.add_argument(
        "-o", "--output",
        default=DEFAULT_OUTPUT,
        help=f"Destination folder for PNG images (default: {DEFAULT_OUTPUT})."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-convert images whose PNG already exists."
    )
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)

    print("=" * 60)
    print(f"Input Dir:    {input_dir}")
    print(f"Output Dir:   {output_dir}")
    print(f"Overwrite:    {args.overwrite}")
    print("=" * 60)

    if not os.path.isdir(input_dir):
        print(f"[Error] Directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    if output_dir == input_dir or output_dir.startswith(input_dir + os.sep):
        print("[Error] Output folder must not be inside the input folder.", file=sys.stderr)
        sys.exit(1)

    rel_paths = collect_heic_images(input_dir)
    if not rel_paths:
        print(f"[Warning] No HEIC/HEIF images found in {input_dir}.")
        return

    print(f"Found {len(rel_paths)} HEIC image(s) to convert.\n")

    converted, skipped, failed = 0, 0, []
    for rel in rel_paths:
        src = os.path.join(input_dir, rel)
        dst = os.path.join(output_dir, os.path.splitext(rel)[0] + ".png")

        if os.path.exists(dst) and not args.overwrite:
            print(f"[Skip] {dst} already exists")
            skipped += 1
            continue

        try:
            convert_to_png(src, dst)
            converted += 1
        except Exception as e:
            print(f"[Fail] {src}: {e}", file=sys.stderr)
            failed.append(rel)

    print(f"\n[Done] Converted: {converted}, skipped: {skipped}, failed: {len(failed)}.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
