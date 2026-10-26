// ============================================================================
// image_io.h - Load / save images from disk (stb_image, stb_image_write)
//
// Author  : Dang Le Duc Thinh (23120360)
// Sources : no external code.
//
// Disk I/O is excluded from every timing, as required by the project spec.
// ============================================================================
#ifndef IMAGE_IO_H
#define IMAGE_IO_H

#include "common.h"

#include <string>

/**
 * Loads a PNG / JPG / BMP / ... image as 8-bit RGB (3 channels, alpha dropped,
 * grayscale expanded). Returns false and prints the reason on failure.
 */
bool load_image_rgb(const std::string& path, Image& img);

/**
 * Saves an image; the format follows the extension: .png, .jpg / .jpeg
 * (quality 95) or .bmp. Any other extension is written as PNG.
 */
bool save_image(const std::string& path, const Image& img);

#endif // IMAGE_IO_H
