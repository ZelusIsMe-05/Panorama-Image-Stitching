// ============================================================================
// stage1_preprocess.h - Stage 1: grayscale conversion + separable Gaussian blur
//
// Author  : Dang Le Duc Thinh (23120360)
// Sources : BT.601 luma weights and separable Gaussian filter (standard
//           formulas). No external code.
//
// Each per-pixel helper below computes one output pixel from read-only input,
// so a V1 kernel maps one thread to one pixel and calls the same helper.
// ============================================================================
#ifndef STAGE1_PREPROCESS_H
#define STAGE1_PREPROCESS_H

#include "common.h"

#include <cstddef>
#include <cstdint>
#include <vector>

// ----------------------------------------------------------------------------
// Per-pixel building blocks (shared with the CUDA kernels)
// ----------------------------------------------------------------------------

/** BT.601 luma: Gray = 0.299 R + 0.587 G + 0.114 B, in [0, 255]. */
HOST_DEVICE inline float rgb_to_gray(uint8_t r, uint8_t g, uint8_t b) {
    return 0.299f * r + 0.587f * g + 0.114f * b;
}

/**
 * Horizontal pass at (x, y): sum over i in [-r, r] of kernel[i + r] * src(x + i, y),
 * with replicated borders. kernel has 2r + 1 taps.
 */
HOST_DEVICE inline float convolve_row_at(const float* src, int width, int x, int y,
                                         const float* kernel, int radius) {
    const float* row = src + static_cast<size_t>(y) * width;
    float sum = 0.0f;
    for (int i = -radius; i <= radius; ++i)
        sum += kernel[i + radius] * row[clamp_index(x + i, width)];
    return sum;
}

/** Vertical pass at (x, y): sum over i in [-r, r] of kernel[i + r] * src(x, y + i). */
HOST_DEVICE inline float convolve_col_at(const float* src, int width, int height, int x, int y,
                                         const float* kernel, int radius) {
    float sum = 0.0f;
    for (int i = -radius; i <= radius; ++i)
        sum += kernel[i + radius] * src[static_cast<size_t>(clamp_index(y + i, height)) * width + x];
    return sum;
}

// ----------------------------------------------------------------------------
// Sequential implementations (V0)
// ----------------------------------------------------------------------------

/** Kernel radius r = ceil(3 * sigma), i.e. kernel size k = 2r + 1 (r = 0 if sigma <= 0). */
int gaussian_radius(float sigma);

/** Normalized 1D Gaussian kernel of 2r + 1 taps: g[i] ~ exp(-(i - r)^2 / (2 sigma^2)). */
std::vector<float> make_gaussian_kernel(float sigma, int radius);

/** Same, with the default radius r = gaussian_radius(sigma). */
std::vector<float> make_gaussian_kernel(float sigma);

/**
 * 8-bit image -> float grayscale with BT.601 weights. Uses the first three
 * channels of RGB/RGBA images; 1- or 2-channel images are copied as is.
 */
void convert_to_gray(const Image& src, GrayImage& dst);

/**
 * Separable filter with a 1D kernel of 2r + 1 taps: pass 1 filters the rows
 * (src -> tmp), pass 2 filters the columns (tmp -> dst). Cost O(2k) per pixel
 * instead of O(k^2) for the equivalent 2D kernel. Each buffer holds
 * width * height floats; dst may alias src because pass 2 only reads tmp.
 */
void separable_filter(const float* src, float* tmp, float* dst, int width, int height,
                      const std::vector<float>& kernel);

/** Gaussian blur of a gray image (copy when sigma <= 0); dst may be the same object as src. */
void gaussian_blur(const GrayImage& src, float sigma, GrayImage& dst);

/** Stage 1 entry point: 8-bit image -> grayscale blurred with sigma = cfg.blur_sigma. */
void preprocess(const Image& src, const StitchConfig& cfg, GrayImage& dst);

#endif // STAGE1_PREPROCESS_H
