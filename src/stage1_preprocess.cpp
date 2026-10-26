// ============================================================================
// stage1_preprocess.cpp - Stage 1 (sequential): grayscale + separable Gaussian
//
// Author  : Dang Le Duc Thinh (23120360)
// Sources : see stage1_preprocess.h. No external code.
// ============================================================================
#include "stage1_preprocess.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

int gaussian_radius(float sigma) {
    return sigma > 0.0f ? static_cast<int>(std::ceil(3.0f * sigma)) : 0;
}

std::vector<float> make_gaussian_kernel(float sigma) {
    return make_gaussian_kernel(sigma, gaussian_radius(sigma));
}

std::vector<float> make_gaussian_kernel(float sigma, int radius) {
    if (sigma <= 0.0f) radius = 0;
    radius = std::max(radius, 0);
    std::vector<float> kernel(2 * radius + 1, 1.0f);
    if (radius == 0) return kernel;

    // Weights in double, normalized to sum 1 so the blur preserves brightness.
    std::vector<double> weights(kernel.size());
    double sum = 0.0;
    for (int i = -radius; i <= radius; ++i) {
        weights[i + radius] = std::exp(-static_cast<double>(i) * i / (2.0 * sigma * sigma));
        sum += weights[i + radius];
    }
    for (size_t i = 0; i < kernel.size(); ++i)
        kernel[i] = static_cast<float>(weights[i] / sum);
    return kernel;
}

void convert_to_gray(const Image& src, GrayImage& dst) {
    const size_t num_pixels = static_cast<size_t>(src.width) * src.height;
    const int channels = src.channels;
    dst.width = src.width;
    dst.height = src.height;
    dst.data.resize(num_pixels);

    const uint8_t* in = src.data.data();
    float* out = dst.data.data();
    if (channels >= 3) {
        for (size_t i = 0; i < num_pixels; ++i) {
            const uint8_t* px = in + i * channels;
            out[i] = rgb_to_gray(px[0], px[1], px[2]);
        }
    } else {
        for (size_t i = 0; i < num_pixels; ++i)
            out[i] = static_cast<float>(in[i * channels]);
    }
}

void separable_filter(const float* src, float* tmp, float* dst, int width, int height,
                      const std::vector<float>& kernel) {
    const int radius = static_cast<int>(kernel.size() / 2);
    const float* k = kernel.data();

    // Pass 1: rows (src -> tmp). Every output pixel is independent.
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            tmp[static_cast<size_t>(y) * width + x] = convolve_row_at(src, width, x, y, k, radius);

    // Pass 2: columns (tmp -> dst). The inner loop walks along a row, so each
    // tap reads consecutive addresses.
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            dst[static_cast<size_t>(y) * width + x] = convolve_col_at(tmp, width, height, x, y, k, radius);
}

void gaussian_blur(const GrayImage& src, float sigma, GrayImage& dst) {
    if (gaussian_radius(sigma) == 0) {
        dst = src;
        return;
    }
    const std::vector<float> kernel = make_gaussian_kernel(sigma);
    std::vector<float> tmp(src.data.size());
    dst.width = src.width;
    dst.height = src.height;
    dst.data.resize(src.data.size());
    separable_filter(src.data.data(), tmp.data(), dst.data.data(), src.width, src.height, kernel);
}

void preprocess(const Image& src, const StitchConfig& cfg, GrayImage& dst) {
    convert_to_gray(src, dst);
    gaussian_blur(dst, cfg.blur_sigma, dst);
}
