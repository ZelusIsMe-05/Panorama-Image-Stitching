// ============================================================================
// stage3_orb.h - Stage 3: ORB descriptor (rotated BRIEF, 256 bits)
//
// Author  : Truong Thanh Thuan (23120366)
// Sources : E. Rublee, V. Rabaud, K. Konolige, G. Bradski, "ORB: An Efficient
//           Alternative to SIFT or SURF", ICCV 2011 (intensity centroid,
//           steered BRIEF), implemented from scratch (no external code).
//
// Data flow (independent per keypoint, one thread per keypoint on the GPU):
//   intensity centroid on a disc of radius r -> theta -> angle bin
//   -> 256 binary tests I(x_i) < I(y_i) with the pattern pre-rotated for that bin
// The pattern (all bins) is a constant table built once: V1+ copies it to
// constant memory, so every thread of a warp reads the same pair at once.
// ============================================================================
#ifndef STAGE3_ORB_H
#define STAGE3_ORB_H

#include "common.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

constexpr int kOrbBits = 256;  // descriptor length (Descriptor = 8 x 32 bits)

// ----------------------------------------------------------------------------
// Per-keypoint helpers
// ----------------------------------------------------------------------------

/** Gray value at (x, y), border pixels repeated. */
HOST_DEVICE inline float gray_at(const float* img, int width, int height, int x, int y) {
    return img[static_cast<size_t>(clamp_index(y, height)) * width + clamp_index(x, width)];
}

/** Orientation theta = atan2(m01, m10), moments taken over the disc of radius r around (cx, cy). */
HOST_DEVICE inline float ic_angle(const float* img, int width, int height, int cx, int cy, int radius) {
    float m10 = 0.0f, m01 = 0.0f;
    for (int dy = -radius; dy <= radius; ++dy)
        for (int dx = -radius; dx <= radius; ++dx) {
            if (dx * dx + dy * dy > radius * radius) continue;
            const float v = gray_at(img, width, height, cx + dx, cy + dy);
            m10 += static_cast<float>(dx) * v;
            m01 += static_cast<float>(dy) * v;
        }
    return atan2f(m01, m10);
}

/** Index of the pattern rotation closest to theta (radians), in [0, num_bins). */
HOST_DEVICE inline int angle_bin(float theta, int num_bins) {
    const float step = 6.28318530718f / static_cast<float>(num_bins);
    int bin = static_cast<int>(floorf(theta / step + 0.5f)) % num_bins;
    return bin < 0 ? bin + num_bins : bin;
}

/**
 * Bit i of desc is I(p1) < I(p2) for test pair i. pattern holds the 256 pairs
 * of one rotation as (x1, y1, x2, y2) offsets from (cx, cy).
 */
HOST_DEVICE inline void orb_descriptor_at(const float* img, int width, int height, int cx, int cy,
                                          const int8_t* pattern, Descriptor& desc) {
    for (int word = 0; word < kOrbBits / 32; ++word) {
        uint32_t bits = 0;
        for (int b = 0; b < 32; ++b) {
            const int8_t* p = pattern + 4 * (word * 32 + b);
            const float i1 = gray_at(img, width, height, cx + p[0], cy + p[1]);
            const float i2 = gray_at(img, width, height, cx + p[2], cy + p[3]);
            bits |= (i1 < i2 ? 1u : 0u) << b;
        }
        desc.bits[word] = bits;
    }
}

// ----------------------------------------------------------------------------
// Sequential version (V0)
// ----------------------------------------------------------------------------

/**
 * Test pairs for every rotation: 256 pairs drawn with a seeded generator
 * (about Gaussian, sigma = S / 5, inside the disc of radius S / 2), then
 * rotated in steps of 360 / num_bins degrees. Pair i of rotation b starts at
 * pattern[(b * 256 + i) * 4].
 */
std::vector<int8_t> make_orb_pattern(const StitchConfig& cfg);

/** Sets kps[i].angle and fills desc[i] for every keypoint; img is the blurred image from Stage 1. */
void compute_descriptors(const GrayImage& img, std::vector<Keypoint>& kps, const StitchConfig& cfg,
                         std::vector<Descriptor>& desc);

#endif // STAGE3_ORB_H
