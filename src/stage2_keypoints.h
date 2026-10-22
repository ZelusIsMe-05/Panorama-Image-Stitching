// ============================================================================
// stage2_keypoints.h - Stage 2: Harris corner + 3x3 NMS + Grid-based Top-K
//
// Data flow (each step is data-parallel, see the notes on each function):
//   Sobel 3x3 -> Ix^2, Iy^2, IxIy -> Gaussian window -> R map -> max(R)
//   -> R > alpha * max(R) and 3x3 NMS -> candidates -> Top-K per grid cell
// Keypoints are ranked by (score desc, y asc, x asc), so the output order is
// deterministic and a GPU version can reproduce it exactly by sorting.
// ============================================================================
#ifndef STAGE2_KEYPOINTS_H
#define STAGE2_KEYPOINTS_H

#include "common.h"

#include <vector>

// ----------------------------------------------------------------------------
// Per-pixel / per-keypoint building blocks (shared with the CUDA kernels)
// ----------------------------------------------------------------------------

/** 3x3 Sobel gradients at (x, y) with replicated borders, scaled by 1/8 (intensity per pixel). */
HOST_DEVICE inline void sobel_at(const float* img, int width, int height, int x, int y,
                                 float& gx, float& gy) {
    const int xl = clamp_index(x - 1, width), xr = clamp_index(x + 1, width);
    const float* up = img + static_cast<size_t>(clamp_index(y - 1, height)) * width;
    const float* mid = img + static_cast<size_t>(y) * width;
    const float* down = img + static_cast<size_t>(clamp_index(y + 1, height)) * width;
    gx = 0.125f * ((up[xr] + 2.0f * mid[xr] + down[xr]) - (up[xl] + 2.0f * mid[xl] + down[xl]));
    gy = 0.125f * ((down[xl] + 2.0f * down[x] + down[xr]) - (up[xl] + 2.0f * up[x] + up[xr]));
}

/** Harris response R = det(M) - k * trace(M)^2 of the tensor M = [[sxx, sxy], [sxy, syy]]. */
HOST_DEVICE inline float harris_response(float sxx, float syy, float sxy, float k) {
    const float det = sxx * syy - sxy * sxy;
    const float trace = sxx + syy;
    return det - k * trace * trace;
}

/**
 * Non-maximum suppression test: true if map(x, y) is the maximum of its
 * (2r+1) x (2r+1) window (r = 1 for the 3x3 NMS). Equal values are resolved by
 * raster order (the earlier pixel wins), so the result does not depend on the
 * visiting order.
 */
HOST_DEVICE inline bool is_local_max(const float* map, int width, int height, int x, int y,
                                     int radius) {
    const float v = map[static_cast<size_t>(y) * width + x];
    const int y0 = y - radius < 0 ? 0 : y - radius;
    const int y1 = y + radius >= height ? height - 1 : y + radius;
    const int x0 = x - radius < 0 ? 0 : x - radius;
    const int x1 = x + radius >= width ? width - 1 : x + radius;
    for (int yy = y0; yy <= y1; ++yy) {
        const float* row = map + static_cast<size_t>(yy) * width;
        for (int xx = x0; xx <= x1; ++xx) {
            const float u = row[xx];
            if (u > v || (u == v && (yy < y || (yy == y && xx < x)))) return false;
        }
    }
    return true;
}

/** Ranking used by the Top-K: higher score first, ties by row then column. */
HOST_DEVICE inline bool keypoint_better(const Keypoint& a, const Keypoint& b) {
    if (a.score != b.score) return a.score > b.score;
    if (a.y != b.y) return a.y < b.y;
    return a.x < b.x;
}

// ----------------------------------------------------------------------------
// Sequential implementations (V0)
// ----------------------------------------------------------------------------

/**
 * Harris response of every pixel: Sobel gradients, products Ix^2, Iy^2, IxIy,
 * Gaussian window (sigma_w, harris_window taps, separable filter of Stage 1),
 * then R. V1: one thread per pixel for each step.
 */
void harris_response_map(const GrayImage& img, const StitchConfig& cfg, std::vector<float>& response);

/** Largest value of the map inside [border, size - border) (a max-reduction on the GPU). */
float max_in_region(const std::vector<float>& map, int width, int height, int border);

/**
 * Candidates: pixels in [border, size - border) whose map value is > threshold
 * and that pass NMS with the given radius. Output in raster order.
 * V1: one thread per pixel + compaction (atomic counter or scan).
 */
void collect_local_maxima(const std::vector<float>& map, int width, int height, int border,
                          float threshold, int nms_radius, std::vector<Keypoint>& out);

/**
 * Grid Top-K: keeps the k best keypoints of every cell_size x cell_size cell
 * (k <= 0 keeps all). Written as histogram -> prefix sum -> scatter -> per-cell
 * selection, which map to GPU primitives (atomic histogram, scan, one block per
 * cell). Output is ordered by cell (row-major), best first inside each cell,
 * whatever the input order.
 */
void select_grid_top_k(std::vector<Keypoint>& kps, int width, int height, int cell_size, int k);

/** Stage 2 entry point: Harris + 3x3 NMS + Grid Top-K on the Stage 1 output. */
void detect_keypoints(const GrayImage& img, const StitchConfig& cfg, std::vector<Keypoint>& kps);

#endif // STAGE2_KEYPOINTS_H
