// ============================================================================
// stage2_keypoints.cpp - Stage 2 (sequential): Harris + 3x3 NMS + Grid Top-K
//
// Author  : Tran Nguyen Tri Thanh (23120356)
// Sources : algorithm of Harris & Stephens 1988 (see stage2_keypoints.h).
//           No external code.
// ============================================================================
#include "stage2_keypoints.h"
#include "stage1_preprocess.h"

#include <algorithm>
#include <cstddef>

void harris_response_map(const GrayImage& img, const StitchConfig& cfg, std::vector<float>& response) {
    const int width = img.width, height = img.height;
    const size_t num_pixels = static_cast<size_t>(width) * height;
    std::vector<float> sxx(num_pixels), syy(num_pixels), sxy(num_pixels);
    // `response` doubles as the scratch buffer of the separable filter until step 3.
    response.resize(num_pixels);

    // Step 1: gradients and their products, independent per pixel.
    const float* src = img.data.data();
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            float gx, gy;
            sobel_at(src, width, height, x, y, gx, gy);
            const size_t i = static_cast<size_t>(y) * width + x;
            sxx[i] = gx * gx;
            syy[i] = gy * gy;
            sxy[i] = gx * gy;
        }

    // Step 2: Gaussian window (sigma_w, harris_window x harris_window), reusing the
    // Stage 1 separable filter in place.
    const std::vector<float> kernel = make_gaussian_kernel(cfg.harris_sigma, cfg.harris_window / 2);
    if (kernel.size() > 1) {
        separable_filter(sxx.data(), response.data(), sxx.data(), width, height, kernel);
        separable_filter(syy.data(), response.data(), syy.data(), width, height, kernel);
        separable_filter(sxy.data(), response.data(), sxy.data(), width, height, kernel);
    }

    // Step 3: response, independent per pixel.
    for (size_t i = 0; i < num_pixels; ++i)
        response[i] = harris_response(sxx[i], syy[i], sxy[i], cfg.harris_k);
}

float max_in_region(const std::vector<float>& map, int width, int height, int border) {
    float best = 0.0f;
    for (int y = border; y < height - border; ++y)
        for (int x = border; x < width - border; ++x)
            best = std::max(best, map[static_cast<size_t>(y) * width + x]);
    return best;
}

void collect_local_maxima(const std::vector<float>& map, int width, int height, int border,
                          float threshold, int nms_radius, std::vector<Keypoint>& out) {
    out.clear();
    const float* m = map.data();
    for (int y = border; y < height - border; ++y)
        for (int x = border; x < width - border; ++x) {
            const float v = m[static_cast<size_t>(y) * width + x];
            if (v > threshold && is_local_max(m, width, height, x, y, nms_radius))
                out.push_back(Keypoint{static_cast<float>(x), static_cast<float>(y), v, 0.0f});
        }
}

void select_grid_top_k(std::vector<Keypoint>& kps, int width, int height, int cell_size, int k) {
    cell_size = std::max(cell_size, 1);
    const int cells_x = (width + cell_size - 1) / cell_size;
    const int cells_y = (height + cell_size - 1) / cell_size;
    const int num_cells = cells_x * cells_y;
    auto cell_of = [&](const Keypoint& kp) {
        return static_cast<int>(kp.y) / cell_size * cells_x + static_cast<int>(kp.x) / cell_size;
    };

    // 1) Histogram of candidates per cell (start[c + 1] = count of cell c).
    std::vector<int> start(num_cells + 1, 0);
    for (const Keypoint& kp : kps) ++start[cell_of(kp) + 1];
    // 2) Exclusive prefix sum: start[c] = first slot of cell c.
    for (int c = 0; c < num_cells; ++c) start[c + 1] += start[c];
    // 3) Scatter the candidates into contiguous per-cell buckets.
    std::vector<Keypoint> bucketed(kps.size());
    std::vector<int> next(start.begin(), start.end() - 1);
    for (const Keypoint& kp : kps) bucketed[next[cell_of(kp)]++] = kp;
    // 4) Keep the k best of every cell (all of them if k <= 0), best first.
    kps.clear();
    for (int c = 0; c < num_cells; ++c) {
        const auto first = bucketed.begin() + start[c];
        const auto last = bucketed.begin() + start[c + 1];
        const auto keep_end = k > 0 ? first + std::min<std::ptrdiff_t>(k, last - first) : last;
        std::partial_sort(first, keep_end, last, keypoint_better);
        kps.insert(kps.end(), first, keep_end);
    }
}

void detect_keypoints(const GrayImage& img, const StitchConfig& cfg, std::vector<Keypoint>& kps) {
    kps.clear();
    if (!img.is_valid()) return;
    const int width = img.width, height = img.height;
    // Keep the complete NMS neighbourhood inside the image. The configured
    // keypoint border may be larger to leave room for the Stage 3 descriptor.
    const int nms_radius = std::max(cfg.harris_nms_radius, 1);
    const int border = std::max(cfg.keypoint_border, nms_radius);

    // Steps 1-3: Harris response map.
    std::vector<float> response;
    harris_response_map(img, cfg, response);

    // Step 4: R > alpha * max(R), configured NMS (3x3 by default), then Grid Top-K.
    const float max_response = max_in_region(response, width, height, border);
    if (max_response <= 0.0f) return;  // flat image: no corner
    const float threshold = std::max(cfg.harris_threshold_ratio, 0.0f) * max_response;
    collect_local_maxima(response, width, height, border, threshold, nms_radius, kps);
    select_grid_top_k(kps, width, height, cfg.grid_cell_size, cfg.grid_top_k);
}
