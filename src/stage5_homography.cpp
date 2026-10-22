// ============================================================================
// stage5_homography.cpp - Stage 5 (sequential): RANSAC + 4-point DLT + refinement
// ============================================================================
#include "stage5_homography.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

// Attempts per hypothesis to draw a non-degenerate sample before giving up on it.
static const int kMaxSampleAttempts = 8;
// Minimum |2 x triangle area| (normalized coordinates) for 3 sample points to count as non-collinear.
static const float kCollinearEps = 1e-3f;

void normalization_transform(const std::vector<float>& x, const std::vector<float>& y, float t[9]) {
    const size_t n = x.size();
    double cx = 0.0, cy = 0.0;
    for (size_t i = 0; i < n; ++i) { cx += x[i]; cy += y[i]; }
    cx /= static_cast<double>(n);
    cy /= static_cast<double>(n);
    double mean_dist = 0.0;
    for (size_t i = 0; i < n; ++i) mean_dist += std::sqrt((x[i] - cx) * (x[i] - cx) + (y[i] - cy) * (y[i] - cy));
    mean_dist /= static_cast<double>(n);
    const double s = mean_dist > 1e-12 ? std::sqrt(2.0) / mean_dist : 1.0;
    const float m[9] = {static_cast<float>(s), 0.0f, static_cast<float>(-s * cx),
                        0.0f, static_cast<float>(s), static_cast<float>(-s * cy),
                        0.0f, 0.0f, 1.0f};
    for (int i = 0; i < 9; ++i) t[i] = m[i];
}

// Inverse of a normalization similarity [s 0 -s*cx; 0 s -s*cy; 0 0 1].
static void invert_normalization(const float t[9], float inv[9]) {
    const float s = t[0];
    const float m[9] = {1.0f / s, 0.0f, -t[2] / s, 0.0f, 1.0f / s, -t[5] / s, 0.0f, 0.0f, 1.0f};
    for (int i = 0; i < 9; ++i) inv[i] = m[i];
}

// h = Ta^-1 * hn * Tb, scaled so that h[8] = 1 (hn maps normalized B to normalized A).
static void denormalize(const float hn[9], const float ta[9], const float tb[9], float h[9]) {
    float ta_inv[9], tmp[9];
    invert_normalization(ta, ta_inv);
    mat3_mul(hn, tb, tmp);
    mat3_mul(ta_inv, tmp, h);
    if (std::fabs(h[8]) > 1e-12f) {
        const float inv = 1.0f / h[8];
        for (int i = 0; i < 9; ++i) h[i] *= inv;
    }
}

static void split_points(const std::vector<PointMatch>& matches, std::vector<float>& ax, std::vector<float>& ay,
                         std::vector<float>& bx, std::vector<float>& by) {
    const size_t n = matches.size();
    ax.resize(n); ay.resize(n); bx.resize(n); by.resize(n);
    for (size_t i = 0; i < n; ++i) {
        ax[i] = matches[i].ax; ay[i] = matches[i].ay;
        bx[i] = matches[i].bx; by[i] = matches[i].by;
    }
}

int count_inliers(const float* h, const std::vector<PointMatch>& matches, float tau) {
    const float tau_sq = tau * tau;
    int count = 0;
    for (const PointMatch& m : matches) count += is_inlier(h, m, tau_sq) ? 1 : 0;
    return count;
}

bool fit_homography_least_squares(const std::vector<PointMatch>& matches, Homography& h) {
    if (matches.size() < 4) return false;
    std::vector<float> ax, ay, bx, by;
    split_points(matches, ax, ay, bx, by);
    float ta[9], tb[9];
    normalization_transform(ax, ay, ta);
    normalization_transform(bx, by, tb);

    // Normal equations (A^T A) h = A^T b, accumulated over the 2 DLT rows of every match.
    double ata[8][9] = {};
    for (size_t i = 0; i < matches.size(); ++i) {
        const double x = tb[0] * bx[i] + tb[2], y = tb[4] * by[i] + tb[5];
        const double u = ta[0] * ax[i] + ta[2], v = ta[4] * ay[i] + ta[5];
        double rows[2][9];
        dlt_rows(x, y, u, v, rows[0], rows[1]);
        for (const auto& r : rows)
            for (int p = 0; p < 8; ++p)
                for (int q = 0; q < 9; ++q) ata[p][q] += r[p] * r[q];  // column 8 holds A^T b
    }
    double sol[8];
    if (!solve_linear_system(ata, sol)) return false;
    float hn[9];
    for (int i = 0; i < 8; ++i) hn[i] = static_cast<float>(sol[i]);
    hn[8] = 1.0f;
    denormalize(hn, ta, tb, h.m);
    return true;
}

RansacResult estimate_homography(const std::vector<PointMatch>& matches, const StitchConfig& cfg,
                                 Homography& h, std::vector<uint8_t>* inlier_mask) {
    RansacResult result;
    const Homography identity = {{1, 0, 0, 0, 1, 0, 0, 0, 1}};
    h = identity;
    if (inlier_mask) inlier_mask->assign(matches.size(), 0);
    const int n = static_cast<int>(matches.size());
    if (n < 4) return result;

    // Normalize both point sets once; every hypothesis is solved in normalized coordinates.
    std::vector<float> ax, ay, bx, by;
    split_points(matches, ax, ay, bx, by);
    float ta[9], tb[9];
    normalization_transform(ax, ay, ta);
    normalization_transform(bx, by, tb);
    for (int i = 0; i < n; ++i) {
        ax[i] = ta[0] * ax[i] + ta[2]; ay[i] = ta[4] * ay[i] + ta[5];
        bx[i] = tb[0] * bx[i] + tb[2]; by[i] = tb[4] * by[i] + tb[5];
    }

    const float tau = cfg.ransac_threshold;
    const int stop_count = cfg.ransac_stop_ratio > 0.0f
                               ? static_cast<int>(std::ceil(cfg.ransac_stop_ratio * static_cast<float>(n)))
                               : n + 1;
    int best_count = -1;
    float best_h[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

    // Hypotheses are independent (one per GPU thread/warp/block). The early stop at
    // the first hypothesis reaching stop_count keeps the result reproducible in
    // parallel: best = argmax over hypotheses [0, first index reaching stop_count].
    for (int it = 0; it < std::max(cfg.ransac_iterations, 1); ++it) {
        result.iterations = it + 1;
        float hn[9];
        bool valid = false;
        for (int attempt = 0; attempt < kMaxSampleAttempts && !valid; ++attempt) {
            int idx[4];
            ransac_sample(cfg.ransac_seed, it, attempt, n, idx);
            float px[4], py[4], qx[4], qy[4];
            for (int k = 0; k < 4; ++k) {
                px[k] = bx[idx[k]]; py[k] = by[idx[k]];
                qx[k] = ax[idx[k]]; qy[k] = ay[idx[k]];
            }
            valid = non_degenerate4(px, py, kCollinearEps) && non_degenerate4(qx, qy, kCollinearEps) &&
                    homography_from_4(px, py, qx, qy, hn);
        }
        if (!valid) continue;

        float hyp[9];
        denormalize(hn, ta, tb, hyp);
        const int count = count_inliers(hyp, matches, tau);
        if (count > best_count) {  // strict: ties keep the lowest hypothesis index
            best_count = count;
            for (int i = 0; i < 9; ++i) best_h[i] = hyp[i];
        }
        if (count >= stop_count) break;
    }
    if (best_count < 4) return result;

    // Refinement: least squares on all inliers of the best hypothesis; kept only if
    // it does not lose inliers.
    const float tau_sq = tau * tau;
    std::vector<PointMatch> inliers;
    for (const PointMatch& m : matches)
        if (is_inlier(best_h, m, tau_sq)) inliers.push_back(m);
    Homography refined;
    int final_count = best_count;
    for (int i = 0; i < 9; ++i) h.m[i] = best_h[i];
    if (fit_homography_least_squares(inliers, refined)) {
        const int refined_count = count_inliers(refined.m, matches, tau);
        if (refined_count >= best_count) {
            h = refined;
            final_count = refined_count;
        }
    }

    if (inlier_mask)
        for (int i = 0; i < n; ++i) (*inlier_mask)[i] = is_inlier(h.m, matches[i], tau_sq) ? 1 : 0;
    result.success = true;
    result.num_inliers = final_count;
    return result;
}
