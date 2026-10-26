// ============================================================================
// stage5_homography.h - Stage 5: RANSAC + 4-point DLT + inlier refinement
//
// Author  : Nguyen Dang Pha (23120315)
// Sources : M. A. Fischler, R. C. Bolles, "Random Sample Consensus", CACM
//           1981; R. I. Hartley, "In Defense of the Eight-Point Algorithm",
//           TPAMI 1997 (point normalization). Implemented from scratch
//           (no external code).
//
// Data flow:
//   normalize points -> for each hypothesis i (independent):
//       sample 4 matches with RNG(seed, i) -> 4-point DLT -> count inliers
//   -> best = most inliers (ties: lowest i) -> least-squares DLT on its inliers
// Hypothesis i always draws the same sample, so a GPU version that evaluates
// hypotheses in parallel (one thread / warp / block per hypothesis) and then
// reduces with the same rule returns exactly the V0 result.
// ============================================================================
#ifndef STAGE5_HOMOGRAPHY_H
#define STAGE5_HOMOGRAPHY_H

#include "common.h"

#include <cstdint>
#include <vector>

// ----------------------------------------------------------------------------
// Per-hypothesis / per-match building blocks (shared with the CUDA kernels)
// ----------------------------------------------------------------------------

/**
 * Draws 4 distinct indices in [0, n) for (hypothesis, attempt), n >= 4.
 * Choice k picks among the n - k unused indices, so no rejection loop is needed.
 */
HOST_DEVICE inline void ransac_sample(uint32_t seed, int hypothesis, int attempt, int n, int idx[4]) {
    uint32_t state = pcg_hash(seed ^ pcg_hash(static_cast<uint32_t>(hypothesis) * 16u +
                                              static_cast<uint32_t>(attempt)));
    int sorted[4];
    for (int k = 0; k < 4; ++k) {
        state = pcg_hash(state);
        int c = static_cast<int>(state % static_cast<uint32_t>(n - k));
        // Skip over the indices already taken (kept sorted ascending).
        int pos = 0;
        while (pos < k && c >= sorted[pos]) {
            ++c;
            ++pos;
        }
        for (int j = k; j > pos; --j) sorted[j] = sorted[j - 1];
        sorted[pos] = c;
        idx[k] = c;
    }
}

/** Twice the signed area of triangle (p, q, r); near 0 means collinear. */
HOST_DEVICE inline float cross3(float px, float py, float qx, float qy, float rx, float ry) {
    return (qx - px) * (ry - py) - (qy - py) * (rx - px);
}

/** True if no 3 of the 4 points (x[i], y[i]) are (nearly) collinear. */
HOST_DEVICE inline bool non_degenerate4(const float* x, const float* y, float eps) {
    const int tri[4][3] = {{0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3}};
    for (int t = 0; t < 4; ++t) {
        const float a = cross3(x[tri[t][0]], y[tri[t][0]], x[tri[t][1]], y[tri[t][1]],
                               x[tri[t][2]], y[tri[t][2]]);
        if ((a < 0 ? -a : a) < eps) return false;
    }
    return true;
}

/**
 * Solves the N x N system stored as an augmented matrix a[N][N+1] by Gaussian
 * elimination with partial pivoting. Returns false if a pivot is ~0 (singular).
 */
template <typename T, int N>
HOST_DEVICE inline bool solve_linear_system(T (&a)[N][N + 1], T (&x)[N]) {
    for (int col = 0; col < N; ++col) {
        int pivot = col;
        T best = a[col][col] < 0 ? -a[col][col] : a[col][col];
        for (int r = col + 1; r < N; ++r) {
            const T v = a[r][col] < 0 ? -a[r][col] : a[r][col];
            if (v > best) { best = v; pivot = r; }
        }
        if (best < static_cast<T>(1e-12)) return false;
        if (pivot != col)
            for (int c = col; c <= N; ++c) { const T t = a[col][c]; a[col][c] = a[pivot][c]; a[pivot][c] = t; }
        for (int r = col + 1; r < N; ++r) {
            const T f = a[r][col] / a[col][col];
            for (int c = col; c <= N; ++c) a[r][c] -= f * a[col][c];
        }
    }
    for (int r = N - 1; r >= 0; --r) {
        T s = a[r][N];
        for (int c = r + 1; c < N; ++c) s -= a[r][c] * x[c];
        x[r] = s / a[r][r];
    }
    return true;
}

/**
 * The two DLT rows of one correspondence (x, y) -> (u, v) with h33 = 1:
 *   [x y 1 0 0 0 -ux -uy] h = u,   [0 0 0 x y 1 -vx -vy] h = v
 */
template <typename T>
HOST_DEVICE inline void dlt_rows(T x, T y, T u, T v, T r0[9], T r1[9]) {
    r0[0] = x; r0[1] = y; r0[2] = 1; r0[3] = 0; r0[4] = 0; r0[5] = 0; r0[6] = -u * x; r0[7] = -u * y; r0[8] = u;
    r1[0] = 0; r1[1] = 0; r1[2] = 0; r1[3] = x; r1[4] = y; r1[5] = 1; r1[6] = -v * x; r1[7] = -v * y; r1[8] = v;
}

/** Exact homography through 4 correspondences (x[i], y[i]) -> (u[i], v[i]); false if degenerate. */
HOST_DEVICE inline bool homography_from_4(const float* x, const float* y, const float* u, const float* v,
                                          float h[9]) {
    float a[8][9];
    for (int i = 0; i < 4; ++i) dlt_rows(x[i], y[i], u[i], v[i], a[2 * i], a[2 * i + 1]);
    float sol[8];
    if (!solve_linear_system(a, sol)) return false;
    for (int i = 0; i < 8; ++i) h[i] = sol[i];
    h[8] = 1.0f;
    return true;
}

/** c = a * b for row-major 3x3 matrices. */
HOST_DEVICE inline void mat3_mul(const float* a, const float* b, float* c) {
    for (int r = 0; r < 3; ++r)
        for (int col = 0; col < 3; ++col)
            c[r * 3 + col] = a[r * 3] * b[col] + a[r * 3 + 1] * b[3 + col] + a[r * 3 + 2] * b[6 + col];
}

/** Applies h to (x, y). Returns false if the point maps to infinity or behind the camera (w <= 0). */
HOST_DEVICE inline bool project_point(const float* h, float x, float y, float& u, float& v) {
    const float w = h[6] * x + h[7] * y + h[8];
    if (!(w > 1e-8f)) return false;
    u = (h[0] * x + h[1] * y + h[2]) / w;
    v = (h[3] * x + h[4] * y + h[5]) / w;
    return true;
}

/** Inlier test of one match: reprojection error |H(b) - a| < tau (compared squared). */
HOST_DEVICE inline bool is_inlier(const float* h, const PointMatch& m, float tau_sq) {
    float u, v;
    if (!project_point(h, m.bx, m.by, u, v)) return false;
    const float du = u - m.ax, dv = v - m.ay;
    return du * du + dv * dv < tau_sq;
}

// ----------------------------------------------------------------------------
// Sequential implementations (V0)
// ----------------------------------------------------------------------------

/** Outcome of Stage 5. */
struct RansacResult {
    bool success = false;    // false: fewer than 4 matches or no valid hypothesis
    int num_inliers = 0;     // inliers of the final (refined) homography
    int iterations = 0;      // hypotheses evaluated (< ransac_iterations after an early stop)
};

/**
 * Hartley normalization of a point set: similarity T moving the centroid to
 * the origin with mean distance sqrt(2). t is row-major 3x3.
 */
void normalization_transform(const std::vector<float>& x, const std::vector<float>& y, float t[9]);

/** Number of matches with reprojection error < tau under h. V1: a reduction over matches. */
int count_inliers(const float* h, const std::vector<PointMatch>& matches, float tau);

/**
 * Least-squares DLT (h33 = 1) on the given matches: accumulates the 8 x 8 normal
 * equations (a reduction) in normalized coordinates and solves them in double.
 */
bool fit_homography_least_squares(const std::vector<PointMatch>& matches, Homography& h);

/**
 * Stage 5 entry point: robust homography mapping image B onto image A.
 * inlier_mask (optional) receives 1 for every inlier of the final homography.
 */
RansacResult estimate_homography(const std::vector<PointMatch>& matches, const StitchConfig& cfg,
                                 Homography& h, std::vector<uint8_t>* inlier_mask = nullptr);

#endif // STAGE5_HOMOGRAPHY_H
