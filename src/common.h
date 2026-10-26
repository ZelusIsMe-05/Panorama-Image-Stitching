// ============================================================================
// common.h - Shared data types used by every version of the pipeline (V0, V1...)
//
// Author  : Dang Le Duc Thinh (23120360)
// Sources : interfaces from the CSC14120 project starter notebook;
//           pcg_hash from M. E. O'Neill, PCG, 2014 (see below).
// ============================================================================
#ifndef COMMON_H
#define COMMON_H

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// ----------------------------------------------------------------------------
// HOST_DEVICE marks the small per-pixel / per-keypoint helpers that are shared
// by the sequential loops (V0) and the CUDA kernels (V1+). Under nvcc they are
// compiled for both host and device, so every version runs the same arithmetic.
// Such helpers must not call std:: functions (not available on the device).
// ----------------------------------------------------------------------------
#if defined(__CUDACC__)
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

/** Replicate-border addressing: clamps i into [0, n - 1]. */
HOST_DEVICE inline int clamp_index(int i, int n) {
    return i < 0 ? 0 : (i >= n ? n - 1 : i);
}

/**
 * PCG integer hash (M. O'Neill, "PCG: a family of simple fast statistically good
 * algorithms for random number generation", 2014): stateless, counter-based
 * random numbers, so CPU and GPU draw the same values from the same counter.
 */
HOST_DEVICE inline uint32_t pcg_hash(uint32_t v) {
    const uint32_t state = v * 747796405u + 2891336453u;
    const uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// ============================================================================
// Images
// ============================================================================

/**
 * 8-bit image as loaded by stb_image: row-major with interleaved channels, so
 * channel c of pixel (x, y) is data[(y * width + x) * channels + c].
 */
struct Image {
    int width = 0;
    int height = 0;
    int channels = 0;
    std::vector<uint8_t> data;

    Image() = default;
    Image(int w, int h, int c)
        : width(w), height(h), channels(c), data(static_cast<size_t>(w) * h * c) {}

    bool is_valid() const {
        return width > 0 && height > 0 && channels > 0 &&
               data.size() == static_cast<size_t>(width) * height * channels;
    }
};

/**
 * Single-channel float image, row-major, intensities in [0, 255].
 * Output of Stage 1 (blurred grayscale) and input of Stages 2 and 3.
 */
struct GrayImage {
    int width = 0;
    int height = 0;
    std::vector<float> data;

    GrayImage() = default;
    GrayImage(int w, int h) : width(w), height(h), data(static_cast<size_t>(w) * h) {}

    bool is_valid() const {
        return width > 0 && height > 0 && data.size() == static_cast<size_t>(width) * height;
    }
};

// ============================================================================
// Features
// ============================================================================

/**
 * Keypoint produced by Stage 2. Kept as a plain struct (no constructors) so it
 * can be copied to the GPU and used in __shared__ arrays.
 */
struct Keypoint {
    float x;      // column in pixels
    float y;      // row in pixels
    float score;  // Harris response R, used for ranking
    float angle;  // orientation in radians, set by Stage 3 (0 until then)
};

/** 256-bit binary descriptor (ORB) produced by Stage 3: bit i is word i / 32, bit i % 32. */
struct Descriptor {
    uint32_t bits[8];
};

/** Match produced by Stage 4: keypoint idx_a of image A <-> keypoint idx_b of image B. */
struct Match {
    int idx_a;
    int idx_b;
    int distance;  // Hamming distance between the two descriptors (0..256)
};

/**
 * One point correspondence, input of Stage 5: (ax, ay) in image A and the
 * matching (bx, by) in image B. Built from the Stage 4 matches.
 */
struct PointMatch {
    float ax, ay;
    float bx, by;
};

/** 3x3 homography, row-major, mapping image B coordinates onto image A (m[8] is 1 after normalization). */
struct Homography {
    float m[9];
};

// ============================================================================
// Configuration
// ============================================================================

/**
 * Algorithm hyper-parameters. All versions must run with the same values so
 * that their outputs and timings are comparable.
 */
struct StitchConfig {
    // ---- Stage 1: Preprocessing ----
    float blur_sigma = 1.0f;               // sigma of the Gaussian blur, kernel size 2*ceil(3*sigma)+1

    // ---- Stage 2: Harris corner + NMS + Grid-based Top-K ----
    float harris_k = 0.04f;                // k in R = det(M) - k * trace(M)^2
    float harris_sigma = 1.0f;             // sigma_w of the Gaussian smoothing the structure tensor
    int harris_window = 5;                 // tensor window size (3 or 5 -> 3x3 or 5x5)
    float harris_threshold_ratio = 0.01f;  // alpha: keep R > alpha * max(R)
    int harris_nms_radius = 1;             // NMS window is (2r+1) x (2r+1), i.e. 3x3
    int grid_cell_size = 32;               // side of a square grid cell in pixels
    int grid_top_k = 5;                    // K/cell: best keypoints kept per cell (must be > 0 for benchmarks)
    int keypoint_border = 16;              // drop keypoints closer than this to the image edge
                                           // (room for the Stage 3 descriptor patch)

    // ---- Stage 3: ORB descriptor (rotated BRIEF, 256 bits) ----
    int orb_patch_size = 31;               // S: side of the patch holding the test pattern (odd)
    int orb_centroid_radius = 15;          // r: radius of the disc used for the intensity centroid
    int orb_angle_bins = 30;               // orientation quantized in 360 / 30 = 12 degree steps
    uint32_t orb_pattern_seed = 0x0B5u;    // seed of the fixed random test pattern

    // ---- Stage 4: brute-force Hamming + Lowe ratio test + cross-check ----
    float match_ratio = 0.75f;             // rho: accept if d1 < rho * d2
    bool match_cross_check = true;         // keep only mutual nearest neighbours

    // ---- Stage 5: RANSAC + 4-point DLT + inlier refinement ----
    int ransac_iterations = 2000;          // N_iter: number of hypotheses
    float ransac_threshold = 3.0f;         // tau: max reprojection error of an inlier (pixels)
    float ransac_stop_ratio = 0.8f;        // early stop once a hypothesis reaches this inlier ratio (<= 0: off)
    uint32_t ransac_seed = 2026u;          // seed of the per-hypothesis random generator

    // ---- Stage 6: Cylindrical inverse warping + bilinear + feathering + crop ----
    // Parameter-free (the focal length is estimated from H).
    bool crop_to_covered = true;           // crop to the largest rectangle with no black pixel (--no-crop: off)

    // ---- Logging (not an algorithm parameter) ----
    bool verbose = true;                   // print per-stage statistics (keypoints, matches, H)
};

/** Validates the shared Stage 2/3 configuration before any pipeline version runs. */
inline bool validate_stitch_config(const StitchConfig& cfg, std::string& error) {
    auto fail = [&](const char* message) {
        error = message;
        return false;
    };

    if (!std::isfinite(cfg.harris_k) || cfg.harris_k <= 0.0f || cfg.harris_k >= 0.25f)
        return fail("harris_k must be finite and in (0, 0.25)");
    if (!std::isfinite(cfg.harris_sigma) || cfg.harris_sigma < 0.0f)
        return fail("harris_sigma must be finite and non-negative");
    if (cfg.harris_window <= 0 || cfg.harris_window % 2 == 0)
        return fail("harris_window must be a positive odd number");
    if (!std::isfinite(cfg.harris_threshold_ratio) || cfg.harris_threshold_ratio <= 0.0f ||
        cfg.harris_threshold_ratio >= 1.0f)
        return fail("harris_threshold_ratio must be finite and in (0, 1)");
    if (cfg.harris_nms_radius < 1)
        return fail("harris_nms_radius must be at least 1");
    if (cfg.grid_cell_size <= 0)
        return fail("grid_cell_size must be positive");
    if (cfg.grid_top_k <= 0)
        return fail("grid_top_k must be positive for the benchmark configuration");
    if (cfg.orb_patch_size <= 0 || cfg.orb_patch_size % 2 == 0)
        return fail("orb_patch_size must be a positive odd number");
    if (cfg.orb_centroid_radius < 0)
        return fail("orb_centroid_radius must be non-negative");

    const int descriptor_radius = std::max(cfg.orb_patch_size / 2, cfg.orb_centroid_radius);
    const int required_border = std::max(cfg.harris_nms_radius, descriptor_radius);
    if (cfg.keypoint_border < required_border)
        return fail("keypoint_border must cover both the NMS and Stage 3 descriptor radii");

    error.clear();
    return true;
}

// ============================================================================
// Timing
// ============================================================================

/** Time of each stage in milliseconds; Stages 1-3 include both images. */
struct StageTimings {
    double preprocess_ms = 0.0;
    double detect_ms = 0.0;
    double describe_ms = 0.0;
    double match_ms = 0.0;
    double homography_ms = 0.0;
    double warp_blend_ms = 0.0;

    double total_ms() const {
        return preprocess_ms + detect_ms + describe_ms + match_ms + homography_ms + warp_blend_ms;
    }
};

/** Wall-clock timer for the CPU version (std::chrono::high_resolution_clock). */
struct CpuTimer {
    std::chrono::high_resolution_clock::time_point start_time;

    void start() { start_time = std::chrono::high_resolution_clock::now(); }

    double elapsed_ms() const {
        return std::chrono::duration<double, std::milli>(
                   std::chrono::high_resolution_clock::now() - start_time).count();
    }
};

#endif // COMMON_H
