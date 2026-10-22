// ============================================================================
// v0_sequential.cpp - Version 0: single-threaded CPU pipeline (baseline)
//
// No multithreading and no hand-written SIMD, so V0 is a fair baseline. Each
// stage is timed with std::chrono (CpuTimer); Stages 1-3 include both images.
// ============================================================================
#include "versions.h"
#include "stage1_preprocess.h"
#include "stage2_keypoints.h"
#include "stage3_orb.h"
#include "stage4_matching.h"
#include "stage5_homography.h"
#include "stage6_warp_blend.h"

#include <iostream>
#include <vector>

bool run_v0(
    const Image& imgA,
    const Image& imgB,
    Image& output,
    StageTimings& timings,
    const StitchConfig& config
) {
    // Sanity check input images
    if (!imgA.is_valid() || !imgB.is_valid()) {
        std::cerr << "[V0 Error] Invalid input image(s).\n";
        return false;
    }

    const bool verbose = config.verbose;
    if (verbose) {
        std::cout << "[V0 Sequential] Starting panorama stitching pipeline...\n";
        std::cout << "  Input A: " << imgA.width << "x" << imgA.height << " (" << imgA.channels << " channels)\n";
        std::cout << "  Input B: " << imgB.width << "x" << imgB.height << " (" << imgB.channels << " channels)\n";
    }

    timings = StageTimings();
    CpuTimer timer;

    // ========================================================================
    // Stage 1: Preprocessing (Grayscale Conversion & Gaussian Smoothing)
    // ========================================================================
    // BT.601 grayscale, then separable Gaussian blur (sigma = config.blur_sigma).
    timer.start();
    GrayImage grayA, grayB;
    preprocess(imgA, config, grayA);
    preprocess(imgB, config, grayB);
    timings.preprocess_ms = timer.elapsed_ms();

    // ========================================================================
    // Stage 2: Keypoint Detection (Harris Corner + 3x3 NMS + Grid-based Top-K)
    // ========================================================================
    timer.start();
    std::vector<Keypoint> kpsA, kpsB;
    detect_keypoints(grayA, config, kpsA);
    detect_keypoints(grayB, config, kpsB);
    timings.detect_ms = timer.elapsed_ms();

    // ========================================================================
    // Stage 3: Feature Description (ORB: intensity centroid + steered BRIEF, 256 bits)
    // ========================================================================
    timer.start();
    std::vector<Descriptor> descA, descB;
    compute_descriptors(grayA, kpsA, config, descA);
    compute_descriptors(grayB, kpsB, config, descB);
    timings.describe_ms = timer.elapsed_ms();

    // ========================================================================
    // Stage 4: Feature Matching (Brute-force Hamming + Lowe Ratio Test + Cross-check)
    // ========================================================================
    timer.start();
    std::vector<Match> matches;
    match_descriptors(descA, descB, config, matches);
    std::vector<PointMatch> points;
    matches_to_points(kpsA, kpsB, matches, points);
    timings.match_ms = timer.elapsed_ms();

    // ========================================================================
    // Stage 5: Homography Estimation (RANSAC + 4-Point DLT + Inlier Refinement)
    // ========================================================================
    timer.start();
    Homography H;
    const RansacResult ransac = estimate_homography(points, config, H);
    timings.homography_ms = timer.elapsed_ms();

    if (verbose) {
        std::cout << "  Stage 2 (Harris)   : " << kpsA.size() << " keypoints in A, " << kpsB.size() << " in B\n";
        std::cout << "  Stage 4 (matching) : " << matches.size() << " matches after ratio test"
                  << (config.match_cross_check ? " + cross-check" : "") << "\n";
        std::cout << "  Stage 5 (RANSAC)   : " << ransac.num_inliers << " inliers / " << points.size()
                  << " matches, " << ransac.iterations << " hypotheses\n";
    }
    if (!ransac.success) {
        std::cerr << "[V0 Error] Homography estimation failed (" << points.size()
                  << " matches, at least 4 consistent ones are needed).\n";
        return false;
    }
    if (verbose) {
        std::cout << "  H (B -> A)         : [";
        for (int i = 0; i < 9; ++i) std::cout << H.m[i] << (i == 8 ? "]\n" : (i % 3 == 2 ? "; " : " "));
    }

    // ========================================================================
    // Stage 6: Warping & Blending (Inverse Warping + Bilinear + Feathering)
    // ========================================================================
    timer.start();
    const bool warped = warp_and_blend(imgA, imgB, H, output);
    timings.warp_blend_ms = timer.elapsed_ms();
    if (!warped) {
        std::cerr << "[V0 Error] Warping failed: the homography maps image B to an invalid canvas.\n";
        return false;
    }
    if (verbose) std::cout << "  Stage 6 (warp)     : panorama " << output.width << "x" << output.height << "\n";
    return true;
}
