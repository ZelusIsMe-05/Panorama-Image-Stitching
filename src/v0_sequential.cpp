#include "versions.h"
#include <iostream>
#include <chrono>

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

    std::cout << "[V0 Sequential] Starting panorama stitching pipeline...\n";
    std::cout << "  Input A: " << imgA.width << "x" << imgA.height << " (" << imgA.channels << " channels)\n";
    std::cout << "  Input B: " << imgB.width << "x" << imgB.height << " (" << imgB.channels << " channels)\n";

    // ========================================================================
    // Stage 1: Preprocessing (Grayscale Conversion & Gaussian Smoothing)
    // ========================================================================
    // TODO: Convert RGB input images into single-channel grayscale and apply
    //       separable 2D Gaussian blur filter to suppress high-frequency noise.



    // ========================================================================
    // Stage 2: Keypoint Detection (e.g. FAST / Harris Corner Detection)
    // ========================================================================
    // TODO: Detect distinctive corner points in both images independently.
    //       Apply Non-Maximum Suppression (NMS) and top-K keypoint selection.



    // ========================================================================
    // Stage 3: Feature Description (e.g. BRIEF / ORB Binary Descriptors)
    // ========================================================================
    // TODO: Compute local binary descriptors (256 bits / 32 bytes) for each
    //       detected keypoint based on intensity comparison pairs.



    // ========================================================================
    // Stage 4: Feature Matching & Ratio Test
    // ========================================================================
    // TODO: Perform all-pairs brute-force nearest-neighbor matching using
    //       Hamming distance. Apply Lowe's ratio test to filter out ambiguous matches.



    // ========================================================================
    // Stage 5: Homography Estimation (RANSAC + 4-Point DLT)
    // ========================================================================
    // TODO: Estimate robust 3x3 projective transformation matrix mapping
    //       Image B onto Image A using random sample consensus (RANSAC).



    // ========================================================================
    // Stage 6: Warping & Blending (Inverse Warping + Bilinear Interpolation)
    // ========================================================================
    // TODO: Compute panorama canvas bounding box. Perform backward warping
    //       with bilinear interpolation and blend overlapping regions.



    return true;
}
