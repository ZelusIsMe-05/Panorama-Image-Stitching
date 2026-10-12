#ifndef VERSIONS_H
#define VERSIONS_H

#include "common.h"

/**
 * Version 0 (Baseline): Sequential single-threaded C++ execution on CPU.
 * 
 * @param imgA     Left/Reference input RGB image.
 * @param imgB     Right/Target input RGB image to be warped onto imgA's frame.
 * @param output   Output panorama image.
 * @param timings  Structure to record elapsed time per stage.
 * @param config   Algorithm hyper-parameters.
 * @return         True if stitching succeeded, false if matching/homography failed.
 */
bool run_v0(
    const Image& imgA,
    const Image& imgB,
    Image& output,
    StageTimings& timings,
    const StitchConfig& config = StitchConfig()
);

/**
 * Version 1: Basic CUDA parallelization across all 6 stages.
 */

bool run_v1(
    const Image& imgA,
    const Image& imgB,
    Image& output,
    StageTimings& timings,
    const StitchConfig& config = StitchConfig()
);

#endif // VERSIONS_H
