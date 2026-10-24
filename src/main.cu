// ============================================================================
// main.cu - Command-line driver: load images, run one pipeline version N times,
//           report per-stage timings, save the panorama.
//
// Usage (project spec, Sec. 4.1):
//   ./stitch <left_image> <right_image> <output_image> --version <0|1|2|...> [--runs N] [--no-crop]
// Timing rules (Sec. 4.2): one warm-up run is not timed, then N timed runs;
// average and minimum are reported; disk I/O is excluded.
// ============================================================================
#include "common.h"
#include "versions.h"
#include "image_io.h"
#if defined(__CUDACC__)
#include "cuda_utils.cuh"
#endif

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

void print_usage(const char* prog_name) {
    std::cout << "Usage:\n"
              << "  " << prog_name << " <left_image> <right_image> <output_image> --version <0|1|2|...> [--runs N] [--no-crop]\n\n"
              << "Arguments:\n"
              << "  <left_image>     Path to reference image (Image A)\n"
              << "  <right_image>    Path to secondary image (Image B to be warped)\n"
              << "  <output_image>   Path for saving final panorama image (.png, .jpg or .bmp)\n"
              << "  --version <V>    Pipeline version (0: Sequential CPU, 1: Basic CUDA, ...)\n"
              << "  --runs <N>       Number of benchmark runs to average (default: 5)\n"
              << "  --no-crop        Keep the whole canvas (black where no image covers it)\n";
}

using PipelineFn = bool (*)(const Image&, const Image&, Image&, StageTimings&, const StitchConfig&);

// Maps --version to its pipeline function (nullptr: not implemented yet).
static PipelineFn select_version(int version) {
    switch (version) {
        case 0: return run_v0;
        // case 1: return run_v1;   // TODO: enable when v1 is implemented
        default: return nullptr;
    }
}

static void print_timing_table(const std::vector<StageTimings>& runs) {
    struct Row { const char* name; double StageTimings::*field; };
    const Row rows[] = {
        {"1. Preprocessing", &StageTimings::preprocess_ms},
        {"2. Keypoint detection", &StageTimings::detect_ms},
        {"3. Description", &StageTimings::describe_ms},
        {"4. Matching", &StageTimings::match_ms},
        {"5. Homography (RANSAC)", &StageTimings::homography_ms},
        {"6. Warping & blending", &StageTimings::warp_blend_ms},
    };
    const double n = static_cast<double>(runs.size());
    auto line = [](const char* name, double avg, double mn) {
        std::cout << std::left << std::setw(26) << name << std::right << std::fixed << std::setprecision(3)
                  << std::setw(13) << avg << std::setw(13) << mn << "\n";
    };
    std::cout << "\n" << std::left << std::setw(26) << "Stage" << std::right << std::setw(13) << "avg (ms)"
              << std::setw(13) << "min (ms)" << "\n" << std::string(52, '-') << "\n";
    for (const Row& row : rows) {
        double sum = 0.0, mn = 1e300;
        for (const StageTimings& t : runs) { sum += t.*row.field; mn = std::min(mn, t.*row.field); }
        line(row.name, sum / n, mn);
    }
    double sum = 0.0, mn = 1e300;
    for (const StageTimings& t : runs) { sum += t.total_ms(); mn = std::min(mn, t.total_ms()); }
    std::cout << std::string(52, '-') << "\n";
    line("Total (end-to-end)", sum / n, mn);
    std::cout.unsetf(std::ios::floatfield);
}

int main(int argc, char* argv[]) {
    if (argc < 6) {
        print_usage(argv[0]);
        return 1;
    }

    std::string left_path = argv[1];
    std::string right_path = argv[2];
    std::string output_path = argv[3];
    int version = -1;
    int runs = 5;
    bool crop = true;

    // Parse options: --version <V>, optional --runs <N> and --no-crop
    for (int i = 4; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--version" && i + 1 < argc) {
            version = std::atoi(argv[++i]);
        } else if (arg == "--runs" && i + 1 < argc) {
            runs = std::max(1, std::atoi(argv[++i]));
        } else if (arg == "--no-crop") {
            crop = false;
        } else {
            std::cerr << "[Warning] Unrecognized command line argument: " << arg << "\n";
        }
    }

    if (version < 0) {
        std::cerr << "[Error] Missing or invalid --version flag.\n";
        print_usage(argv[0]);
        return 1;
    }
    const PipelineFn pipeline = select_version(version);
    if (!pipeline) {
        std::cerr << "[Error] Version " << version << " is not implemented yet.\n";
        return 1;
    }

    std::cout << "============================================================\n";
    std::cout << "  Left Image:     " << left_path << "\n";
    std::cout << "  Right Image:    " << right_path << "\n";
    std::cout << "  Output Image:   " << output_path << "\n";
    std::cout << "  Target Version: V" << version << "\n";
    std::cout << "  Repetitions:    " << runs << " (+1 warm-up run)\n";
    std::cout << "============================================================\n";

    // Query GPU hardware if running CUDA versions
#if defined(__CUDACC__)
    if (version >= 1) {
        print_cuda_device_info();
    }
#endif

    // Load input images (not timed)
    Image imgA, imgB, output;
    if (!load_image_rgb(left_path, imgA) || !load_image_rgb(right_path, imgB)) return 1;

    StitchConfig config;
    config.crop_to_covered = crop;
    std::string config_error;
    if (!validate_stitch_config(config, config_error)) {
        std::cerr << "[Error] Invalid stitch configuration: " << config_error << "\n";
        return 1;
    }

    // Warm-up run (not timed): prints the per-stage statistics once
    std::cout << "[Execution] Warm-up run of V" << version << "...\n";
    StageTimings warmup;
    if (!pipeline(imgA, imgB, output, warmup, config)) {
        std::cerr << "[Error] Stitching pipeline failed.\n";
        return 1;
    }

    // Timed runs
    config.verbose = false;
    std::vector<StageTimings> timed(static_cast<size_t>(runs));
    std::cout << "[Execution] " << runs << " timed run(s)...\n";
    for (int r = 0; r < runs; ++r) {
        if (!pipeline(imgA, imgB, output, timed[static_cast<size_t>(r)], config)) {
            std::cerr << "[Error] Stitching pipeline failed in timed run " << r + 1 << ".\n";
            return 1;
        }
    }
    std::cout << "Image size: " << imgA.width << "x" << imgA.height << " + " << imgB.width << "x" << imgB.height
              << ", version V" << version << ", " << runs << " run(s)";
    print_timing_table(timed);

    // Save output panorama (not timed)
    if (!save_image(output_path, output)) return 1;
    std::cout << "\n[Success] Panorama " << output.width << "x" << output.height << " saved to " << output_path << "\n";
    return 0;
}
