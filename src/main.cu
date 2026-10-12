#include "common.h"
#include "versions.h"
#include "cuda_utils.cuh"

#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <iomanip>

void print_usage(const char* prog_name) {
    std::cout << "Usage:\n"
              << "  " << prog_name << " <left_image> <right_image> <output_image> --version <0|1|2|...> [--runs N]\n\n"
              << "Arguments:\n"
              << "  <left_image>     Path to reference image (Image A)\n"
              << "  <right_image>    Path to secondary image (Image B to be warped)\n"
              << "  <output_image>   Path for saving final panorama image\n"
              << "  --version <V>    Pipeline version (0: Sequential CPU, 1: Basic CUDA, ...)\n"
              << "  --runs <N>       Number of benchmark runs to average (default: 5)\n";
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

    // Parse options: --version <V> and optional --runs <N>
    for (int i = 4; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--version" && i + 1 < argc) {
            version = std::atoi(argv[++i]);
        } else if (arg == "--runs" && i + 1 < argc) {
            runs = std::max(1, std::atoi(argv[++i]));
        } else {
            std::cerr << "[Warning] Unrecognized command line argument: " << arg << "\n";
        }
    }

    if (version < 0) {
        std::cerr << "[Error] Missing or invalid --version flag.\n";
        print_usage(argv[0]);
        return 1;
    }

    std::cout << "============================================================\n";
    std::cout << "  Left Image:     " << left_path << "\n";
    std::cout << "  Right Image:    " << right_path << "\n";
    std::cout << "  Output Image:   " << output_path << "\n";
    std::cout << "  Target Version: V" << version << "\n";
    std::cout << "  Repetitions:    " << runs << "\n";
    std::cout << "============================================================\n";

    // Query GPU hardware if running CUDA versions
    if (version >= 1) {
        print_cuda_device_info();
    }

    // TODO: Load input images from disk using stb_image
    Image imgA, imgB, output;
    StageTimings timings;
    StitchConfig config;

    // Dispatch to requested pipeline version
    bool success = false;
    switch (version) {
        case 0:
            std::cout << "[Execution] Running Version 0 (Sequential Baseline)...\n";
            success = run_v0(imgA, imgB, output, timings, config);
            break;
        case 1:
            std::cout << "[Execution] Running Version 1 (Basic CUDA)...\n";
            // success = run_v1(imgA, imgB, output, timings, config);
            std::cerr << "[Info] Version 1 is not yet implemented.\n";
            return 1;
        default:
            std::cerr << "[Error] Unsupported version: " << version << "\n";
            return 1;
    }

    if (!success) {
        std::cerr << "[Error] Stitching pipeline failed.\n";
        return 1;
    }

    // TODO: Save output panorama using stb_image_write
    std::cout << "[Success] Stitching completed successfully.\n";
    return 0;
}
