// ============================================================================
// cuda_utils.cuh - CUDA error checks, event timer and device info (V1+)
//
// Author  : Dang Le Duc Thinh (23120360)
// Sources : interfaces from the CSC14120 project starter notebook;
//           no other external code.
// ============================================================================
#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>

// ============================================================================
// CUDA Error Checking Macro
// ============================================================================

#define CHECK_CUDA_ERROR(call)                                                 \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::fprintf(stderr, "[CUDA Error] %s at %s:%d (Code: %d)\n",      \
                         cudaGetErrorString(err), __FILE__, __LINE__, err);    \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

#define CHECK_LAST_CUDA_ERROR()                                               \
    do {                                                                       \
        cudaError_t err = cudaGetLastError();                                  \
        if (err != cudaSuccess) {                                              \
            std::fprintf(stderr, "[CUDA Kernel Error] %s at %s:%d (Code: %d)\n",\
                         cudaGetErrorString(err), __FILE__, __LINE__, err);    \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// ============================================================================
// CUDA Event-Based Timer
// ============================================================================

/**
 * High-precision timer using CUDA events for profiling kernel execution times.
 */
struct GpuTimer {
    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;

    GpuTimer() {
        CHECK_CUDA_ERROR(cudaEventCreate(&start_event));
        CHECK_CUDA_ERROR(cudaEventCreate(&stop_event));
    }

    ~GpuTimer() {
        if (start_event) cudaEventDestroy(start_event);
        if (stop_event) cudaEventDestroy(stop_event);
    }

    void start(cudaStream_t stream = 0) {
        CHECK_CUDA_ERROR(cudaEventRecord(start_event, stream));
    }

    void stop(cudaStream_t stream = 0) {
        CHECK_CUDA_ERROR(cudaEventRecord(stop_event, stream));
    }

    // Synchronizes and returns elapsed time in milliseconds
    float elapsed_ms() {
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop_event));
        float ms = 0.0f;
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&ms, start_event, stop_event));
        return ms;
    }
};

// ============================================================================
// GPU Device Query
// ============================================================================

inline void print_cuda_device_info() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        std::cout << "[Device Query] No CUDA-capable devices detected.\n";
        return;
    }

    int dev_id = 0;
    cudaDeviceProp prop;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, dev_id));

    std::cout << "============================================================\n";
    std::cout << "CUDA Device Information:\n";
    std::cout << "  Device Name:            " << prop.name << "\n";
    std::cout << "  Compute Capability:     " << prop.major << "." << prop.minor << "\n";
    std::cout << "  Total Global Memory:    " << (prop.totalGlobalMem / (1024 * 1024)) << " MB\n";
    std::cout << "  Multiprocessors (SMs):  " << prop.multiProcessorCount << "\n";
    std::cout << "  Warp Size:              " << prop.warpSize << "\n";
    std::cout << "  Max Threads Per Block:  " << prop.maxThreadsPerBlock << "\n";
    std::cout << "============================================================\n";
}

#endif // CUDA_UTILS_CUH
