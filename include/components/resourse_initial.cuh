#pragma once

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

inline void init() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        spdlog::error("cudaGetDeviceCount failed: {}", cudaGetErrorString(err));
        return;
    }

    if (device_count == 0) {
        spdlog::warn("No CUDA devices detected.");
        return;
    }

    spdlog::info("Detected {} CUDA device(s).", device_count);
    for (int device = 0; device < device_count; ++device) {
        cudaDeviceProp prop{};
        err = cudaGetDeviceProperties(&prop, device);
        if (err != cudaSuccess) {
            spdlog::error("cudaGetDeviceProperties failed for device {}: {}",
                          device,
                          cudaGetErrorString(err));
            continue;
        }

        const size_t mem_mb = static_cast<size_t>(prop.totalGlobalMem / (1024 * 1024));
        const int clock_mhz = prop.clockRate / 1000;
        spdlog::info("GPU {}: {} (SM {}.{}, {} MB, {} MHz, {} SMs)",
                     device,
                     prop.name,
                     prop.major,
                     prop.minor,
                     mem_mb,
                     clock_mhz,
                     prop.multiProcessorCount);
    }
}
